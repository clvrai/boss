from ET.et_iql_fixed_embedding import ETIQLFixedEmbeddingModel
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn
import torch
import os
from gen import constants
import boss.models.cic_utils as utils
from ET.et_model import FeatureFlat

# import normflows as nf


class RMS(object):
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (
            self.S * self.n
            + torch.var(x, dim=0) * bs
            + (delta**2) * self.n * bs / (self.n + bs)
        ) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class APTArgs:
    def __init__(
        self,
        knn_k=16,
        knn_avg=True,
        rms=True,
        knn_clip=0.0005,
    ):
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.rms = rms
        self.knn_clip = knn_clip


class CIC(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, project_skill):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

        self.state_net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.skill_dim),
        )

        self.next_state_net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.skill_dim),
        )

        self.pred_net = nn.Sequential(
            nn.Linear(2 * self.skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.skill_dim),
        )

        if project_skill:
            self.skill_net = nn.Sequential(
                nn.Linear(self.skill_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.skill_dim),
            )
        else:
            self.skill_net = nn.Identity()

        self.apply(utils.weight_init)

    def forward(self, state, next_state, skill):
        assert len(state.size()) == len(next_state.size())
        state = self.state_net(state)
        next_state = self.state_net(next_state)
        query = self.skill_net(skill)
        key = self.pred_net(torch.cat([state, next_state], 1))
        return query, key


def compute_apt_reward(source, target, rms, args):

    b1, b2 = source.size(0), target.size(0)
    device = target.device
    # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
    sim_matrix = torch.norm(
        source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1),
        dim=-1,
        p=2,
    )
    reward, _ = sim_matrix.topk(
        args.knn_k, dim=1, largest=False, sorted=True
    )  # (b1, k)

    if not args.knn_avg:  # only keep k-th nearest neighbor
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(
            reward - args.knn_clip, torch.zeros_like(reward).to(device)
        )  # (b1, )
    else:  # average over all k nearest neighbors
        reward = reward.reshape(-1, 1)  # (b1 * k, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward).to(device))
        reward = reward.reshape((b1, args.knn_k))  # (b1, k)
        reward = reward.mean(dim=1)  # (b1,)
    reward = torch.log(reward + 1.0)
    return reward


class CICETIQLModel(ETIQLFixedEmbeddingModel):

    def __init__(self, args, pad=0, seg=1):
        """
        apt IQL agent
        """
        super().__init__(args, pad, seg)
        self.encoder = FeatureFlat(input_shape=(512, 7, 7), output_size=1024)
        # vocab_obj_path = os.path.join(constants.OBJ_CLS_VOCAB)
        # vocab_obj = torch.load(vocab_obj_path)
        # num_objects = len(vocab_obj)
        self.aug = utils.RandomShiftsAug(pad=1)
        self.hidden_dim = 1024
        self.icm_scale = 1.0
        self.knn_clip = 0.0
        self.knn_k = 12  # TODO: make a parameter
        self.knn_avg = True
        self.knn_rms = False
        self.skill_dim = 768
        args.device = torch.device(args.gpus[0])
        self.device = args.device
        self.rms = RMS(args.device)
        self.cic_temp = 0.5

        # TODO: maybe investigate projecting embedding space to smaller dim and using flow?

        # particle-based entropy from APT
        # self.pbe = utils.PBE(rms, self.knn_clip, self.knn_k, self.knn_avg, self.knn_rms, args.device)
        self.cic = CIC(1024, self.skill_dim, self.hidden_dim, True)
        self.cic_opt = torch.optim.Adam(
            list(self.cic.parameters()) + list(self.encoder.parameters()),
            lr=self.args.lr["final"],
        )
        self.cic.train()

    def sample_skill(self, eval=False):
        if eval:
            # selects mean skill of 0.5 (to select skill automatically use CEM or Grid Sweep
            # procedures described in the CIC paper)
            skill = np.ones(self.skill_dim).astype(np.float32) * 0.5
        else:
            skill = np.random.uniform(0, 1, self.skill_dim).astype(np.float32)
        return torch.from_numpy(skill).unsqueeze(0).to(self.device)

    def set_skill(self, skill):
        self.skill = skill

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        obs = self.encoder(obs)
        return obs

    def compute_cpc_loss(self, obs, next_obs, skill):
        temperature = self.cic_temp
        eps = 1e-6
        query, key = self.cic.forward(obs, next_obs, skill)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = torch.mm(query, key.T)  # (b,b)
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)  # (b,)
        row_sub = (
            torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        )
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(torch.sum(query * key, dim=-1) / temperature)  # (b,)
        loss = -torch.log(pos / (neg + eps))  # (b,)
        return loss, cov / temperature

    def update_cic(self, obs, skill, next_obs):
        metrics = dict()

        loss, logits = self.compute_cpc_loss(obs, next_obs, skill)
        loss = loss.mean()
        self.cic_opt.zero_grad()
        loss.backward()
        self.cic_opt.step()

        metrics["cic/loss"] = loss.item()
        metrics["cic/logits"] = logits.norm().item()

        return metrics

    @torch.no_grad()
    def compute_apt_reward(self, obs, next_obs):
        args = APTArgs()
        source = self.cic.state_net(obs)
        target = self.cic.state_net(next_obs)
        reward = compute_apt_reward(source, target, self.rms, args)  # (b,)
        return reward.unsqueeze(-1)  # (b,1)

    def train_offline_from_batch(
        self,
        frames,
        skill,
        action,
        obj_id,
        lengths_frames,
        interact_mask,
        rewards,
        terminals,
        eval=False,
    ):
        # all we have to do is modify the rewards being passed and then use that to run super().train_offline_from_batch
        batch_size = frames.shape[0]
        seq_length = action.shape[1] - 1
        skill_flattened = (
            skill.unsqueeze(1).repeat(1, seq_length, 1).reshape(-1, self.skill_dim)
        )
        obs_flattened = frames[:, :-1].reshape(-1, 512, 7, 7)
        next_obs_flattened = frames[:, 1:].reshape(-1, 512, 7, 7)
        # action_flattened = action.view(-1)
        # obj_id_flattened = obj_id.view(-1)
        with torch.no_grad():
            obs = self.aug_and_encode(obs_flattened)
            next_obs = self.aug_and_encode(next_obs_flattened)
        # update cic
        cic_metrics = self.update_cic(obs, skill_flattened, next_obs)
        metrics = {}
        metrics.update(cic_metrics)
        with torch.no_grad():
            intr_reward = self.compute_apt_reward(next_obs, next_obs)
            intr_reward = intr_reward.view(batch_size, seq_length)
        metrics["cic/intrinsic_reward"] = intr_reward.cpu().numpy()
        assert intr_reward.shape == rewards.shape
        # replace the language embedding with 0
        metrics.update(
            super().train_offline_from_batch(
                frames,
                skill,
                action,
                obj_id,
                lengths_frames,
                None,
                interact_mask,
                intr_reward,
                terminals,
                eval,
            )
        )
        return metrics

    def load_from_checkpoint(self, state_dicts):
        super().load_from_checkpoint(state_dicts)
        if "cic" in state_dicts:
            self.cic.load_state_dict(state_dicts["cic"])
            self.encoder.load_state_dict(state_dicts["cic_encoder"])
            self.cic_opt.load_state_dict(state_dicts["cic_opt"])

    def get_all_state_dicts(self):
        state_dicts = super().get_all_state_dicts()
        state_dicts.update(
            {
                "cic": self.cic.state_dict(),
                "cic_encoder": self.encoder.state_dict(),
                "cic_opt": self.cic_opt.state_dict(),
            }
        )
        return state_dicts
