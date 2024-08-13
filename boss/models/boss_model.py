import torch
from torch import nn
from torch.nn import functional as F
import collections

from boss.models.enc_lang import EncoderLang
import numpy as np

from boss.models.enc_vl import EncoderVL
import boss.utils.model_utils as model_utils
import boss.utils.data_utils as data_util
from boss.models.base_model import FeatureFlat, ObjectClassifier, ETOfflineRLBaseModel
from boss.alfred.gen import constants

from torch.nn.utils.rnn import pad_sequence

import os
import copy


class ETIQLPolicy(ETOfflineRLBaseModel):
    def __init__(self, args, vocab_word, pad=0, seg=1):
        """
        transformer IQL Policy
        """
        self.vocab_word = vocab_word
        embs_ann = {"word": len(self.vocab_word["word"])}
        vocab_out = self.vocab_word["action_low"]
        super().__init__(args, embs_ann, vocab_out, pad, seg)
        # encoder and visual embeddings
        self.encoder_vl = EncoderVL(args)
        # pre-encoder for language tokens
        self.encoder_lang = EncoderLang(args.encoder_lang["layers"], args, embs_ann)
        # feature embeddings
        self.vis_feat = FeatureFlat(
            input_shape=self.visual_tensor_shape, output_size=args.demb
        )
        # embeddings for actions. 12 actions + 1 for padding
        self.emb_action = nn.Embedding(13, args.demb)
        # dropouts
        self.dropout_action = nn.Dropout2d(args.dropout["transformer"]["action"])

        # decoder parts
        encoder_output_size = args.demb
        self.dec_action = nn.Linear(encoder_output_size, args.demb)
        self.dec_object = ObjectClassifier(encoder_output_size)

        # skip connection for object predictions
        self.object_feat = FeatureFlat(
            input_shape=self.visual_tensor_shape, output_size=args.demb
        )

        # final touch
        self.init_weights()

        self.optimizer, self.schedulers = model_utils.create_optimizer_and_schedulers(
            0, self.args, self.parameters(), None
        )

    def adjust_lr(self, config, epoch):
        model_utils.adjust_lr(self.optimizer, config, epoch, self.schedulers)

    def init_weights(self, init_range=0.1):
        """
        init linear layers in embeddings
        """
        for emb_ann in self.embs_ann.values():
            emb_ann.weight.data.uniform_(-init_range, init_range)

    def forward(self, vocab, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # embed language
        output = {}
        emb_lang, lengths_lang = self.embed_lang(inputs["lang"], vocab)

        # embed frames and actions
        # -1 here because frames includes the last frame which is not used for prediction
        emb_frames, emb_object = self.embed_frames(inputs["frames"][:, :-1])
        lengths_frames = inputs["lengths_frames"] - 1
        emb_actions = self.embed_actions(inputs["action"][:, :-1])
        assert emb_frames.shape == emb_actions.shape
        lengths_actions = lengths_frames.clone()
        length_frames_max = inputs["length_frames_max"] - 1

        # concatenate language, frames and actions and add encodings
        encoder_out, _ = self.encoder_vl(
            emb_lang,
            emb_frames,
            emb_actions,
            lengths_lang,
            lengths_frames,
            lengths_actions,
            length_frames_max,
        )
        # use outputs corresponding to visual frames for prediction only
        encoder_out_visual = encoder_out[
            :, lengths_lang.max().item() : lengths_lang.max().item() + length_frames_max
        ]

        # get the output actions
        decoder_input = encoder_out_visual.reshape(-1, self.args.demb)
        action_emb_flat = self.dec_action(decoder_input)
        # assert self.pad == 12, "pad should be 12 for following line to work to ignore the padding dimension in action_out"
        # action_flat = action_emb_flat.mm(self.emb_action.weight.t()[:, :self.pad])
        # nvm the above lines the model should learn to never output pad anyway.
        action_flat = action_emb_flat.mm(self.emb_action.weight.t())
        action = action_flat.view(*encoder_out_visual.shape[:2], *action_flat.shape[1:])

        # get the output objects
        emb_object_flat = emb_object.view(-1, self.args.demb)
        decoder_input = decoder_input + emb_object_flat
        object_flat = self.dec_object(decoder_input)
        objects = object_flat.view(
            *encoder_out_visual.shape[:2], *object_flat.shape[1:]
        )
        output.update({"action": action, "object": objects})
        return output

    def get_all_state_dicts(self):
        return {
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "schedulers": {
                key: scheduler.state_dict() if scheduler is not None else None
                for key, scheduler in self.schedulers.items()
            },
            "config": self.args,
        }

    def load_all_state_dicts(self, state_dicts):
        self.load_state_dict(state_dicts["model"])
        self.optimizer.load_state_dict(state_dicts["optimizer"])
        if "config" in state_dicts:
            self.args = state_dicts["config"]
        for scheduler, state_dict in state_dicts["schedulers"].items():
            if state_dict is not None:
                self.schedulers[scheduler].load_state_dict(state_dict)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def embed_actions(self, actions):
        """
        embed previous actions
        """
        emb_actions = self.emb_action(actions)
        emb_actions = self.dropout_action(emb_actions)
        return emb_actions

    def compute_batch_loss(self, model_out, weights, gt_dict):
        losses = dict()

        # action loss
        action_pred = model_out["action"].view(-1, model_out["action"].shape[-1])
        action_gt = gt_dict["action"].reshape(-1)
        pad_mask = action_gt != self.pad
        action_loss = F.cross_entropy(action_pred, action_gt, reduction="none")
        action_loss *= pad_mask.float() * weights
        action_loss = action_loss.mean()
        losses["policy_action_loss"] = action_loss

        # object classes loss
        object_pred = model_out["object"]
        # object_gt = torch.cat(gt_dict["object"], dim=0)
        object_gt = gt_dict["object"]
        interact_idxs = (
            gt_dict["action_valid_interact"].view(-1).nonzero(as_tuple=False).view(-1)
        )
        object_loss = 0
        if interact_idxs.nelement() > 0:
            object_pred = object_pred.view(
                object_pred.shape[0] * object_pred.shape[1], *object_pred.shape[2:]
            )
            object_gt = object_gt.reshape(object_gt.shape[0] * object_gt.shape[1])
            object_loss = model_utils.obj_classes_loss(
                object_pred, object_gt, interact_idxs, weights
            )
            losses["policy_object_loss"] = object_loss
        losses["policy_total_loss"] = action_loss + object_loss
        return losses

    def perform_model_update(self, losses, grad_scaler):
        self.optimizer.zero_grad()

        grad_scaler.scale(losses["policy_total_loss"]).backward()

        grad_scaler.step(self.optimizer)

    def init_weights(self, init_range=0.1):
        """
        init embeddings uniformly
        """
        super().init_weights(init_range)
        self.dec_action.bias.data.zero_()
        self.dec_action.weight.data.uniform_(-init_range, init_range)
        self.emb_action.weight.data.uniform_(-init_range, init_range)

    def compute_metrics(self, model_out, gt_dict, metrics_dict, verbose=False):
        """
        compute exact matching and f1 score for action predictions
        """
        preds = model_utils.extract_action_preds(
            model_out, self.pad, self.vocab_out, lang_only=True, offset=2
        )
        # stop_token = self.vocab_out.word2index("<<stop>>")
        # gt_actions = model_util.tokens_to_lang(
        #    gt_dict["action"], self.vocab_out, {self.pad, stop_token}, offset=3
        # )
        gt_actions = model_utils.tokens_to_lang(
            gt_dict["action"], self.vocab_out, {self.pad}, offset=2
        )
        model_utils.compute_f1_and_exact(
            metrics_dict, [p["action"] for p in preds], gt_actions, "action"
        )
        model_utils.compute_obj_class_precision(
            metrics_dict, gt_dict, model_out["object"]
        )


class ETIQLCritics(ETOfflineRLBaseModel):
    def __init__(self, args, vocab_word, pad=0, seg=1):
        """
        transformer IQL Critics, combined
        """
        self.vocab_word = vocab_word
        embs_ann = {"word": len(self.vocab_word["word"])}
        vocab_out = self.vocab_word["action_low"]
        super().__init__(args, embs_ann, vocab_out, pad, seg)
        # encoder and visual embeddings
        args.num_input_actions = (
            1  # attend to past action for V (to match policy) and curr action for Q
        )
        self.encoder_vl = EncoderVL(args)
        # pre-encoder for language tokens
        self.encoder_lang = EncoderLang(args.encoder_lang["layers"], args, embs_ann)
        # feature embeddings
        self.vis_feat = FeatureFlat(
            input_shape=self.visual_tensor_shape, output_size=args.demb
        )
        # embeddings for actions. 12 actions + 1 for padding
        self.emb_action = nn.Embedding(13, args.demb)
        # embeddings for object classes. 82 classes + 1 for padding
        vocab_obj_path = os.path.join(constants.OBJ_CLS_VOCAB)
        vocab_obj = torch.load(vocab_obj_path)
        num_objects = len(vocab_obj)
        self.emb_object = nn.Embedding(num_objects + 1, args.demb)
        self.action_object_embedding_combiner = nn.Linear(args.demb * 2, args.demb)
        # dropouts
        self.dropout_action = nn.Dropout2d(args.dropout["transformer"]["action"])

        base_model_params = (
            list(self.encoder_vl.parameters())
            + list(self.encoder_lang.parameters())
            + list(self.vis_feat.parameters())
            + list(self.emb_action.parameters())
            + list(self.emb_object.parameters())
            + list(self.action_object_embedding_combiner.parameters())
        )

        # q functions
        encoder_output_size = args.demb

        critic_gen = lambda: nn.Sequential(
            *(
                nn.Linear(encoder_output_size, args.demb),
                nn.LayerNorm(args.demb),
                nn.ReLU(),
                nn.Linear(args.demb, args.demb // 2),
                nn.LayerNorm(args.demb // 2),
                nn.ReLU(),
                nn.Linear(args.demb // 2, 1),
            )
        )
        self.qf1 = critic_gen()
        self.qf2 = critic_gen()
        self.vf = critic_gen()
        # final touch
        self.init_weights()
        self.optimizer, self.schedulers = model_utils.create_optimizer_and_schedulers(
            0,
            self.args,
            base_model_params
            + list(self.qf1.parameters())
            + list(self.qf2.parameters())
            + list(self.vf.parameters()),
            None,
        )

        self.mse_loss = nn.MSELoss()
        self.discount = args.discount

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def adjust_lr(self, config, epoch):
        model_utils.adjust_lr(self.optimizer, config, epoch, self.schedulers)

    def embed_actions(self, actions):
        """
        embed previous actions
        """
        emb_actions = self.emb_action(actions)
        emb_actions = self.dropout_action(emb_actions)
        return emb_actions

    def embed_and_combine_obj_and_action(self, obj, emb_actions):
        """
        embed object and action and combine them
        """
        emb_object = self.emb_object(obj)
        emb_object_action = torch.cat((emb_object, emb_actions), dim=2)
        emb_object_action = self.action_object_embedding_combiner(emb_object_action)
        return emb_object_action

    def init_weights(self, init_range=0.1):
        """
        init linear layers in embeddings
        """
        for emb_ann in self.embs_ann.values():
            emb_ann.weight.data.uniform_(-init_range, init_range)

    def get_value(self, frames, lang):
        """
        get just the value for a batch of data (used during real-time execution and only for first states in skills)
        """
        # at first state no previous actions
        action_traj_pad = (
            torch.zeros((lang.shape[0], 1)).to(lang.device).long() + self.pad
        )
        # same with object
        obj_traj_pad = torch.zeros((lang.shape[0], 1)).to(lang.device).long() + self.pad
        # pad frames so to work with assumption of having 1 more frame than action
        assert frames.size(1) == action_traj_pad.size(1)
        vocab = self.vocab_word["word"]
        # embed language
        emb_lang, lengths_lang = self.embed_lang(lang, vocab)

        # embed frames and actions
        emb_frames, _ = self.embed_frames(frames)
        lengths_frames = torch.ones_like(lengths_lang)
        emb_actions = self.embed_actions(action_traj_pad)
        emb_object_action = self.embed_and_combine_obj_and_action(
            obj_traj_pad,
            emb_actions,
        )
        # q_actions_input = inputs["action"][:, 1:]
        assert emb_frames.shape == emb_object_action.shape

        lengths_actions = lengths_frames.clone()
        length_frames_max = 1
        lengths_actions = lengths_frames.clone()
        length_frames_max = 1

        # concatenate language, frames and actions and add encodings
        v_val = None
        v_encoder_out, _ = self.encoder_vl(
            emb_lang,
            emb_frames,
            emb_object_action,
            lengths_lang,
            lengths_frames,
            lengths_actions,
            length_frames_max,
        )
        v_encoder_out_visual = v_encoder_out[
            :,
            lengths_lang.max().item() : lengths_lang.max().item() + length_frames_max,
        ]
        v_val = self.vf(v_encoder_out_visual.reshape(-1, self.args.demb)).view(-1)
        return v_val

    def get_sequential_q(self, vocab, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # embed language
        emb_lang, lengths_lang = self.embed_lang(inputs["lang"], vocab)

        # embed frames and actions
        emb_frames, _ = self.embed_frames(inputs["frames"])
        lengths_frames = inputs["lengths_frames"]
        emb_actions = self.embed_actions(inputs["action"])
        emb_object_action = self.embed_and_combine_obj_and_action(
            inputs["object"],
            emb_actions,
        )
        # q_actions_input = inputs["action"][:, 1:]
        assert emb_frames.shape == emb_object_action.shape

        lengths_actions = lengths_frames.clone()
        length_frames_max = inputs["length_frames_max"]

        # concatenate language, frames and actions and add encodings
        q_encoder_out, _ = self.encoder_vl(
            emb_lang,
            emb_frames[:, :-1],
            emb_object_action[
                :, 1:
            ],  # shift to use the action that was taken for each state
            lengths_lang,
            lengths_frames - 1,
            lengths_actions - 1,
            length_frames_max - 1,
        )
        # use outputs corresponding to visual frames for prediction only
        q_encoder_out_visual = q_encoder_out[
            :,
            lengths_lang.max().item() : lengths_lang.max().item()
            + length_frames_max
            - 1,
        ]

        # get the output actions
        qf_input = q_encoder_out_visual.reshape(-1, self.args.demb)
        sequential_output_shape = (emb_frames.shape[0], length_frames_max - 1, 1)
        q1_val = self.qf1(qf_input).view(*sequential_output_shape).squeeze(-1)
        q2_val = self.qf2(qf_input).view(*sequential_output_shape).squeeze(-1)
        return q1_val, q2_val

    def get_sequential_v(self, vocab, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # embed language
        output = {}
        emb_lang, lengths_lang = self.embed_lang(inputs["lang"], vocab)

        # embed frames and actions
        emb_frames, _ = self.embed_frames(inputs["frames"])
        lengths_frames = inputs["lengths_frames"]
        emb_actions = self.embed_actions(inputs["action"])
        emb_object_action = self.embed_and_combine_obj_and_action(
            inputs["object"],
            emb_actions,
        )
        # q_actions_input = inputs["action"][:, 1:]
        assert emb_frames.shape == emb_object_action.shape

        lengths_actions = lengths_frames.clone()
        length_frames_max = inputs["length_frames_max"]

        # concatenate language, frames and actions and add encodings
        v_val = None
        v_encoder_out, _ = self.encoder_vl(
            emb_lang,
            emb_frames,
            emb_object_action,
            lengths_lang,
            lengths_frames,
            lengths_actions,
            length_frames_max,
        )
        sequential_output_shape = (emb_frames.shape[0], length_frames_max - 1, 1)
        v_encoder_out_visual = v_encoder_out[
            :,
            lengths_lang.max().item() : lengths_lang.max().item() + length_frames_max,
        ]
        v_val = (
            self.vf(v_encoder_out_visual[:, :-1].reshape(-1, self.args.demb))
            .view(*sequential_output_shape)
            .squeeze(-1)
        )
        return v_val

    def forward(self, vocab, forward_v, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # embed language
        output = {}
        emb_lang, lengths_lang = self.embed_lang(inputs["lang"], vocab)

        # embed frames and actions
        emb_frames, _ = self.embed_frames(inputs["frames"])
        lengths_frames = inputs["lengths_frames"]
        emb_actions = self.embed_actions(inputs["action"])
        emb_object_action = self.embed_and_combine_obj_and_action(
            inputs["object"],
            emb_actions,
        )
        # q_actions_input = inputs["action"][:, 1:]
        assert emb_frames.shape == emb_object_action.shape

        lengths_actions = lengths_frames.clone()
        length_frames_max = inputs["length_frames_max"]

        # concatenate language, frames and actions and add encodings
        v_val = None
        next_v_val = None
        if forward_v:
            v_encoder_out, _ = self.encoder_vl(
                emb_lang,
                emb_frames,
                emb_object_action,
                lengths_lang,
                lengths_frames,
                lengths_actions,
                length_frames_max,
            )
            v_encoder_out_visual = v_encoder_out[
                :,
                lengths_lang.max().item() : lengths_lang.max().item()
                + length_frames_max,
            ]
            v_val = self.vf(
                v_encoder_out_visual[:, :-1].reshape(-1, self.args.demb)
            ).view(-1)
            next_v_val = self.vf(
                v_encoder_out_visual[:, 1:].reshape(-1, self.args.demb)
            ).view(-1)
        q_encoder_out, _ = self.encoder_vl(
            emb_lang,
            emb_frames[:, :-1],
            emb_object_action[
                :, 1:
            ],  # shift to use the action that was taken for each state
            lengths_lang,
            lengths_frames - 1,
            lengths_actions - 1,
            length_frames_max - 1,
        )
        # use outputs corresponding to visual frames for prediction only
        q_encoder_out_visual = q_encoder_out[
            :,
            lengths_lang.max().item() : lengths_lang.max().item()
            + length_frames_max
            - 1,
        ]

        # get the output actions
        qf_input = q_encoder_out_visual.reshape(-1, self.args.demb)
        q1_val = self.qf1(qf_input).view(-1)
        q2_val = self.qf2(qf_input).view(-1)

        output.update(
            {
                "q1_val": q1_val,
                "q2_val": q2_val,
                "next_v_val": next_v_val,
                "v_val": v_val,
            }
        )
        return output

    def get_all_state_dicts(self):
        return {
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "schedulers": {
                key: scheduler.state_dict() if scheduler is not None else None
                for key, scheduler in self.schedulers.items()
            },
            "config": self.args,
        }

    def load_all_state_dicts(self, state_dicts):
        self.load_state_dict(state_dicts["model"])
        self.optimizer.load_state_dict(state_dicts["optimizer"])
        if "config" in state_dicts:
            self.args = state_dicts["config"]
        for scheduler, state_dict in state_dicts["schedulers"].items():
            if state_dict is not None:
                self.schedulers[scheduler].load_state_dict(state_dict)

    def compute_batch_loss(self, model_out, rewards, terminals):
        losses = dict()
        rewards = rewards.reshape(-1)
        terminals = terminals.reshape(-1)
        pad_mask = terminals != -1
        target_q = (
            rewards[pad_mask]
            + (1.0 - terminals[pad_mask])
            * self.discount
            * model_out["next_v_val"][pad_mask]
        ).detach()
        # Compute the Q function loss
        q1_loss = self.mse_loss(model_out["q1_val"][pad_mask], target_q)
        q2_loss = self.mse_loss(model_out["q2_val"][pad_mask], target_q)
        losses["q1_loss"] = q1_loss
        losses["q2_loss"] = q2_loss

        # Compute the expectile regression value function loss
        vf_err = model_out["v_val"][pad_mask] - model_out["target_q_pred"][pad_mask]
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.args.quantile + vf_sign * (
            1 - self.args.quantile
        )
        vf_loss = (vf_weight * (vf_err**2)).mean()
        losses["vf_loss"] = vf_loss
        return losses

    def perform_model_update(self, losses, grad_scaler):
        self.optimizer.zero_grad()

        all_losses = losses["q1_loss"] + losses["q2_loss"] + losses["vf_loss"]
        grad_scaler.scale(all_losses).backward()

        grad_scaler.step(self.optimizer)

    def init_weights(self, init_range=0.1):
        """
        init embeddings uniformly
        """
        super().init_weights(init_range)
        self.emb_action.weight.data.uniform_(-init_range, init_range)
        self.emb_object.weight.data.uniform_(-init_range, init_range)

    def compute_metrics(self, model_out, gt_dict, metrics_dict, verbose=False):
        return


class BOSSETIQLModel(nn.Module):
    def __init__(self, args, pad=0, seg=1):
        """
        transformer IQL agent
        """
        super(BOSSETIQLModel, self).__init__()
        self.vocab_word = torch.load(
            os.path.join(f"{os.environ['BOSS']}/boss/models/human.vocab")
        )  # vocab file for language annotations
        # self.vocab_word["word"] = vocab_ann
        self.vocab_word["word"].name = "word"
        self.vocab_word["action_low"] = torch.load(
            f"{os.environ['BOSS']}/boss/models/low_level_actions.vocab"
        )[
            "action_low"
        ]  # our custom vocab
        self.policy = ETIQLPolicy(args, self.vocab_word, pad, seg)

        self.critics = ETIQLCritics(args, self.vocab_word, pad, seg)
        self.target_critics = copy.deepcopy(self.critics)
        self.training_steps = 0
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.use_amp = args.use_amp
        self.args = args
        self.pad = pad
        self.seg = seg
        self.train_with_advantage = args.train_with_advantage

    def accum_gradient_from_batch(
        self,
        frames,
        lang,
        action,
        obj_id,
        lengths_frames,
        lengths_lang,
        interact_mask,
        rewards,
        terminals,
        accum_factor,
    ):
        combined_outs = {}
        policy_outs = self.policy.forward(
            vocab=self.vocab_word["word"],
            lang=lang,
            lengths_lang=lengths_lang,
            frames=frames,
            lengths_frames=lengths_frames,
            length_frames_max=lengths_frames.max().item(),
            action=action,
        )
        combined_outs.update(policy_outs)
        critic_outs = self.critics.forward(
            vocab=self.vocab_word["word"],
            lang=lang,
            lengths_lang=lengths_lang,
            frames=frames,
            lengths_frames=lengths_frames,
            length_frames_max=lengths_frames.max().item(),
            action=action,
            forward_v=True,
            object=obj_id,
        )
        combined_outs.update(critic_outs)
        with torch.no_grad():
            target_critic_outs = self.target_critics.forward(
                vocab=self.vocab_word["word"],
                lang=lang,
                lengths_lang=lengths_lang,
                frames=frames,
                lengths_frames=lengths_frames,
                length_frames_max=lengths_frames.max().item(),
                action=action,
                forward_v=False,
                object=obj_id,
            )
        gt_dict = {
            "action": action[:, :-1],
            "object": obj_id[:, :-1],
            "action_valid_interact": interact_mask,
        }

        metrics = collections.defaultdict(list)
        # compute losses
        combined_outs.update({"target_" + k: v for k, v in target_critic_outs.items()})
        target_q_pred = torch.min(
            combined_outs["target_q1_val"], combined_outs["target_q2_val"]
        ).detach()
        combined_outs["target_q_pred"] = target_q_pred
        # advantage weights for IQL policy training
        advantage = target_q_pred - combined_outs["v_val"].detach()

        exp_advantage = torch.exp(advantage * self.args.advantage_temp)
        if self.args.clip_score is not None:
            exp_advantage = torch.clamp(exp_advantage, max=self.args.clip_score)

        if self.train_with_advantage:
            weights = exp_advantage.detach().squeeze(-1)
        else:
            weights = torch.ones_like(exp_advantage).detach().squeeze(-1)
        self.training_steps += 1
        policy_losses = self.policy.compute_batch_loss(combined_outs, weights, gt_dict)

        critic_losses = self.critics.compute_batch_loss(
            combined_outs, rewards, terminals
        )
        for key, value in policy_losses.items():
            policy_losses[key] = value * accum_factor
        for key, value in critic_losses.items():
            critic_losses[key] = value * accum_factor
        metrics.update(policy_losses)
        metrics.update(critic_losses)
        metrics["exp_advantage"] = exp_advantage.mean().item()
        metrics["exp_advantage_max"] = exp_advantage.max().item()
        metrics["exp_advantage_min"] = exp_advantage.min().item()

        # compute metrics
        self.policy.compute_metrics(
            combined_outs,
            gt_dict,
            metrics,
        )
        self.critics.compute_metrics(
            critic_outs,
            gt_dict,
            metrics,
        )
        for key, value in metrics.items():
            if type(value) is torch.Tensor:
                metrics[key] = torch.mean(value).item()
            else:
                metrics[key] = np.mean(value)
        for key, value in combined_outs.items():
            if value is not None:
                metrics[key] = value.mean().item()
        losses = {}
        losses.update(policy_losses)
        losses.update(critic_losses)
        return metrics, losses

    def train_offline_from_batch(
        self,
        frames,
        lang,
        action,
        obj_id,
        lengths_frames,
        lengths_lang,
        interact_mask,
        rewards,
        terminals,
        eval=False,
    ):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            in_traj_reward = rewards
            in_traj_terminal = terminals
            ret_dict, losses = self.accum_gradient_from_batch(
                frames,
                lang,
                action,
                obj_id,
                lengths_frames,
                lengths_lang,
                interact_mask.view(-1),
                in_traj_reward,
                in_traj_terminal,
                1.0,
            )
            # add losses to ret_dict
            for key, value in losses.items():
                ret_dict[key] = value.item()

            if not eval:
                self.policy.perform_model_update(losses, self.grad_scaler)
                self.critics.perform_model_update(losses, self.grad_scaler)
                self.grad_scaler.update()
                # soft target adjustment
                model_utils.soft_update_from_to(
                    self.critics, self.target_critics, self.args.soft_target_tau
                )
        return ret_dict

    def get_all_state_dicts(self):
        state_dicts = {
            "policy": self.policy.get_all_state_dicts(),
            "critics": self.critics.get_all_state_dicts(),
            "target_critics": self.target_critics.get_all_state_dicts(),
            "config": self.args,
        }
        return state_dicts

    def load_from_checkpoint(self, state_dicts):
        if "config" in state_dicts:
            self.args = state_dicts["config"]
        self.policy.load_all_state_dicts(state_dicts["policy"])
        self.critics.load_all_state_dicts(state_dicts["critics"])
        self.target_critics.load_all_state_dicts(state_dicts["target_critics"])

    def step(self, input_dict, ret_value):
        """
        forward the model for a single time-step (used for real-time execution during eval)
        """
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            lang = input_dict["language_ann"]
            # at timestep t we have t-1 prev actions so we should pad them, then pad 1 more for extra dummy action to match with frames length below
            action_traj_pad = torch.cat(
                (
                    input_dict["action_traj"],
                    torch.zeros((1, 2)).to(lang.device).long() + self.pad,
                ),
                dim=1,
            )
            # same with object
            obj_traj_pad = torch.cat(
                (
                    input_dict["object_traj"],
                    torch.zeros((1, 2)).to(lang.device).long() + self.pad,
                ),
                dim=1,
            )
            # pad frames so to work with assumption of having 1 more frame than action
            frames_traj_pad = torch.cat(
                (
                    input_dict["frames_buffer"],
                    torch.zeros((1, 1, 512, 7, 7)).to(lang.device) + self.pad,
                ),
                dim=1,
            )
            assert frames_traj_pad.size(1) == action_traj_pad.size(
                1
            ), f"{frames_traj_pad.size(1)} != {action_traj_pad.size(1)}"
            assert frames_traj_pad.size(1) == obj_traj_pad.size(
                1
            ), f"{frames_traj_pad.size(1)} != {obj_traj_pad.size(1)}"
            model_out = self.policy.forward(
                vocab=self.vocab_word["word"],
                lang=lang,
                lengths_lang=torch.tensor([lang.shape[1]]),
                frames=frames_traj_pad.clone(),
                lengths_frames=torch.tensor([frames_traj_pad.size(1)]),
                length_frames_max=frames_traj_pad.size(1),
                action=action_traj_pad,
            )
            step_out = {}
            for key, value in model_out.items():
                # return only the last actions, ignore the rest
                step_out[key] = value[:, -1:]
            value = None
            if ret_value:
                value = self.critics.forward(
                    vocab=self.vocab_word["word"],
                    lang=lang,
                    lengths_lang=torch.tensor([lang.shape[1]]),
                    frames=frames_traj_pad.clone(),
                    lengths_frames=torch.tensor([frames_traj_pad.size(1)]),
                    length_frames_max=frames_traj_pad.size(1),
                    action=action_traj_pad,
                    forward_v=True,
                    object=obj_traj_pad,
                )["v_val"][-1:]
            return (
                step_out["action"].squeeze(0),
                step_out["object"].squeeze(0),
                value.squeeze(0) if value is not None else value,
            )

    def adjust_lr(self, config, epoch):
        self.policy.adjust_lr(config, epoch)
        self.critics.adjust_lr(config, epoch)

    def set_lr(self, lr):
        self.policy.set_lr(lr)
        self.critics.set_lr(lr)
