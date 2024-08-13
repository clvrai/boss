import torch
from torch import nn
from torch.nn import functional as F
import collections

import numpy as np

from boss.models.enc_lang import EncoderLang
from boss.models.enc_vl import EncoderVL
from boss.models.encodings import DatasetLearnedEncoding
import boss.utils.model_utils as model_utils
from alfred.gen import constants

import os

from boss.utils.utils import extract_item

# This is pretty much extracted from the original codebase for Episodic Transformers, with some modifications


class BaseModel(nn.Module):
    def __init__(self, args, embs_ann, vocab_out, pad, seg):
        """
        Abstract model
        """
        nn.Module.__init__(self)
        self.args = args
        self.vocab_out = vocab_out
        self.pad, self.seg = pad, seg
        self.visual_tensor_shape = (512, 7, 7)

        # create language and action embeddings
        self.embs_ann = nn.ModuleDict({})
        for emb_name, emb_size in embs_ann.items():
            self.embs_ann[emb_name] = nn.Embedding(emb_size, args.demb)

        # dropouts
        self.dropout_vis = nn.Dropout(args.dropout["vis"], inplace=True)
        self.dropout_lang = nn.Dropout2d(args.dropout["lang"])

    def embed_lang(self, lang_pad, vocab):
        """
        take a list of annotation tokens and extract embeddings with EncoderLang
        """
        assert lang_pad.max().item() < len(vocab)
        embedder_lang = self.embs_ann[vocab.name]
        emb_lang, lengths_lang = self.encoder_lang(
            lang_pad, embedder_lang, vocab, self.pad
        )
        if self.args.detach_lang_emb:
            emb_lang = emb_lang.clone().detach()
        return emb_lang, lengths_lang

    def init_weights(self, init_range=0.1):
        """
        init linear layers in embeddings
        """
        for emb_ann in self.embs_ann.values():
            emb_ann.weight.data.uniform_(-init_range, init_range)

    def compute_metrics(self, model_out, gt_dict, metrics_dict, verbose):
        """
        compute model-specific metrics and put it to metrics dict
        """
        raise NotImplementedError

    def forward(self, vocab, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        raise NotImplementedError()

    def compute_batch_loss(self, model_out, gt_dict):
        """
        compute the loss function for a single batch
        """
        raise NotImplementedError()

    def compute_loss(self, model_outs, gt_dicts):
        """
        compute the loss function for several batches
        """
        # compute losses for each batch
        losses = {}
        for dataset_key in model_outs.keys():
            losses[dataset_key] = self.compute_batch_loss(
                model_outs[dataset_key], gt_dicts[dataset_key]
            )
        return losses


class ObjectClassifier(nn.Module):
    """
    object classifier module (a single FF layer)
    """

    def __init__(self, input_size):
        super().__init__()
        vocab_obj_path = os.path.join(constants.OBJ_CLS_VOCAB)
        vocab_obj = torch.load(vocab_obj_path)
        num_objects = len(vocab_obj)
        self.linear = nn.Linear(input_size, num_objects)

    def forward(self, x):
        out = self.linear(x)
        return out


# base episodic transformers model, for BC
class ETModel(BaseModel):
    def __init__(self, args, embs_ann, vocab_out, pad, seg):
        """
        transformer agent
        """
        self.vocab_word = torch.load(
            os.path.join(f"{os.environ['BOSS']}/boss/models/human.vocab")
        )  # vocab file for language annotations
        self.vocab_word["word"].name = "word"
        embs_ann = {"word": len(self.vocab_word["word"])}
        self.vocab_word["action_low"] = torch.load(
            f"{os.environ['BOSS']}/boss/models/low_level_actions.vocab"
        )[
            "action_low"
        ]  # our custom vocab
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
        # dataset id learned encoding (applied after the encoder_lang)
        self.dataset_enc = None
        if args.enc["dataset"]:
            self.dataset_enc = DatasetLearnedEncoding(args.demb, args.data["train"])
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
        self.reset()

        self.training_steps = 0

        self.optimizer, self.schedulers = model_utils.create_optimizer_and_schedulers(
            0, self.args, self.parameters(), None
        )

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
        emb_lang = self.dataset_enc(emb_lang, vocab) if self.dataset_enc else emb_lang

        # embed frames and actions
        emb_frames, emb_object = self.embed_frames(inputs["frames"])
        lengths_frames = inputs["lengths_frames"]
        emb_actions = self.embed_actions(inputs["action"])
        assert emb_frames.shape == emb_actions.shape
        lengths_actions = lengths_frames.clone()
        length_frames_max = inputs["length_frames_max"]

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

    def train_offline_from_batch(
        self,
        frames,
        lang,
        action,
        obj_id,
        lengths_frames,
        lengths_lang,
        subgoals_completed,
        goal_progress,
        interact_mask,
    ):
        outs = self.forward(
            vocab=self.vocab_word["word"],
            lang=lang,
            lengths_lang=lengths_lang,
            frames=frames,
            lengths_frames=lengths_frames,
            length_frames_max=lengths_frames.max().item(),
            action=action,
        )
        self.training_steps += 1
        gt_dict = {
            "action": action,
            "object": obj_id,
            "subgoals_completed": subgoals_completed,
            "goal_progress": goal_progress,
            "action_valid_interact": interact_mask,
        }
        # compute losses
        losses_train = self.compute_batch_loss(outs, gt_dict)

        # do the gradient step
        self.optimizer.zero_grad()
        sum_loss = sum([loss for name, loss in losses_train.items()])
        sum_loss.backward()
        self.optimizer.step()

        metrics = collections.defaultdict(list)
        # compute metrics
        self.compute_metrics(
            outs,
            gt_dict,
            metrics,
        )
        total_loss = sum_loss.detach().cpu().item()
        losses_train["total_loss"] = total_loss
        for key, value in losses_train.items():
            losses_train[key] = extract_item(value)
        for key, value in metrics.items():
            new_key = f"metrics_{key.replace('/', '_')}"
            losses_train[new_key] = extract_item(value[0])
        return losses_train

    def get_all_state_dicts(self):
        return {
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "schedulers": {
                key: scheduler.state_dict() if scheduler is not None else None
                for key, scheduler in self.schedulers.items()
            },
        }

    def load_all_state_dicts(self, state_dicts):
        self.load_state_dict(state_dicts["model"])
        self.optimizer.load_state_dict(state_dicts["optimizer"])
        for scheduler, state_dict in state_dicts["schedulers"].items():
            if state_dict is not None:
                self.schedulers[scheduler].load_state_dict(state_dict)

    def embed_lang(self, lang_pad, vocab):
        """
        take a list of annotation tokens and extract embeddings with EncoderLang
        """
        assert lang_pad.max().item() < len(vocab)
        embedder_lang = self.embs_ann[vocab.name]
        emb_lang, lengths_lang = self.encoder_lang(
            lang_pad, embedder_lang, vocab, self.pad
        )
        if self.args.detach_lang_emb:
            emb_lang = emb_lang.clone().detach()
        return emb_lang, lengths_lang

    def embed_frames(self, frames_pad):
        """
        take a list of frames tensors, pad it, apply dropout and extract embeddings
        """
        self.dropout_vis(frames_pad)
        frames_4d = frames_pad.view(-1, *frames_pad.shape[2:])
        frames_pad_emb = self.vis_feat(frames_4d).view(*frames_pad.shape[:2], -1)
        frames_pad_emb_skip = self.object_feat(frames_4d).view(
            *frames_pad.shape[:2], -1
        )
        return frames_pad_emb, frames_pad_emb_skip

    def embed_actions(self, actions):
        """
        embed previous actions
        """
        emb_actions = self.emb_action(actions)
        emb_actions = self.dropout_action(emb_actions)
        return emb_actions

    def reset(self):
        """
        reset internal states (used for real-time execution during eval)
        """
        self.frames_traj = torch.zeros(1, 0, *self.visual_tensor_shape)
        self.action_traj = torch.zeros(1, 0).long()

    def step(self, input_dict):
        """
        forward the model for a single time-step (used for real-time execution during eval)
        """
        frames_traj = input_dict["frames_buffer"]
        lang = input_dict["language_latent"]
        # at timestep t we have t-1 prev actions so we should pad them
        action_traj_pad = torch.cat(
            (
                input_dict["action_traj"],
                torch.zeros((1, 1)).to(frames_traj.device).long() + self.pad,
            ),
            dim=1,
        )
        model_out = self.forward(
            vocab=self.vocab_word["word"],
            lang=lang,
            lengths_lang=torch.tensor([lang.shape[1]]),
            frames=frames_traj.clone(),
            lengths_frames=torch.tensor([frames_traj.size(1)]),
            length_frames_max=frames_traj.size(1),
            action=action_traj_pad,
        )
        step_out = {}
        for key, value in model_out.items():
            # return only the last actions, ignore the rest
            step_out[key] = value[:, -1:]
        return step_out["action"].squeeze(0), step_out["object"].squeeze(0)

    def compute_batch_loss(self, model_out, gt_dict):
        """object"].squeeze(0)
        eq2Seq agent
        """
        losses = dict()

        # action loss
        action_pred = model_out["action"].view(-1, model_out["action"].shape[-1])
        action_gt = gt_dict["action"].view(-1)
        pad_mask = action_gt != self.pad
        action_loss = F.cross_entropy(action_pred, action_gt, reduction="none")
        action_loss *= pad_mask.float()
        action_loss = action_loss.mean()
        losses["action"] = action_loss * self.args.action_loss_wt

        # object classes loss
        object_pred = model_out["object"]
        object_gt = gt_dict["object"]
        interact_idxs = (
            gt_dict["action_valid_interact"].view(-1).nonzero(as_tuple=False).view(-1)
        )
        if interact_idxs.nelement() > 0:
            object_pred = object_pred.view(
                object_pred.shape[0] * object_pred.shape[1], *object_pred.shape[2:]
            )
            object_gt = object_gt.view(object_gt.shape[0] * object_gt.shape[1])
            object_loss = model_utils.obj_classes_loss(
                object_pred, object_gt, interact_idxs
            )
            losses["object"] = object_loss * self.args.object_loss_wt

        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            subgoal_pred = model_out["subgoal"].squeeze(2)
            subgoal_gt = gt_dict["subgoals_completed"]
            subgoal_loss = F.mse_loss(subgoal_pred, subgoal_gt, reduction="none")
            subgoal_loss = subgoal_loss.view(-1) * pad_mask.float()
            subgoal_loss = subgoal_loss.mean()
            losses["subgoal_aux"] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.progress_aux_loss_wt > 0:
            progress_pred = model_out["progress"].squeeze(2)
            progress_gt = gt_dict["goal_progress"]
            progress_loss = F.mse_loss(progress_pred, progress_gt, reduction="none")
            progress_loss = progress_loss.view(-1) * pad_mask.float()
            progress_loss = progress_loss.mean()
            losses["progress_aux"] = self.args.progress_aux_loss_wt * progress_loss

        # maximize entropy of the policy if asked
        if self.args.entropy_wt > 0.0:
            policy_entropy = -F.softmax(action_pred, dim=1) * F.log_softmax(
                action_pred, dim=1
            )
            policy_entropy = policy_entropy.mean(dim=1)
            policy_entropy *= pad_mask.float()
            losses["entropy"] = -policy_entropy.mean() * self.args.entropy_wt

        return losses

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
        gt_actions = model_utils.tokens_to_lang(
            gt_dict["action"], self.vocab_out, {self.pad}, offset=2
        )
        model_utils.compute_f1_and_exact(
            metrics_dict, [p["action"] for p in preds], gt_actions, "action"
        )
        model_utils.compute_obj_class_precision(
            metrics_dict, gt_dict, model_out["object"]
        )


class FeatureFlat(nn.Module):
    """
    a few conv layers to flatten features that come out of ResNet
    """

    def __init__(self, input_shape, output_size):
        super().__init__()
        if input_shape[0] == -1:
            input_shape = input_shape[1:]
        layers, activation_shape = self.init_cnn(
            input_shape, channels=[256, 64], kernels=[1, 1], paddings=[0, 0]
        )
        layers += [
            model_utils.Flatten(),
            nn.Linear(np.prod(activation_shape), output_size),
        ]
        self.layers = nn.Sequential(*layers)

    def init_cnn(self, input_shape, channels, kernels, paddings):
        layers = []
        planes_in, spatial = input_shape[0], input_shape[-1]
        for planes_out, kernel, padding in zip(channels, kernels, paddings):
            # do not use striding
            stride = 1
            layers += [
                nn.Conv2d(
                    planes_in,
                    planes_out,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(planes_out),
                nn.ReLU(inplace=True),
            ]
            planes_in = planes_out
            spatial = (spatial - kernel + 2 * padding) // stride + 1
        activation_shape = (planes_in, spatial, spatial)
        return layers, activation_shape

    def forward(self, frames):
        activation = self.layers(frames)
        return activation


# minor modifications so that it can use pretrained language annotations and don't use dropout_vis
class ETOfflineRLBaseModel(BaseModel):

    def embed_lang(self, lang_pad, vocab):
        """
        take a list of annotation tokens and extract embeddings with EncoderLang
        """
        if hasattr(self, "use_pretrained_lang") and self.use_pretrained_lang:
            with torch.no_grad():
                emb_lang = self.encoder_lang.encode(
                    lang_pad,
                    convert_to_tensor=True,
                    batch_size=lang_pad.shape[0],
                )
                lengths_lang = torch.ones(lang_pad.shape[0], dtype=torch.long)
            return emb_lang, lengths_lang
        return super().embed_lang(lang_pad, vocab)

    def embed_frames(self, frames_pad):
        """
        take a list of frames tensors, pad it, apply dropout and extract embeddings
        """
        # self.dropout_vis(frames_pad) removed this for some reason
        frames_4d = frames_pad.reshape(-1, *frames_pad.shape[2:])
        frames_pad_emb = self.vis_feat(frames_4d).reshape(*frames_pad.shape[:2], -1)
        frames_pad_emb_skip = None
        if hasattr(self, "object_feat"):
            frames_pad_emb_skip = self.object_feat(frames_4d).reshape(
                *frames_pad.shape[:2], -1
            )
        return frames_pad_emb, frames_pad_emb_skip
