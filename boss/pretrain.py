from code import interact
import numpy as np
import torch
import os
from boss.dataloaders.boss_dataloader import BOSSDataset, collate_func
from boss.models.boss_model import BOSSETIQLModel
from boss.utils.wandb_info import WANDB_ENTITY_NAME, WANDB_PROJECT_NAME
from boss.utils.utils import (
    AttrDict,
    send_to_device_if_not_none,
    extract_item,
    str2bool,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import random
import wandb


def evaluate_val_dataloader(agent_model, dataloader, device):
    agent_model.eval()
    ret_dict = {}
    with torch.no_grad():
        running_tracker_dict = {}
        for _, data_dict in enumerate(dataloader):
            frames = send_to_device_if_not_none(data_dict, "skill_feature", device)
            action = send_to_device_if_not_none(data_dict, "low_action", device).long()
            obj_id = send_to_device_if_not_none(data_dict, "object_ids", device).long()
            interact_mask = send_to_device_if_not_none(
                data_dict, "valid_interact", device
            )
            lengths_frames = send_to_device_if_not_none(
                data_dict, "feature_length", device
            )
            lengths_lang = send_to_device_if_not_none(data_dict, "token_length", device)
            lang = send_to_device_if_not_none(data_dict, "ann_token", device).int()
            word_lang = data_dict["annotation"]  # testing
            rewards = send_to_device_if_not_none(data_dict, "reward", device)
            terminals = send_to_device_if_not_none(data_dict, "terminal", device)

            # interact_mask = segmentation_loss_mask == 1
            loss_info = agent_model.train_offline_from_batch(
                frames,
                lang,
                action,
                obj_id,
                lengths_frames,
                lengths_lang,
                interact_mask,
                rewards,
                terminals,
                eval=True,
            )
            policy_loss = loss_info["policy_total_loss"]
            vf_loss = loss_info["vf_loss"]

            for k, v in loss_info.items():
                if k not in running_tracker_dict:
                    running_tracker_dict[k] = []
                running_tracker_dict[k].append(v)

    for k, v in running_tracker_dict.items():
        if isinstance(v[0], torch.Tensor):
            ret_dict[k] = torch.mean(torch.stack(v))
        else:
            ret_dict[k] = np.mean(v)
    return ret_dict


def train_epoch(agent_model, dataloader, epoch, device, save_dir, experiment_name):
    agent_model.train()
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch_i, data_dict in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            frames = send_to_device_if_not_none(data_dict, "skill_feature", device)
            goal_frames = send_to_device_if_not_none(data_dict, "goal_feature", device)
            action = send_to_device_if_not_none(data_dict, "low_action", device).long()
            obj_id = send_to_device_if_not_none(data_dict, "object_ids", device).long()
            interact_mask = send_to_device_if_not_none(
                data_dict, "valid_interact", device
            )
            lengths_frames = send_to_device_if_not_none(
                data_dict, "feature_length", device
            )
            lengths_lang = send_to_device_if_not_none(data_dict, "token_length", device)
            lang = send_to_device_if_not_none(data_dict, "ann_token", device).int()
            lang_list = data_dict["ann_token_list"]
            word_lang = data_dict["annotation"]  # testing
            rewards = send_to_device_if_not_none(data_dict, "reward", device)
            terminals = send_to_device_if_not_none(data_dict, "terminal", device)

            # interact_mask = segmentation_loss_mask == 1
            loss_info = agent_model.train_offline_from_batch(
                frames,
                lang,
                action,
                obj_id,
                lengths_frames,
                lengths_lang,
                interact_mask,
                rewards,
                terminals,
            )
            policy_loss = loss_info["policy_total_loss"]
            vf_loss = loss_info["vf_loss"]

            tepoch.set_postfix(
                policy_loss=policy_loss,
                vf_loss=vf_loss,
            )

            stuff_to_log = dict(
                batch_i=batch_i,
                **loss_info,
            )
            wandb.log(stuff_to_log)
        model_state_dict = agent_model.get_all_state_dicts()
        model_state_dict.update(dict(epoch=epoch))
        if (
            config.save_frequency
            and (epoch + 1) % config.save_frequency == 0
            or (epoch + 1) == config.epochs
        ):
            torch.save(
                model_state_dict,
                os.path.join(
                    save_dir,
                    experiment_name if experiment_name else wandb.run.name,
                    f"alfred_action_model_{epoch}.pth",
                ),
            )
    agent_model.adjust_lr(config, epoch)


def main(config):
    load_model_path = config.load_model_path

    if load_model_path:
        checkpoint = torch.load(load_model_path, map_location="cpu")
        start_epoch = checkpoint["epoch"] + 1
        if "config" in checkpoint:
            old_config = checkpoint["config"]
            # overwrite only certain aspects of the config
            overwrite_set = set(
                [
                    "num_workers",
                    "batch_size",
                    "experiment_name",
                    "save_dir",
                    "lr",
                    "max_skill_length",
                    "obj_pred_loss_coef",
                    "progress_pred_coef",
                    "weight_decay",
                ]
            )
            for key in overwrite_set:
                if key in vars(old_config):
                    vars(config)[key] = vars(old_config)[key]
    else:
        start_epoch = 0

    run = wandb.init(
        resume=config.experiment_name,
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY_NAME,
        notes=config.notes,
        config=config,
        group=config.run_group,
    )
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    random.seed(seed)
    torch.backends.cudnn.benchmark = False

    os.makedirs(
        os.path.join(
            config.save_dir,
            (
                config.experiment_name
                if config.experiment_name is not None
                else wandb.run.name
            ),
        ),
        exist_ok=True,
    )

    # instantiate the model and datasets
    device = torch.device(config.gpus[0])
    dataset = BOSSDataset(
        config.data_dir,
        "train",
        sample_primitive_skill=True,
        max_skill_length=config.max_skill_length,
    )
    valid_seen_dataset = BOSSDataset(
        config.data_dir + "_valid_seen",
        "valid_seen",
        sample_primitive_skill=True,
        max_skill_length=config.max_skill_length,
        use_full_skill=True,
    )
    valid_unseen_dataset = BOSSDataset(
        config.data_dir + "_valid_unseen",
        "valid_unseen",
        sample_primitive_skill=True,
        max_skill_length=config.max_skill_length,
        use_full_skill=True,
    )
    agent_model = BOSSETIQLModel(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_func,
    )
    valid_seen_dataloader = DataLoader(
        valid_seen_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=min(4, config.num_workers),
        pin_memory=True,
        collate_fn=collate_func,
    )
    valid_unseen_dataloader = DataLoader(
        valid_unseen_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=min(4, config.num_workers),
        pin_memory=True,
        collate_fn=collate_func,
    )

    if len(config.gpus) > 1:
        print(f"-----Using {len(config.gpus)} GPUs-----")
        agent_model = nn.DataParallel(
            agent_model,
            device_ids=config.gpus,
        )
        agent_model.train_offline_from_batch = (
            agent_model.module.train_offline_from_batch
        )
    agent_model.to(device)

    if load_model_path:
        agent_model.load_from_checkpoint(checkpoint)

    print(agent_model)

    for epoch in range(start_epoch, config.epochs):
        train_epoch(
            agent_model,
            dataloader,
            epoch,
            device,
            config.save_dir,
            config.experiment_name,
        )
        for name, valid_dataloader in zip(
            ("seen", "unseen"), (valid_seen_dataloader, valid_unseen_dataloader)
        ):
            ret_dict = evaluate_val_dataloader(
                agent_model,
                valid_dataloader,
                device,
            )
            eval_metrics = {}
            for k, metric in ret_dict.items():
                eval_metrics[f"{k}_valid_{name}"] = extract_item(metric)
            wandb.log(eval_metrics)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ALFRED IQL offline RL model on a fixed dataset of low-level, language annotated skills"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/boss_offline_dataset/boss_offline_dataset",
        help="Parent directory containing the dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models/",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="learning rate"
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="path to load the model from to continue training",
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="number of epochs for training"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="which gpus. pass in as comma separated string to use DataParallel on multiple GPUs",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers for data loading"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="name of the experiment for logging on WandB",
    )
    parser.add_argument(
        "--run_group",
        type=str,
        default=None,
        required=True,
        help="name of the GROUP for grouping on WandB. This is used to group runs together for analysis",
    )
    parser.add_argument(
        "--use_amp",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="use automatic mixed precision training",
    )
    parser.add_argument(
        "--soft_target_tau",
        type=float,
        default=0.005,
        help="soft target update coefficient",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.97,
        help="discount factor for the reward",
    )
    parser.add_argument(
        "--notes", type=str, default="", help="optional notes for logging on WandB"
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=50,
        help="number of epochs between saving the model",
    )
    parser.add_argument(
        "--max_skill_length",
        type=int,
        default=21,
        help="maximum number of frames/actions in a skill",
    )
    parser.add_argument(
        "--clip_score",
        type=float,
        default=100,
        help="max to clip the advantage to",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.8,
        help="quantile for computing the reward",
    )
    parser.add_argument(
        "--advantage_temp",
        type=float,
        default=5.0,
        help="temperature for computing the advantage",
    )
    parser.add_argument(
        "--use_lang_embedding",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="Whether to use pre-trained language embeddings as opposed to tokens",
    )
    parser.add_argument(
        "--train_with_advantage",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to pretrain with the advantage weighted bc loss (IQL)",
    )

    base_config = parser.parse_args()
    base_config.gpus = [int(gpu) for gpu in base_config.gpus.strip().split(",")]

    config = AttrDict()

    # DEFAULT HYPER PARAMETERS FOR EPISODIC TRANSFORMERS
    # batch size
    # optimizer type, must be in ('adam', 'adamw')
    config.optimizer = "adamw"
    # L2 regularization weight
    config.weight_decay = 0.1  # 0.33#0.1#0.33
    # num epochs
    # config.epochs = 54
    # learning rate settings
    config.lr = {
        # learning rate initial value
        # "init": 1e-4,
        "init": base_config.learning_rate,
        # lr scheduler type: {'linear', 'cosine', 'triangular', 'triangular2'}
        "profile": "linear",
        # (LINEAR PROFILE) num epoch to adjust learning rate
        # "decay_epoch": 10,
        "decay_epoch": 1,
        # (LINEAR PROFILE) scaling multiplier at each milestone
        # "decay_scale": 0.1,
        "decay_scale": 1.0,  # this basically forces no decay
        # (COSINE & TRIANGULAR PROFILE) learning rate final value
        "final": base_config.learning_rate,
        # (TRIANGULAR PROFILE) period of the cycle to increase the learning rate
        "cycle_epoch_up": 0,
        # (TRIANGULAR PROFILE) period of the cycle to decrease the learning rate
        "cycle_epoch_down": 0,
        # warm up period length in epochs
        "warmup_epoch": 0,
        # initial learning rate will be divided by this value
        "warmup_scale": 1,
    }
    # weight of action loss
    config.action_loss_wt = 1.0
    # weight of object loss
    config.object_loss_wt = 1.0

    # TRANSFORMER settings
    # size of transformer embeddings
    config.demb = 768
    # number of heads in multi-head attention
    config.encoder_heads = 12
    # number of layers in transformer encoder
    config.encoder_layers = 2
    # how many previous actions to use as input
    config.num_input_actions = 1
    # which encoder to use for language encoder (by default no encoder)
    config.encoder_lang = {
        "shared": True,
        "layers": 2,
        "pos_enc": True,
        "instr_enc": False,
    }
    # which decoder to use for the speaker model
    config.decoder_lang = {
        "layers": 2,
        "heads": 12,
        "demb": 768,
        "dropout": 0.1,
        "pos_enc": True,
    }
    # do not propagate gradients to the look-up table and the language encoder
    config.detach_lang_emb = False

    # DROPOUTS
    config.dropout = {
        # dropout rate for language (goal + instr)
        "lang": 0.0,
        # dropout rate for Resnet feats
        # "vis": 0.3,
        "vis": 0.1,
        # dropout rate for processed lang and visual embeddings
        "emb": 0.0,
        # transformer model specific dropouts
        "transformer": {
            # dropout for transformer encoder
            "encoder": 0.1,
            # remove previous actions
            "action": 0.0,
        },
    }

    # ENCODINGS
    config.enc = {
        # use positional encoding
        "pos": True,
        # use learned positional encoding
        "pos_learn": False,
        # use learned token ([WORD] or [IMG]) encoding
        "token": False,
        # dataset id learned encoding
        "dataset": False,
    }

    config.update(vars(base_config))

    main(config)
