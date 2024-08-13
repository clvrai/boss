from collections import defaultdict
import queue
import time
import numpy as np
import torch
import os
from torch import nn
from tqdm import tqdm
import argparse
import random
import wandb
from boss.alfred.models.nn.resnet import Resnet
import torch.multiprocessing as mp
import threading
import signal
import sys


from boss.models.boss_model import BOSSETIQLModel
from boss.models.saycan import SaycanPlanner
from boss.utils.utils import (
    AttrDict,
    str2bool,
    cleanup_mp,
)
from boss.utils.wandb_info import WANDB_PROJECT_NAME, WANDB_ENTITY_NAME
from boss.rollouts.saycan_rollout import run_policy_multi_process

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_mp(
    result_queue,
    task_queue,
    agent_model,
    saycan_planner,
    resnet,
    config,
    device,
):
    num_workers = config.num_rollout_workers
    workers = []
    # start workers
    worker_class = threading.Thread
    worker_target = run_policy_multi_process
    for _ in range(num_workers):
        worker = worker_class(
            target=worker_target,
            args=(
                result_queue,
                task_queue,
                config,
                device,
                agent_model,
                saycan_planner,
                resnet,
            ),
        )
        worker.daemon = True  # kills thread/process when parent thread terminates
        worker.start()
        time.sleep(0.5)
        workers.append(worker)
        num_tasks = result_queue.get()
    return workers, num_tasks


def multithread_dataset_aggregation(
    result_queue,
    rollout_returns,
    subgoal_successes,
    rollout_gifs,
    video_captions,
    extra_info,
    dataset,
    config,
    num_env_samples_list,
    eval,
):
    # asynchronously collect results from result_queue
    num_env_samples = 0
    num_finished_tasks = 0
    num_rollouts = config.num_eval_tasks if eval else config.num_rollouts_per_epoch
    with tqdm(total=num_rollouts) as pbar:
        while num_finished_tasks < num_rollouts:
            result = result_queue.get()
            rollout_returns.append(result["rews"].sum().item())
            subgoal_successes.append(result["dones"][-1])
            rollout_gifs.append(result["video_frames"])
            video_captions.append(result["video_caption"])
            extra_info["skill_lengths"].append(result["skill_length"])
            if "completed_skills" in result:
                extra_info["completed_skills"].append(result["completed_skills"])
            if "predicted_skills" in result:
                extra_info["predicted_skills"].append(result["predicted_skills"])
            if "high_level_skill" in result:
                extra_info["high_level_skill"].append(result["high_level_skill"])
            if "ground_truth_sequence" in result:
                extra_info["ground_truth_sequence"].append(
                    result["ground_truth_sequence"]
                )
            num_env_samples += result["rews"].shape[0]
            num_finished_tasks += 1
            pbar.update(1)
            if eval:
                pbar.set_description(
                    "EVAL: Finished %d/%d rollouts" % (num_finished_tasks, num_rollouts)
                )
            else:
                pbar.set_description(
                    "TRAIN: Finished %d/%d rollouts"
                    % (num_finished_tasks, num_rollouts)
                )
    num_env_samples_list.append(num_env_samples)


def multiprocess_rollout(
    task_queue,
    result_queue,
    config,
    epsilon,
    make_video,
):
    rollout_returns = []
    subgoal_successes = []
    rollout_gifs = []
    video_captions = []
    extra_info = defaultdict(list)

    num_rollouts = config.num_eval_tasks if eval else config.num_rollouts_per_epoch
    # create tasks for MP Queue

    # create tasks for thread/process Queue
    args_func = lambda subgoal: (True, True, epsilon, subgoal)

    for subgoal in range(num_rollouts):
        task_queue.put(args_func(subgoal))

    num_env_samples_list = []  # use list for thread safety
    multithread_dataset_aggregation(
        result_queue,
        rollout_returns,
        subgoal_successes,
        rollout_gifs,
        video_captions,
        extra_info,
        None,
        config,
        num_env_samples_list,
        True,
    )

    num_env_samples = num_env_samples_list[0]
    # aggregate metrics
    rollout_metrics = dict(
        eval_average_return=np.mean(rollout_returns),
        eval_subgoal_success=np.mean(subgoal_successes),
    )
    # make a WandB table for the high level skill, ground truth sequence, predicted, completed skills
    saycan_completed_skill_data = []
    keys = [
        "high_level_skill",
        "ground_truth_sequence",
        "predicted_skills",
        "completed_skills",
    ]
    for i in range(len(extra_info["high_level_skill"])):
        saycan_completed_skill_data.append([])
        for key in keys:
            saycan_completed_skill_data[-1].append(extra_info[key][i])

    skill_lengths = extra_info["skill_lengths"]
    per_number_return = defaultdict(list)
    per_number_success = defaultdict(list)
    for skill_length, success, rollout_return in zip(
        skill_lengths, subgoal_successes, rollout_returns
    ):
        per_number_return[skill_length].append(rollout_return)
        per_number_success[skill_length].append(success)
    for num_attempts, returns in per_number_return.items():
        rollout_metrics[f"length_{num_attempts}_return"] = np.mean(returns)
        rollout_metrics[f"length_{num_attempts}_success"] = np.mean(
            per_number_success[num_attempts]
        )

    if make_video:
        # sort both rollout_gifs and video_captions by the caption so that we have a consistent ordering
        rollout_gifs, video_captions = zip(
            *sorted(zip(rollout_gifs, video_captions), key=lambda x: x[1])
        )
        for i, (gif, caption) in enumerate(zip(rollout_gifs, video_captions)):
            rollout_metrics["videos_%d" % i] = wandb.Video(
                gif, caption=caption, fps=3, format="mp4"
            )
    table = wandb.Table(
        columns=[
            "High Level Skill",
            "Ground Truth Sequence",
            "Predicted Skills",
            "Completed Skills",
        ],
        data=saycan_completed_skill_data,
    )
    rollout_metrics["evaluation_table"] = table
    return rollout_metrics, num_env_samples


def main(config):
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    run = wandb.init(
        resume=config.experiment_name,
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY_NAME,
        notes=config.notes,
        config=config,
        group=config.run_group,
    )
    load_model_path = config.load_model_path
    checkpoint = torch.load(load_model_path, map_location="cpu")
    # load one of the checkpoints' configs

    old_config = checkpoint["config"]

    # overwrite only certain aspects of the config
    for key in vars(old_config):
        if (
            key not in vars(config)
            or vars(config)[key] is None
            and vars(old_config)[key] is not None
            and key != "experiment_name"
        ):
            vars(config)[key] = vars(old_config)[key]

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
    agent_model = BOSSETIQLModel(config)

    device = torch.device(config.gpus[0])
    if len(config.gpus) > 1:
        print(f"-----Using {len(config.gpus)} GPUs-----")
        agent_model = nn.DataParallel(
            agent_model,
            device_ids=config.gpus,
        )
    agent_model.to(device)

    agent_model.load_from_checkpoint(checkpoint)

    resnet_args = AttrDict()
    resnet_args.visual_model = "resnet18"
    resnet_args.gpu = config.gpus[0]
    resnet = Resnet(resnet_args, eval=True, use_conv_feat=True)

    saycan_planner = SaycanPlanner(config)

    # multiprocessed rollout setup
    task_queue = queue.SimpleQueue()
    result_queue = queue.SimpleQueue()

    processes, num_eval_tasks = setup_mp(
        result_queue,
        task_queue,
        agent_model,
        saycan_planner,
        resnet,
        config,
        device,
    )
    config.num_eval_tasks = num_eval_tasks

    def signal_handler(sig, frame):
        print("SIGINT received. Exiting...closing all processes first")
        cleanup_mp(task_queue, processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    eval_metrics, _ = multiprocess_rollout(
        task_queue,
        result_queue,
        config,
        0,
        True,
    )
    wandb.log(
        eval_metrics,
        step=0,
    )
    cleanup_mp(task_queue, processes)
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Autonomous exploration of an ALFRED IQL offline RL model for new language annotated skills"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Parent directory containing the dataset",
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
        "--num_rollout_workers",
        type=int,
        default=3,
        help="number of workers for policy rollouts",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="name of the experiment for logging on WandB",
    )
    parser.add_argument(
        "--notes", type=str, default="", help="optional notes for logging on WandB"
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        required=True,
        help="path to load the model from for SayCan eval from",
    )
    parser.add_argument(
        "--run_group",
        type=str,
        default=None,
        help="group to run the experiment in. If None, no group is used",
    )
    parser.add_argument(
        "--scene_type",
        type=str,
        default="valid_unseen",
        choices=["train", "valid_seen", "valid_unseen"],
        help="which type of scenes to sample from/evaluate on",
    )
    parser.add_argument(
        "--eval_json",
        type=str,
        default="boss/rollouts/evaluation_tasks.json",
        help="path to the json file containing the evaluation scenes and skills",
    )
    parser.add_argument(
        "--use_amp",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to use automatic mixed precision. set default to false to disable nans during online training.",
    )
    # LLM arguments
    parser.add_argument(
        "--llm_model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="which model to use for the large language model. defaults to llama-3-8b, paper used llama-1-13b",
        choices=[
            "None",
            "facebook/opt-125m",
            "facebook/opt-1.3b",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-7b-hf",
            "huggyllama/llama-13b",
        ],
    )
    parser.add_argument(
        "--llm_gpus",
        type=str,
        default="0",
        help="comma separated list of which gpus to use for the large language model",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=1,
        help="num concurrent queries for the LLM",
    )
    parser.add_argument(
        "--skill_match_with_dataset",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to match the skills with the dataset. if enabled, it's SayCan+P in the paper which performs better than regular SayCan. Otherwise this is just regular SayCan.",
    )
    parser.add_argument(
        "--specific_task",
        type=int,
        default=None,
        help="if specified, only train on this subgoal index",
    )  # for reporting the finetuning exps
    config = parser.parse_args()
    config.gpus = [int(gpu) for gpu in config.gpus.strip().split(",")]
    config.llm_gpus = [int(gpu) for gpu in config.llm_gpus.strip().split(",")]
    config.use_pretrained_lang = False
    if config.experiment_name is None and config.run_group is not None:
        config.experiment_name = f"{config.run_group}_{config.seed}"
    mp.set_sharing_strategy(
        "file_system"
    )  # to ensure the too many open files error doesn't happen with the dataloader
    main(config)
