from collections import defaultdict
import queue
import time
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
import os
from collections import Counter
from sentence_transformers import SentenceTransformer
from torch import nn
from tqdm import tqdm
import argparse
import random
import wandb
from torch.utils.data import DataLoader
from boss.alfred.models.nn.resnet import Resnet
import torch.multiprocessing as mp
import threading
import signal
import sys

from boss.models.boss_model import BOSSETIQLModel
from boss.models.large_language_model import LargeLanguageModel
from boss.dataloaders.boss_dataloader import RLBuffer, collate_func
from boss.utils.wandb_info import WANDB_PROJECT_NAME, WANDB_ENTITY_NAME
from boss.utils.data_utils import CombinedDataset, process_annotation_inference
from boss.utils.utils import (
    primitive_skill_types,
    AttrDict,
    log_rollout_metrics,
    make_bar_chart,
    send_to_device_if_not_none,
    str2bool,
)
from boss.rollouts.bootstrapping_rollout import (
    run_policy_multi_process as bootstrap_run_policy_multi_process,
)
from boss.rollouts.rollout import (
    run_policy_multi_process as no_bootstrap_run_policy_multi_process,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
EVAL_INTERVAL = 50000


class ETLanguageEncoder:
    def __init__(self, vocab_word):
        self.vocab_word = vocab_word

    def encode(self, annotations, convert_to_tensor=False):
        if convert_to_tensor:
            return pad_sequence(
                [process_annotation_inference(a, self.vocab_word) for a in annotations],
                batch_first=True,
                padding_value=0,
            )
        return [process_annotation_inference(a, self.vocab_word) for a in annotations]


def setup_mp(
    result_queue,
    task_queue,
    config,
    device,
    offline_rl_model,
    lang_embedding_model,
    semantic_search_model,
    llm,
    resnet,
):
    num_workers = config.num_rollout_workers
    workers = []
    # start workers
    worker_class = threading.Thread
    if config.no_bootstrap:
        worker_target = no_bootstrap_run_policy_multi_process
    else:
        worker_target = bootstrap_run_policy_multi_process
    for _ in range(num_workers):
        worker = worker_class(
            target=worker_target,
            args=(
                result_queue,
                task_queue,
                config,
                device,
                offline_rl_model,
                lang_embedding_model,
                semantic_search_model,
                llm,
                resnet,
            ),
        )
        worker.daemon = True  # kills thread/process when parent thread terminates
        worker.start()
        time.sleep(0.5)
        workers.append(worker)
        num_tasks = result_queue.get()
    return workers, num_tasks


def cleanup_mp(task_queue, processes):
    # generate termination signal for each worker
    for _ in range(len(processes)):
        task_queue.put(None)

    # wait for workers to terminate
    for worker in processes:
        worker.join()


def process_rollout_results(
    dataset,
    rollout_returns,
    subgoal_successes,
    rollout_gifs,
    video_captions,
    extra_info,
    result,
    eval,
):
    if not eval:
        dataset.add_traj_to_buffer(
            result["obs"],
            result["acs"],
            result["obj_acs"],
            result["rews"],
            result["dones"],
            result["lang_ann"],
        )
        # when we relabel with LLM, we will also include the summarized language embedding as summarized_lang_embedding
        if "summarized_lang_ann" in result:
            dataset.add_traj_to_buffer(
                result["obs"],
                result["acs"],
                result["obj_acs"],
                result["rews"],
                result["dones"],
                result["summarized_lang_ann"],
            )
        # also add the skills that composed the chain to the buffer
        switch_point = 0
        if "skill_switch_points" in result and torch.any(result["skill_switch_points"]):
            switch_point = result["skill_switch_points"].nonzero()[0].item()
            first_skill_done = result["dones"][:switch_point].clone()
            first_skill_done[-1] = 1.0

            # add the first skill to the buffer
            dataset.add_traj_to_buffer(
                result["obs"][: switch_point + 1],
                result["acs"][:switch_point],
                result["obj_acs"][:switch_point],
                result["rews"][:switch_point],
                first_skill_done,
                result["per_step_lang_anns"][0],
            )
            assert result["rews"][switch_point - 1].item() == 1

            # add the second skill to the buffer
            dataset.add_traj_to_buffer(
                result["obs"][switch_point:],
                result["acs"][switch_point:],
                result["obj_acs"][switch_point:],
                result["rews"][switch_point:],
                result["dones"][switch_point:],
                result["per_step_lang_anns"][switch_point],
            )

        # add all primitive skills that compose the chain to the buffer unless they are at switch point (already added)
        if (
            "current_skill_object" in result
            and result["current_skill_object"].num_skills > 1
        ):
            primitive_ends = result["rews"].clone()
            primitive_ends[-1] = 1
            primitive_skill_end_points = primitive_ends.nonzero()
            last_start_point = 0
            for i, skill_end_point in enumerate(primitive_skill_end_points):
                # two conditions to consider if doing bootstrapping:
                # 1. The primitive skill is part of the first skill in the chain and does not represent the full skill
                # 2. The primitive skill is part of the second skill in the chain and does not represent the full skill
                if "skill_switch_points" in result:
                    is_primitive_in_skill_1 = (
                        (skill_end_point.item() + 1 < switch_point) and i == 0
                    ) or ((skill_end_point.item() + 1 == switch_point) and i != 0)

                    is_primitive_in_skill_2 = (
                        (skill_end_point.item() + 1 > switch_point)
                        and (skill_end_point.item() + 1 < len(result["rews"]))
                        or (
                            skill_end_point.item() + 1 == len(result["rews"])
                            and last_start_point > switch_point
                        )
                    )

                # if any of the above are true, add back to buffer; also secondary or condition for when not bootstrapping and doing oracle fine-tuning instead
                # as there are no "skill_switch_points" when doing oracle fine-tuning
                if (
                    "skill_switch_points" not in result
                    or is_primitive_in_skill_1
                    or is_primitive_in_skill_2
                ):
                    start_point = last_start_point
                    end_point = skill_end_point.item() + 1
                    primitive_skill_done = result["rews"][start_point:end_point].clone()
                    curr_lang_embedding = (
                        result["current_skill_object"]
                        .get_skill_at_index(i)
                        .language_embedding
                    )
                    dataset.add_traj_to_buffer(
                        result["obs"][start_point : end_point + 1],
                        result["acs"][start_point:end_point],
                        result["obj_acs"][start_point:end_point],
                        result["rews"][start_point:end_point],
                        primitive_skill_done,
                        curr_lang_embedding,
                    )
                last_start_point = skill_end_point.item() + 1

    rollout_returns.append(result["rews"].sum().item())
    subgoal_successes.append(result["dones"][-1])
    # extra things to log related to bootstrapping
    if result["video_frames"] is not None:
        rollout_gifs.append(result["video_frames"])
    video_captions.append(result["video_caption"])
    if "first_skill_length" in result:
        extra_info["first_skill_length"].append(result["first_skill_length"])
        end_point = (
            len(result["rews"])
            if not torch.any(result["skill_switch_points"])
            else result["skill_switch_points"].nonzero()[0]
        )
        extra_info["first_skill_return"].append(result["rews"][:end_point].sum().item())
        extra_info["first_skill_success"].append(
            float(torch.any(result["skill_switch_points"]) or result["dones"][-1])
        )
    if "second_skill_length" in result:
        if result["second_skill_length"] != 0:
            extra_info["second_skill_length"].append(result["second_skill_length"])
            extra_info["second_skill_return"].append(
                result["rews"][result["skill_switch_points"].nonzero()[0] :]
                .sum()
                .item()
            )
            extra_info["second_skill_success"].append(result["dones"][-1])
        else:
            extra_info["second_skill_length"].append(None)
            extra_info["second_skill_return"].append(None)
            extra_info["second_skill_success"].append(None)
        # extra_info["second_skill_length"].append(result["second_skill_length"])
    if "num_primitive_skills_attempted" in result:
        extra_info["num_primitive_skills_attempted"].append(
            result["num_primitive_skills_attempted"]
        )
    if "new_skill_values" in result and len(result["new_skill_values"]) > 0:
        extra_info["new_skill_values"].extend(result["new_skill_values"])
    if "new_skill_llm_probs" in result and len(result["new_skill_llm_probs"]) > 0:
        for i, llm_prob in enumerate(result["new_skill_llm_probs"]):
            if torch.all(torch.tensor(llm_prob) == 1):
                extra_info["new_skill_llm_probs"].append(None)
            else:
                extra_info["new_skill_llm_probs"].append(llm_prob)

    if (
        "new_skill_sampled_types" in result
        and len(result["new_skill_sampled_types"]) > 0
    ):
        extra_info["new_skill_sampled_types"].extend(result["new_skill_sampled_types"])
    if "primitive_skill_types" in result and len(result["primitive_skill_types"]) > 0:
        extra_info["primitive_skill_types"].extend(result["primitive_skill_types"])
    extra_info["num_steps"].append(len(result["rews"]))
    if "sampled_skill_llm_probs" in result:
        extra_info["sampled_skill_llm_probs"].extend(result["sampled_skill_llm_probs"])
    if "valid_masks" in result:
        extra_info["valid_masks"].extend(result["valid_masks"])
    if "is_composite" in result and len(result["is_composite"]) > 0:
        extra_info["is_composite"].extend(result["is_composite"])


def multithread_dataset_aggregation(
    result_queue,
    rollout_returns,
    subgoal_successes,
    rollout_gifs,
    video_captions,
    extra_info,
    dataset,
    num_env_samples_list: list,
    composite_skill_list: list,
    composite_skill_set: set,
    eval: bool,
    unlabeled_data_list: list,
    num_rollouts: int,
):
    # asynchronously collect results from result_queue
    num_env_samples = 0
    num_finished_tasks = 0
    new_composite_skills = []
    with tqdm(total=num_rollouts) as pbar:
        while num_finished_tasks < num_rollouts:
            result = result_queue.get()
            if not eval:
                if (
                    "composite_skill_object" in result
                    and result["composite_skill_object"] is not None
                ):
                    new_skill_dict = dict(
                        skill=result["composite_skill_object"],
                        scene_index=result["scene_index"],
                        init_action=result["init_action"],
                    )
                    skill_in_list = (
                        tuple([new_skill_dict["skill"], new_skill_dict["scene_index"]])
                        in composite_skill_set
                    )
                    if not skill_in_list:
                        new_composite_skills.append(new_skill_dict)
                        composite_skill_set.add(
                            tuple(
                                [new_skill_dict["skill"], new_skill_dict["scene_index"]]
                            )
                        )
                    # still need to relabel this skill with the language annotation and embedding
                    unlabeled_data_list.append(result)
                else:
                    process_rollout_results(
                        dataset,
                        rollout_returns,
                        subgoal_successes,
                        rollout_gifs,
                        video_captions,
                        extra_info,
                        result,
                        eval,
                    )
            else:
                process_rollout_results(
                    dataset,
                    rollout_returns,
                    subgoal_successes,
                    rollout_gifs,
                    video_captions,
                    extra_info,
                    result,
                    eval,
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
    composite_skill_list.extend(new_composite_skills)


def multiprocess_rollout(
    task_queue,
    result_queue,
    config,
    dataset,
    composite_skill_list,
    composite_skill_set,
    language_embedding_model,
    large_language_model,
    eval,
    num_rollouts,
):
    rollout_returns = []
    subgoal_successes = []
    rollout_gifs = []
    video_captions = []
    extra_info = defaultdict(list)

    # create tasks for thread/process Queue
    args_func = lambda skill_index: dict(
        deterministic=(
            True if eval else config.deterministic_action
        ),  # deterministic action
        log_video=eval,  # log to video if eval is true
        composite_skill_list=composite_skill_list,
        epsilon=config.epsilon,
        eval=eval,
        selected_specific_task=skill_index if eval else None,
    )

    for i in range(num_rollouts):
        if eval:
            eval_skill_index = i
        else:
            eval_skill_index = None
        task_queue.put(args_func(eval_skill_index))

    num_env_samples_list = []  # use list for thread safety
    unlabeled_data_list = []  # to be relabeled by LLM or naively
    multithread_dataset_aggregation(
        result_queue,
        rollout_returns,
        subgoal_successes,
        rollout_gifs,
        video_captions,
        extra_info,
        dataset,
        num_env_samples_list,
        composite_skill_list,
        composite_skill_set,
        eval,
        unlabeled_data_list,
        num_rollouts,
    )

    # process unlabeled data in batches to reduce computation time with large language models
    # so here we relabel the data with the language model and generate embeddings in batches
    # then re-add these to the dataset
    if len(unlabeled_data_list) > 0:
        (
            concat_language_instructions,
            summarized_instructions,
            concat_instr_embeddings,
            summarized_instr_embeddings,
            skill_probs,
        ) = relabel_skills(
            config,
            unlabeled_data_list,
            language_embedding_model,
            large_language_model,
        )
        if len(summarized_instructions) > 0:
            print(summarized_instructions)
        print(concat_language_instructions)

        for i, (instruction, embedding, skill_prob) in enumerate(
            zip(
                (
                    concat_language_instructions
                    if len(summarized_instructions) == 0
                    else summarized_instructions
                ),
                concat_instr_embeddings,
                skill_probs,
            )
        ):
            rollout_result = unlabeled_data_list[i]
            rollout_result["video_caption"] = (
                f"{instruction} Completed {int(rollout_result['rews'].sum().item())}/{rollout_result['composite_skill_object'].num_skills} subgoals. Scene: {rollout_result['scene_index']}."
            )
            embedding = embedding
            rollout_result["lang_ann"] = embedding
            if len(summarized_instructions) > 0:
                summarized_embedding = summarized_instr_embeddings[i]
                rollout_result["summarized_lang_ann"] = summarized_embedding
                # set the label for the composite skill object
                rollout_result["composite_skill_object"].set_label(
                    instruction, summarized_embedding, skill_prob
                )
            else:
                # set the label for the composite skill object
                rollout_result["composite_skill_object"].set_label(
                    instruction, embedding, skill_prob
                )
            process_rollout_results(
                dataset,
                rollout_returns,
                subgoal_successes,
                rollout_gifs,
                video_captions,
                extra_info,
                rollout_result,
                eval,
            )

    num_env_samples = num_env_samples_list[0]
    rollout_metrics = log_rollout_metrics(
        rollout_returns,
        subgoal_successes,
        extra_info,
        rollout_gifs,
        video_captions,
        composite_skill_list,
        config,
    )
    return rollout_metrics, num_env_samples


def relabel_skills(
    config,
    unlabeled_result_list: list[dict],
    language_embedding_model: ETLanguageEncoder,
    llm: LargeLanguageModel,
):
    concat_language_instructions = []
    to_be_summarized_instructions = []
    skill_probs = []
    for result in unlabeled_result_list:
        primitive_instructions = result[
            "composite_skill_object"
        ].get_precomposed_language_instructions()
        skill_prob = 0
        if config.use_llm_for_next_skill:
            skill_prob = result["sampled_skill_llm_probs"][0]
        if config.llm_make_summary:
            to_be_summarized_instructions.append(primitive_instructions)
        concat_language_instructions.append(" ".join(primitive_instructions))
        skill_probs.append(skill_prob)

    summarized_instructions = []
    if config.use_llm and config.llm_make_summary:
        summarized_instructions = llm.generate_skill_labels(
            to_be_summarized_instructions
        )
    joined_instr_processed = language_embedding_model.encode(
        concat_language_instructions + summarized_instructions, convert_to_tensor=True
    ).cpu()
    concat_instr_processed = joined_instr_processed[: len(concat_language_instructions)]
    summarized_instr_processed = joined_instr_processed[
        len(concat_language_instructions) :
    ]
    return (
        concat_language_instructions,
        summarized_instructions,
        concat_instr_processed,
        summarized_instr_processed,
        skill_probs,
    )


def train_epoch(
    offline_rl_model: BOSSETIQLModel,
    dataloader,
    epoch,
    device,
):
    offline_rl_model.train()
    running_tracker_dict = defaultdict(list)
    with tqdm(dataloader, unit="batch") as tepoch:
        for data_dict in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            frames = send_to_device_if_not_none(data_dict, "skill_feature", device)
            action = send_to_device_if_not_none(data_dict, "low_action", device).long()
            obj_id = send_to_device_if_not_none(data_dict, "object_ids", device).long()
            interact_mask = send_to_device_if_not_none(
                data_dict, "valid_interact", device
            ).reshape(-1)
            lengths_frames = send_to_device_if_not_none(
                data_dict, "feature_length", device
            )
            lengths_lang = send_to_device_if_not_none(data_dict, "token_length", device)
            lang = send_to_device_if_not_none(data_dict, "ann_token", device).int()
            rewards = send_to_device_if_not_none(data_dict, "reward", device)
            terminals = send_to_device_if_not_none(data_dict, "terminal", device)

            rl_train_info = offline_rl_model.train_offline_from_batch(
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
            tepoch.set_postfix(
                loss=rl_train_info["policy_total_loss"],
                vf_loss=rl_train_info["vf_loss"],
            )
            for k, v in rl_train_info.items():
                running_tracker_dict[k].append(v)
    eval_metrics = {}
    for k, v in running_tracker_dict.items():
        eval_metrics[f"train_{k}"] = np.mean(v)
    return eval_metrics


def main(config):
    load_model_path = config.load_model_path

    checkpoint = torch.load(load_model_path, map_location="cpu")

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

    run = wandb.init(
        # resume=config.experiment_name,  # "allow",
        # name=config.experiment_name,
        resume="allow",
        id=config.experiment_name,
        name=config.experiment_name,
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY_NAME,
        notes=config.notes,
        config=config,
        group=config.run_group,
    )
    seed = config.seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = False  # uses a lot of gpu memory if True

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

    device = torch.device(config.gpus[0])
    dataset = RLBuffer(
        path=config.data_dir,
        split="train",
        use_full_skill=False,
        max_skill_length=config.max_skill_length,
        drop_old_data=config.drop_old_data,
    )
    offline_rl_model = BOSSETIQLModel(config)
    num_env_samples = 0
    start_epoch = 0
    composite_skill_list = []  # learned/attempted composite skills
    composite_skill_set = set()
    composite_skill_primitive_counter = (
        Counter()
    )  # learned/attempted primitive skills in composite

    # if num env samples in checkpoint then this is resuming an RL rollout checkpoint
    if config.old_new_sampling:
        new_dataset = RLBuffer(
            path=config.data_dir,
            split="train",
            use_full_skill=False,
            max_skill_length=config.max_skill_length,
            drop_old_data=True,  # always drop old data for the new_dataset buffer since we already have the data in the old dataset buffer
        )
        # weigh both equally
        dataset = CombinedDataset(
            dataset,
            new_dataset,
            first_dataset_ratio=config.old_data_ratio,
        )
    # if loading an already (partly) trained RL model,
    # then start from the same num samples and restore the replay buffer
    if "num_env_samples" in checkpoint:
        num_env_samples = checkpoint["num_env_samples"]
        start_epoch = checkpoint["epoch"]
        dataset.rl_buffer = checkpoint["saved_buffer"]
        print(
            f"Restoring from epoch {start_epoch}, num_env_samples {num_env_samples}, buffer size: {len(dataset.rl_buffer)}"
        )
        composite_skill_list = checkpoint["composite_skill_list"]
        composite_skill_primitive_counter = checkpoint[
            "composite_skill_primitive_counter"
        ]
        # load random states
        torch.set_rng_state(checkpoint["torch_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        random.setstate(checkpoint["python_random_state"])

    dataset_random_sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=True,
        num_samples=int(
            config.policy_update_ratio
            * config.batch_size
            * config.num_rollouts_per_epoch
        ),
    )
    if not config.eval_only:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=dataset_random_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_func,
        )
    if len(config.gpus) > 1:
        print(f"-----Using {len(config.gpus)} GPUs-----")
        offline_rl_model = nn.DataParallel(
            offline_rl_model,
            device_ids=config.gpus,
        )
    offline_rl_model.to(device)

    offline_rl_model.load_from_checkpoint(checkpoint)

    if config.new_lr is not None:
        offline_rl_model.set_lr(config.new_lr)

    resnet_args = AttrDict()
    resnet_args.visual_model = "resnet18"
    resnet_args.gpu = config.gpus[0]
    resnet = Resnet(resnet_args, eval=True, use_conv_feat=True)
    sentence_encoder = ETLanguageEncoder(offline_rl_model.vocab_word)
    semantic_search_model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
    ).to(device)

    if config.use_llm:
        llm_model = LargeLanguageModel(config)
    else:
        llm_model = None

    # multiprocessed rollout setup
    task_queue = queue.SimpleQueue()
    result_queue = queue.SimpleQueue()

    processes, num_tasks = setup_mp(
        result_queue,
        task_queue,
        config,
        device,
        offline_rl_model,
        sentence_encoder,
        semantic_search_model,
        llm_model,
        resnet,
    )

    def signal_handler(sig, frame):
        print("SIGINT received. Exiting...closing all processes first")
        cleanup_mp(task_queue, processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    old_num_composite_skills = 0
    epoch = start_epoch
    while num_env_samples < config.total_env_steps:
        print(
            f"Num env samples: {num_env_samples} / {config.total_env_steps}: {num_env_samples / config.total_env_steps * 100:.2f}% done"
        )
        if not config.eval_only:
            epoch += 1
            log_dict = {}
            if num_env_samples == 0:
                # first do an evaluation
                eval_metrics, _ = multiprocess_rollout(
                    task_queue,
                    result_queue,
                    config,
                    dataset,
                    composite_skill_list,
                    composite_skill_set,
                    sentence_encoder,
                    llm_model,
                    eval=True,
                    num_rollouts=num_tasks,
                )
                first_eval_dict = {}
                for k, v in eval_metrics.items():
                    first_eval_dict[f"fixed_eval_{k}"] = v
                wandb.log(first_eval_dict)
                # collect warmup samples
                result_metrics, new_env_samples = multiprocess_rollout(
                    task_queue,
                    result_queue,
                    config,
                    dataset,
                    composite_skill_list,
                    composite_skill_set,
                    sentence_encoder,
                    llm_model,
                    eval=False,
                    num_rollouts=50,
                )
            else:
                result_metrics, new_env_samples = multiprocess_rollout(
                    task_queue,
                    result_queue,
                    config,
                    dataset,
                    composite_skill_list,
                    composite_skill_set,
                    sentence_encoder,
                    llm_model,
                    eval=False,
                    num_rollouts=config.num_rollouts_per_epoch,
                )
            # update the number of samples to sample
            dataset_random_sampler._num_samples = int(
                config.policy_update_ratio
                * config.batch_size
                * config.num_rollouts_per_epoch
            )
            for k, v in result_metrics.items():
                log_dict[f"rollout_{k}"] = v
            num_env_samples += new_env_samples

            # train policy on a few updates
            training_metrics = train_epoch(
                offline_rl_model,
                dataloader,
                epoch,
                device,
            )
            log_dict.update(training_metrics)
            log_dict["epoch"] = epoch

            # Eval every EVAL_INTERVAL timesteps or once reached total env steps
            if (
                num_env_samples % EVAL_INTERVAL
                < (num_env_samples - new_env_samples) % EVAL_INTERVAL
            ) or num_env_samples >= config.total_env_steps:
                eval_metrics, _ = multiprocess_rollout(
                    task_queue,
                    result_queue,
                    config,
                    dataset,
                    composite_skill_list,
                    composite_skill_set,
                    sentence_encoder,
                    llm_model,
                    eval=True,
                    num_rollouts=num_tasks,
                )
                for k, v in eval_metrics.items():
                    log_dict[f"fixed_eval_{k}"] = v

            log_dict.update({"num_composite_skills": len(composite_skill_list)})

            # distribution of primitives in composite skills
            for skill_dict in composite_skill_list[old_num_composite_skills:]:
                primitive_skills = []
                for primitive in skill_dict["skill"].primitive_skills:
                    primitive_skills.append(primitive["planner_action"]["action"])
                composite_skill_primitive_counter.update(primitive_skills)
            # plot bar chart of sampled new skill types frequency distribution
            freq_data = np.array(
                [
                    composite_skill_primitive_counter[skill_type]
                    for skill_type in primitive_skill_types
                ]
            )
            freq_data = freq_data / freq_data.sum()

            log_dict.update(
                {
                    "composite_skill_primitive_distribution": make_bar_chart(
                        freq_data,
                        primitive_skill_types,
                        "Composite Skill Primitive Distribution",
                        "Primitive Skill Type",
                        "Frequency",
                        ylim=(0, 1),
                    )
                }
            )
            old_num_composite_skills = len(composite_skill_list)

            wandb.log(
                log_dict,
                step=num_env_samples,
            )

            model_state_dict = offline_rl_model.get_all_state_dicts()
            model_state_dict.update(
                dict(
                    epoch=epoch,
                    num_env_samples=num_env_samples,
                    saved_buffer=dataset.rl_buffer,
                    composite_skill_list=composite_skill_list,
                    torch_random_state=torch.get_rng_state(),
                    numpy_random_state=np.random.get_state(),
                    python_random_state=random.getstate(),
                    composite_skill_primitive_counter=composite_skill_primitive_counter,
                )
            )
            if epoch % 5 == 0 or num_env_samples >= config.total_env_steps:
                torch.save(
                    model_state_dict,
                    os.path.join(
                        config.save_dir,
                        (
                            config.experiment_name
                            if config.experiment_name
                            else wandb.run.name
                        ),
                        f"rl_finetune_model.pth",
                    ),
                )
        if config.eval_only:
            break

    cleanup_mp(task_queue, processes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill bootstrapping script")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Parent directory containing the dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_rl_models/",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "--new_lr",
        type=float,
        default=None,
        help="new learning rate to use. If None, use the one in the config file",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="which gpus. pass in as comma separated string to use DataParallel on multiple GPUs",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for initialization"
    )
    parser.add_argument(
        "--which_floorplan",
        type=int,
        default=None,
        required=True,
        help="index of the specific floorplan to use",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers for data loading"
    )
    parser.add_argument(
        "--num_rollout_workers",
        type=int,
        default=4,
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
        "--eval_only",
        action="store_true",
        help="whether to only evaluate the model w/o training",
    )
    parser.add_argument(
        "--total_env_steps",
        type=int,
        default=500000,
        help="total number of environment steps to train for",
    )
    parser.add_argument(
        "--policy_update_ratio",
        type=float,
        default=15,
        help="ratio of updates between policy and env trajs collected each epoch. Each policy update step here updates based on all (s, a, s') samples in one whole trajectory",
    )
    parser.add_argument(
        "--num_rollouts_per_epoch",
        type=int,
        default=10,
        help="number of env rollouts per epoch when training online",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        required=True,
        help="path to load the model from to finetune from",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="starting epsilon for epsilon-greedy action sampling",
    )
    parser.add_argument(
        "--deterministic_action",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="whether to use deterministic actions during training",
    )
    parser.add_argument(
        "--advantage_temp",
        type=float,
        default=None,
        help="temperature for computing the advantage",
    )
    parser.add_argument(
        "--gamma", type=float, default=None, help="discount factor for the RL loss"
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="quantile for computing the reward",
    )
    parser.add_argument(
        "--clip_score",
        type=float,
        default=100,
        help="max to clip the advantage to",
    )
    parser.add_argument(
        "--run_group",
        type=str,
        default=None,
        help="group to run the experiment in. If None, no group is used",
    )
    parser.add_argument(
        "--rand_init",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="whether to randomly initialize the agent in a reachable navigation point when training",
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
    parser.add_argument(
        "--drop_old_data",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="whether to drop the old offline RL data.",
    )
    parser.add_argument(
        "--value_sampling_temp",
        type=float,
        default=0.6,
        help="temperature for sampling skills from the value function",
    )
    parser.add_argument(
        "--use_value_func",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to use the value function for sampling first or next skills",
    )
    parser.add_argument(
        "--use_value_for_next_skill",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="whether to use the value function for sampling next skills",
    )
    parser.add_argument(
        "--old_new_sampling",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to sample old and new trajectories simultaneously, via --old_data_ratio, for training",
    )
    parser.add_argument(
        "--old_data_ratio",
        type=float,
        default=0.5,
        help="ratio of old data to sample for training",
    )
    # LLM arguments
    parser.add_argument(
        "--use_llm",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to use the large language model",
    )
    parser.add_argument(
        "--use_llm_for_next_skill",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to use the large language model for the next skill",
    )
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
            "huggyllama/llama-7b",
            "google/gemma-2-9b",
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
        default=3,
        help="batch size for the large language model",
    )
    parser.add_argument(
        "--llm_sampling_temp",
        type=float,
        default=1.0,
        help="temperature for sampling skills from the LLM skill distribution",
    )
    parser.add_argument(
        "--llm_make_summary",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to make a summary of the skills with the large language model",
    )
    parser.add_argument(
        "--llm_max_new_tokens",
        type=int,
        default=30,
        help="max number of new tokens to sample from the large language model",
    )
    parser.add_argument(
        "--llm_logprob_weight",
        type=float,
        default=1.0,
        help="weight for the logprob of the large language model during online skill sampling. Valid between 0 and 1.",
    )
    parser.add_argument(
        "--skill_match_with_dataset",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to match the skills sampled from the LLM with the dataset for NEXT skill sampling",
    )
    parser.add_argument(
        "--num_skill_match_generations",
        type=int,
        default=10,
        help="how many generations to generate. 10 works for opt-6.7 on a 3090",
    )
    parser.add_argument(
        "--llm_next_skill_temp",
        type=float,
        default=0.8,
        help="temperature for sampling skills from the LLM skill distribution for the next skill",
    )
    parser.add_argument(
        "--llm_next_skill_top_p",
        type=float,
        default=0.8,
        help="top p for sampling skills from the LLM text distribution for the next skill",
    )
    parser.add_argument(
        "--llm_summary_temp",
        type=float,
        default=0.5,
        help="temperature for generating summaries from the LLM",
    )
    # runs no bootstrap (e.g., oracle fine-tuning)
    parser.add_argument(
        "--no_bootstrap",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="whether to run the model without bootstrapping",
    )
    # rebuttal arg
    parser.add_argument(
        "--forced_max_skills_to_chain",
        type=int,
        default=None,
        help="force the maximum number of skills to chain. used to fine-tune the model on just primitive skills to then give to SayCan (set to 1)",
    )
    config = parser.parse_args()
    config.use_pretrained_lang = False
    config.gpus = [int(gpu) for gpu in config.gpus.strip().split(",")]
    config.llm_gpus = [int(gpu) for gpu in config.llm_gpus.strip().split(",")]
    if config.no_bootstrap:
        config.use_llm = False
        config.llm_model = None
    if config.experiment_name is None and config.run_group is not None:
        if config.seed is None:
            config.experiment_name = f"{config.run_group}"
        else:
            config.experiment_name = f"{config.run_group}_{config.seed}"
    mp.set_sharing_strategy(
        "file_system"
    )  # to ensure the too many open files error doesn't happen with the dataloader
    main(config)
