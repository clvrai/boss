import torch
import os

from boss.rollouts.alfred_gym_env import ALFREDRLBootstrappingEnv
from boss.models.boss_model import BOSSETIQLModel
from boss.utils.utils import get_action_from_agent, load_object_class, generate_video
from boss.rollouts.rollout import run_policy

DATA_PATH = f"{os.environ['BOSS']}/boss/alfred/data/json_2.1.0_merge_goto/preprocess"
REWARD_CONFIG_PATH = f"{os.environ['BOSS']}/boss/alfred/models/config/rewards.json"


def run_bootstrapping_rollout(
    env: ALFREDRLBootstrappingEnv,
    model,
    deterministic,
    log_video,
    composite_skill_list,
    epsilon,
    eval,
    selected_specific_task=None,
):
    model.eval()
    # format of skills in composite_skill_list is a dictionary with the following keys:
    # "skill": Skill Object
    # "scene_index": int of specific string index
    # "init_action": initial eval action
    model_input = {}
    obss = []
    per_step_lang_annotations = []
    acs = []
    obj_acs = []
    env_rewards = []
    str_act = []
    video_frames = []
    value_predict = []
    new_skill_values = []
    new_skill_llm_probs = []
    primitive_skill_types = []
    per_step_primitive_skill_types = []
    skill_switch_points = []
    sampled_skill_types = []
    valid_masks = []
    sampled_skill_llm_probs = []

    obs, info = env.bootstrap_reset(
        composite_skill_list=composite_skill_list,
    )
    model_input["frames_buffer"], model_input["language_ann"] = (
        obs,
        info.lang_embedding,
    )
    model_input["action_traj"] = torch.zeros(1, 0).long().to(env.device)
    model_input["object_traj"] = torch.zeros(1, 0).long().to(env.device)
    curr_frame = info["frame"]
    sampled_scene_index = info["scene_index"]
    # distribution of values for the current skill, means
    new_skill_values.append(info.values.tolist())
    sampled_skill_types.append(info.sampled_skill_types)
    primitive_skill_types.append(info.primitive_skill_type)
    valid_masks.append(info.valid_mask)
    if "sampled_skill_llm_prob" in info:
        sampled_skill_llm_probs.append(info.sampled_skill_llm_prob)
    init_action = info["init_action"]
    done, timeout = False, False
    skill_switched = 0
    current_skill_embedding = (
        info.current_skill.composite_language_embedding.detach().cpu()
    )
    first_skill_length = info.current_skill.num_skills
    while not (done or timeout):
        obss.append(obs[:, -1].cpu().detach())
        per_step_lang_annotations.append(current_skill_embedding)
        skill_switch_points.append(skill_switched)
        video_frames.append(curr_frame)
        (action, output_object, value_output) = get_action_from_agent(
            model,
            model_input,
            env.vocab,
            env.vocab_obj,
            env,
            deterministic=deterministic,
            epsilon=epsilon,
            ret_value=True,
        )
        value_output = value_output.squeeze().cpu().detach().numpy()
        value_predict.append(value_output)

        # environment step
        step_input = dict(action=action, object=output_object)
        next_obs, reward, done, info = env.bootstrap_step(step_input)
        timeout = info.timeout
        # langauge instruction
        # might change if the current skill is
        # a composite skill and we just finished executing
        # a primitive skill
        next_lang_embedding = info.lang_embedding
        # rgb frame
        next_frame = info.frame

        # update the feature buffer
        model_input["frames_buffer"] = next_obs
        model_input["language_ann"] = next_lang_embedding

        # update actions
        tensor_action = torch.tensor(env.vocab["action_low"].word2index(action) - 2).to(
            env.device
        )
        acs.append(tensor_action.cpu())
        obj_index = load_object_class(env.vocab_obj, output_object)
        obj_acs.append(obj_index)

        model_input["action_traj"] = torch.cat(
            [model_input["action_traj"], tensor_action.unsqueeze(0).unsqueeze(0)],
            dim=1,
        )
        model_input["object_traj"] = torch.cat(
            [
                model_input["object_traj"],
                torch.tensor(obj_index).unsqueeze(0).unsqueeze(0).to(env.device),
            ],
            dim=1,
        )
        model_input["action_traj"] = model_input["action_traj"][
            :, -env.obs_concat_length + 1 :
        ]
        model_input["object_traj"] = model_input["object_traj"][
            :, -env.obs_concat_length + 1 :
        ]
        # we have reset the frame buffer, we need to reset the corresponding action buffers
        if next_obs.shape[1] == 1:
            model_input["action_traj"] = torch.zeros(1, 0).long().to(env.device)
            model_input["object_traj"] = torch.zeros(1, 0).long().to(env.device)
        # next frames
        curr_frame = next_frame
        obs = next_obs

        # save things
        if "primitive_skill_type" in info:
            primitive_skill_types.append(info.primitive_skill_type)
        str_act.append(
            dict(
                action=action,
                object=(
                    output_object.split("|")[0] if output_object is not None else None
                ),
            )
        )
        env_rewards.append(reward)
        if "values" in info:
            # only when a new skill is sampled and used for skill sampling
            new_skill_values.append(info.values.tolist())
            sampled_skill_types.append(info.sampled_skill_types)
        if "valid_mask" in info:
            valid_masks.append(info.valid_mask)
        if "sampled_skill_llm_prob" in info:
            sampled_skill_llm_probs.append(info.sampled_skill_llm_prob)
        per_step_primitive_skill_types.append(info.per_step_primitive_skill_type)
        if info.second_sampled_skill is not None:
            # if we have started to chain a new composite skill, we should add the second primtiive/composite skill's language_embedding
            current_skill_embedding = (
                info.second_sampled_skill.composite_language_embedding.detach().cpu()
            )
            skill_switched = 1
        else:
            skill_switched = 0

    # generate last frame and last action to get the value output for the last frame
    video_frames.append(next_frame)
    (_, _, value_output) = get_action_from_agent(
        model,
        model_input,
        env.vocab,
        env.vocab_obj,
        env,
        deterministic=deterministic,
        epsilon=epsilon,
        ret_value=True,
    )
    value_output = value_output.squeeze().cpu().detach().numpy()
    value_predict.append(value_output)

    if log_video:
        video_frames = generate_video(
            value_predict,
            str_act,
            video_frames,
            env_rewards,
            primitive_skill_types=per_step_primitive_skill_types,
            log_debugging_info=False,
        )
    obss.append(obs[:, -1].cpu().detach())  # next obs
    rewards = torch.tensor(env_rewards, dtype=torch.float)
    dones = torch.zeros(len(rewards))
    dones[-1] = done
    # if a new composite skill has been created through chaining, get the info for that
    if info.composite_skill is not None:
        language_annotation = None  # we don't have labels for this yet
        language_instruction = " ".join(
            info.composite_skill.primitive_instructions_to_compose
        )  # we don't have labels for this yet. Mark as UNLABELED for visualization.
        language_instruction = f"UNLABELED: {language_instruction}"
        # mark how long the skill the agent attempted was regardless of success
        skill_attempt_length = info.composite_skill.num_skills
    else:
        language_annotation = (
            info.current_skill.composite_language_embedding.cpu().detach()
        )
        language_instruction = info.current_skill.composite_language_instruction
        skill_attempt_length = info.current_skill.num_skills
    vid_caption = f"{language_instruction}: {'SUCCESS' if done else 'FAIL'}. Completed {rewards.sum().int().item()}/{skill_attempt_length} subgoals. Scene: {sampled_scene_index}"
    return dict(
        obs=torch.cat(obss),
        acs=torch.tensor(acs),
        obj_acs=torch.tensor(obj_acs),
        rews=rewards,
        dones=dones,
        lang_ann=language_annotation,
        per_step_lang_anns=per_step_lang_annotations,
        skill_switch_points=torch.tensor(skill_switch_points),
        video_frames=video_frames if log_video else None,
        video_caption=vid_caption,
        composite_skill_object=info["composite_skill"],
        current_skill_object=info["current_skill"],
        init_action=init_action,
        scene_index=sampled_scene_index,
        primitive_skill_types=primitive_skill_types,
        first_skill_length=first_skill_length,
        second_skill_length=skill_attempt_length - first_skill_length,
        num_primitive_skills_attempted=info["current_skill"].num_skills,
        new_skill_values=new_skill_values,
        new_skill_llm_probs=new_skill_llm_probs,
        new_skill_sampled_types=sampled_skill_types,
        sampled_skill_llm_probs=sampled_skill_llm_probs,
        valid_masks=valid_masks,
    )


def run_policy_multi_process(
    ret_queue,
    task_queue,
    config,
    device,
    offline_rl_model: BOSSETIQLModel,
    lang_embedding_model,
    semantic_search_model,
    llm,
    resnet,
):
    env = ALFREDRLBootstrappingEnv(
        eval_json=config.eval_json,
        which_floorplan=config.which_floorplan,
        use_llm=config.use_llm,
        device=device,
        rand_init=config.rand_init,
        visual_preprocessor=resnet,
        lang_embedding_model=lang_embedding_model,
        semantic_search_model=semantic_search_model,
        llm_model=llm,
        obs_concat_length=config.max_skill_length,
        value_func=offline_rl_model.critics,
        forced_max_skills_to_chain=config.forced_max_skills_to_chain,
    )
    num_eval_tasks = env.num_tasks
    # put the number of tasks into the return queue to tell the calling script thing how many rollouts to perform for evaluation
    ret_queue.put(num_eval_tasks)
    while True:
        with torch.no_grad():
            task_kwargs = task_queue.get()
            if task_kwargs is None:
                break
            eval = task_kwargs["eval"]
            if eval:
                ret_queue.put(
                    run_policy(
                        env,
                        offline_rl_model,
                        resnet,
                        device,
                        config.max_skill_length,
                        **task_kwargs,
                    )
                )
            else:
                ret_queue.put(
                    run_bootstrapping_rollout(env, offline_rl_model, **task_kwargs)
                )
    env.cleanup()
