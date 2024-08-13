import torch
from PIL import Image

from boss.rollouts.alfred_gym_env import ALFREDRLBootstrappingEnv
from boss.rollouts.skill import Skill
from boss.utils.data_utils import process_annotation_inference
from boss.utils.utils import (
    generate_video,
    get_action_from_agent,
    load_object_class,
)


def run_policy(
    env: ALFREDRLBootstrappingEnv,
    model,
    visual_preprocessor,
    device,
    max_skill_length,
    deterministic,
    log_video,
    composite_skill_list,  # unused
    epsilon,
    eval,
    selected_specific_task=None,
):
    # actually does the rollout. This function is a bit tricky to integrate with ALFRED's required code.
    model.eval()
    ob, info = env.reset(selected_specific_task)
    # build the features to give as input to the actual model
    feat = {}
    # initialize frames and action buffers for the transformer
    ob = visual_preprocessor.featurize([Image.fromarray(ob)], batch=1)
    feat["frames_buffer"] = ob.unsqueeze(0).to(device)
    feat["action_traj"] = torch.zeros(1, 0).long().to(device)
    feat["object_traj"] = torch.zeros(1, 0).long().to(device)

    full_task_instruction = info.lang_instruction
    primitive_subgoal_instr = info.primitive_lang_instruction

    if eval:
        curr_lang_instruction = full_task_instruction
    else:
        curr_lang_instruction = primitive_subgoal_instr

    feat["language_ann"] = (
        process_annotation_inference(curr_lang_instruction, model.vocab_word)
        .to(device)
        .unsqueeze(0)
    )

    obs = []
    acs = []
    obj_acs = []
    primitive_skill_lang_anns = []
    dones = []
    env_rewards = []
    str_act = []
    video_frames = []
    value_predict = []
    # skill_switch_points = []
    primitive_skill_embeddings = (
        []
    )  # used for adding all primitive skill data to buffer individually
    primitive_skill_embeddings.append(feat["language_ann"].cpu().detach().squeeze(0))
    primitive_skill_lang_anns.append(primitive_subgoal_instr)
    # skill_switched = False
    done, timeout = False, False
    while not (done or timeout):
        # skill_switch_points.append(int(skill_switched))
        obs.append(ob.cpu().detach().squeeze(1))
        video_frames.append(env.obs)

        (action, output_object, _) = get_action_from_agent(
            model,
            feat,
            env.vocab,
            env.vocab_obj,
            env,
            deterministic=deterministic,
            epsilon=epsilon,
            ret_value=False,
        )
        value_output = None
        if value_output != None:
            value_output = value_output.squeeze().cpu().detach().numpy()
            value_predict.append(value_output)

        action_dict = dict(action=action, object=output_object)
        next_ob, rew, done, info = env.step(action_dict)
        timeout = info.timeout
        next_ob = Image.fromarray(next_ob)
        next_ob = (
            visual_preprocessor.featurize([next_ob], batch=1).unsqueeze(0).to(device)
        )
        ob = next_ob
        feat["frames_buffer"] = torch.cat([feat["frames_buffer"], next_ob], dim=1).to(
            device
        )
        # - 2 because ET dataloader had a -1 for padding reasons on the action, and we did - 1 when processing ALFRED actions to get rid of
        # the extraneous END action
        tensor_action = torch.tensor(env.vocab["action_low"].word2index(action) - 2).to(
            device
        )
        feat["action_traj"] = torch.cat(
            [feat["action_traj"], tensor_action.unsqueeze(0).unsqueeze(0)],
            dim=1,
        ).to(device)
        obj_index = load_object_class(env.vocab_obj, output_object)
        feat["object_traj"] = torch.cat(
            [
                feat["object_traj"],
                torch.tensor(obj_index).unsqueeze(0).unsqueeze(0).to(device),
            ],
            dim=1,
        )
        feat["frames_buffer"] = feat["frames_buffer"][:, -max_skill_length:]
        feat["action_traj"] = feat["action_traj"][:, -max_skill_length + 1 :]
        feat["object_traj"] = feat["object_traj"][:, -max_skill_length + 1 :]

        # skill_switched = False
        if rew > 0 and not done:
            # skill_switched = True
            if not eval:  # task has switched, redo the model input features
                feat["frames_buffer"] = feat["frames_buffer"][:, -1:]
                feat["action_traj"] = torch.zeros(1, 0).long().to(device)
                feat["object_traj"] = torch.zeros(1, 0).long().to(device)
                curr_lang_instruction = info.primitive_lang_instruction
                embedded_annotation = process_annotation_inference(
                    curr_lang_instruction, model.vocab_word
                )
                feat["language_ann"] = embedded_annotation.to(device).unsqueeze(0)
                primitive_skill_embeddings.append(embedded_annotation)
                primitive_skill_lang_anns.append(curr_lang_instruction)

        env_rewards.append(rew)

        acs.append(tensor_action.cpu())
        str_act.append(
            dict(
                action=action,
                object=(
                    output_object.split("|")[0] if output_object is not None else None
                ),
            )
        )
        obj_acs.append(obj_index)

    subgoal_last_frame_video = env.obs
    video_frames.append(subgoal_last_frame_video)
    (*_,) = get_action_from_agent(
        model,
        feat,
        env.vocab,
        env.vocab_obj,
        env,
        deterministic=deterministic,
        epsilon=epsilon,
        ret_value=False,
    )
    obs.append(ob.cpu().detach().squeeze(1))  # last next obs
    value_output = None
    if value_output != None:
        value_output = value_output.squeeze().cpu().detach().numpy()
        value_predict.append(value_output)

    if log_video:
        video_frames = generate_video(
            value_predict, str_act, video_frames, env_rewards, log_debugging_info=False
        )

    rewards = torch.tensor(env_rewards, dtype=torch.float)
    dones = torch.zeros(len(rewards))
    dones[-1] = done
    vid_caption = f"{full_task_instruction}: {'SUCCESS' if done else 'FAIL'}. Return: {rewards.sum()}/{env.num_subgoals_to_complete}."
    # create a dummy skill object to return to `run_skill_bootstrapping` so all primitive skills are added to buffer
    ret_skill_object = Skill(
        [
            dict(annotations=annotation, discrete_action=dict(action=None, args=[]))
            for annotation in primitive_skill_lang_anns
        ],
        embeddings=primitive_skill_embeddings,
    )
    return dict(
        obs=torch.cat(obs),
        acs=torch.tensor(acs),
        obj_acs=torch.tensor(obj_acs),
        rews=rewards,
        dones=dones,
        # skill_switch_points=torch.tensor(skill_switch_points),
        video_frames=video_frames if log_video else None,
        video_caption=vid_caption,
        lang_ann=process_annotation_inference(
            full_task_instruction,
            model.vocab_word,
        ),
        current_skill_object=ret_skill_object,
        # skill_length=env.num_subgoals_to_complete,
        num_primitive_skills_attempted=env.num_subgoals_to_complete,
    )


def run_policy_multi_process(
    ret_queue,
    task_queue,
    config,
    device,
    offline_rl_model,
    lang_embedding_model,
    semantic_search_model,
    llm,
    resnet,
):
    env = ALFREDRLBootstrappingEnv(
        eval_json=config.eval_json,
        which_floorplan=config.which_floorplan,
        use_llm=config.use_llm,
        rand_init=config.rand_init,
    )
    num_eval_tasks = env.num_tasks
    # put the number of tasks into the return queue to tell the calling script thing how many rollouts to perform for evaluation
    ret_queue.put(num_eval_tasks)
    while True:
        task_kwargs = task_queue.get()
        if task_kwargs is None:
            break
        with torch.no_grad():
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
    env.cleanup()
