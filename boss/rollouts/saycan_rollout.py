import torch
from PIL import Image
import revtok

from boss.models.saycan import SaycanPlanner
from boss.models.boss_model import BOSSETIQLModel
from boss.rollouts.alfred_gym_env import ALFREDRLBootstrappingEnv
from boss.utils.data_utils import (
    remove_spaces_and_lower,
    pad_sequence,
    numericalize,
    process_annotation_inference,
)
from boss.utils.utils import (
    generate_video,
    load_object_class,
    get_action_from_agent,
    process_skill_strings,
)


def encode(annotations, vocab_word, convert_to_tensor=False):
    if convert_to_tensor:
        return pad_sequence(
            [process_annotation_inference(a, vocab_word) for a in annotations],
            batch_first=True,
            padding_value=0,
        )
    return [process_annotation_inference(a, vocab_word) for a in annotations]


def get_next_skill_from_saycan(
    model,
    saycan_planner: SaycanPlanner,
    high_level_skill: str,
    primitive_skill_annotations: list[str],
    already_completed_skills: list[str],
    feat: dict,
    device,
):
    llm_logprobs = saycan_planner.get_saycan_logprobs(
        already_completed_skills,
        primitive_skill_annotations,
        [high_level_skill],
    )
    # get value logprobs
    primitive_embeddings = encode(
        primitive_skill_annotations, model.vocab_word, convert_to_tensor=True
    )
    values = []
    for primitive_embedding in primitive_embeddings:
        primitive_embedding = primitive_embedding.to(device)
        feat["language_ann"] = primitive_embedding.unsqueeze(0)
        *_, value = model.step(feat, ret_value=True)
        values.append(value.unsqueeze(0))
    values = torch.cat(values, dim=0)
    values = torch.clamp(values, min=0, max=1).cpu()
    # combine LLM and values
    llm_probs = torch.exp(llm_logprobs)
    combined_affordance_probs = llm_probs * values
    # now take the argmax
    next_skill_idx = torch.argmax(combined_affordance_probs).item()
    feat["language_ann"] = encode(
        primitive_skill_annotations[next_skill_idx : next_skill_idx + 1],
        model.vocab_word,
        convert_to_tensor=True,
    ).to(
        device
    )  # re-encode the selected skill so there's no padding
    return primitive_skill_annotations[next_skill_idx]


def run_policy(
    env: ALFREDRLBootstrappingEnv,
    model: BOSSETIQLModel,
    saycan_planner: SaycanPlanner,
    visual_preprocessor,
    device,
    max_skill_length,
    deterministic,
    log_video,
    epsilon,
    selected_specific_subgoal=None,
):
    # actually does the rollout. This function is a bit tricky to integrate with ALFRED's required code.
    model.eval()
    ob, info = env.reset(selected_specific_subgoal)
    # build the features to give as input to the actual model
    feat = {}
    # initialize frames and action buffers for the transformer
    ob = visual_preprocessor.featurize([Image.fromarray(ob)], batch=1)
    feat["frames_buffer"] = ob.unsqueeze(0).to(device)
    feat["action_traj"] = torch.zeros(1, 0).long().to(device)
    feat["object_traj"] = torch.zeros(1, 0).long().to(device)

    chained_subgoal_instr = info.lang_instruction
    actually_selected_subgoal = info.task
    # get all primitive skills from this env
    subgoal_pool = env.subgoal_pool
    # subgoal info
    primitive_skills_to_choose_from = process_skill_strings(
        [subgoal_pool[actually_selected_subgoal]["primitive_skills"][0]["annotations"]]
    )

    ann_l = revtok.tokenize(remove_spaces_and_lower(chained_subgoal_instr))
    ann_l = [w.strip().lower() for w in ann_l]
    ann_token = numericalize(model.vocab_word, ann_l, train=False)
    ann_token = torch.tensor(ann_token).long()

    obs = []
    acs = []
    obj_acs = []
    dones = []
    env_rewards = []
    str_act = []
    video_frames = []
    value_predict = []
    predicted_skills = []
    completed_skills = []
    done, timeout = False, False
    # get first saycan planner action
    saycan_selected_next_skill = get_next_skill_from_saycan(
        model,
        saycan_planner,
        chained_subgoal_instr,
        primitive_skills_to_choose_from,
        completed_skills,
        feat,
        device,
    )
    predicted_skills.append(saycan_selected_next_skill)

    while not (done or timeout):
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
        # move onto the next skill predicted by saycan
        if rew == 1 and not done:
            completed_skills.append(saycan_selected_next_skill)
            feat["frames_buffer"] = next_ob.to(device)
            feat["action_traj"] = torch.zeros(1, 0).long().to(device)
            feat["object_traj"] = torch.zeros(1, 0).long().to(device)
            saycan_selected_next_skill = get_next_skill_from_saycan(
                model,
                saycan_planner,
                chained_subgoal_instr,
                primitive_skills_to_choose_from,
                completed_skills,
                feat,
                device,
            )
            predicted_skills.append(saycan_selected_next_skill)
        feat["frames_buffer"] = feat["frames_buffer"][:, -max_skill_length:]
        feat["action_traj"] = feat["action_traj"][:, -max_skill_length + 1 :]
        feat["object_traj"] = feat["object_traj"][:, -max_skill_length + 1 :]

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
    vid_caption = f"{chained_subgoal_instr[0] if isinstance(chained_subgoal_instr, list) else chained_subgoal_instr}: {'SUCCESS' if done else 'FAIL'}. Return: {rewards.sum()}/{env.num_subgoals_to_complete}."
    ground_truth_sequence = " ".join(primitive_skills_to_choose_from)
    return dict(
        completed_skills=" ".join(completed_skills),
        predicted_skills=" ".join(predicted_skills),
        ground_truth_sequence=ground_truth_sequence,
        high_level_skill=chained_subgoal_instr,
        rews=rewards,
        dones=dones,
        video_frames=video_frames if log_video else None,
        video_caption=vid_caption,
        chained_language_instruction=process_annotation_inference(
            chained_subgoal_instr,
            model.vocab_word,
        ),
        skill_length=env.num_subgoals_to_complete,
    )


def run_policy_multi_process(
    ret_queue,
    task_queue,
    config,
    device,
    offline_rl_model,
    saycan_planner,
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
        task_args = task_queue.get()
        if task_args is None:
            break
        with torch.no_grad():
            ret_queue.put(
                run_policy(
                    env,
                    offline_rl_model,
                    saycan_planner,
                    resnet,
                    device,
                    config.max_skill_length,
                    *task_args,
                )
            )
    env.stop()
