from boss.alfred.gen.constants import VISIBILITY_DISTANCE
from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image
import random
import torch
import torch
import numpy as np
import boss.alfred.gen.constants as constants
import random
import re
import argparse
import copy
import torch
import numpy as np
import textacy
import spacy

nlp = spacy.load("en_core_web_sm")
import matplotlib.pyplot as plt
import wandb
from collections import Counter, defaultdict

plot_color = {
    # "GotoLocation": "red",
    "PickupObject": "brown",
    "PutObject": "blue",
    "ToggleObject": "orange",
    "SliceObject": "black",
    "CleanObject": "purple",
    "HeatObject": "green",
    "CoolObject": "pink",
    "Composite": "red",
}
primitive_skill_types = [
    # "GotoLocation",
    "PickupObject",
    "PutObject",
    "ToggleObject",
    "SliceObject",
    "CleanObject",
    "HeatObject",
    "CoolObject",
]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __deepcopy__(self, memo):
        return AttrDict(copy.deepcopy(dict(self), memo))


####### General definitions/constants for discrete actions in the environment #####
visibility_distance = constants.VISIBILITY_DISTANCE

interactive_actions = [
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
]
knives = ["ButterKnife", "Knife"]
##########


def send_to_device_if_not_none(data_dict, entry_name, device):
    # helper function to send torch tensor to device if it is not None
    if entry_name not in data_dict or data_dict[entry_name] is None:
        return None
    else:
        return data_dict[entry_name].to(device)


def load_object_class(vocab_obj, object_name):
    """
    load object classes for interactive actions
    """
    if object_name is None:
        return 0
    object_class = object_name.split("|")[0]
    return vocab_obj.word2index(object_class)


def extract_item(possible_tensor):
    # if it is a tensor then extract the item otherwise just return it
    if isinstance(possible_tensor, torch.Tensor):
        return possible_tensor.item()
    return possible_tensor


def str2bool(v):
    # used for parsing boolean arguments
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def generate_invalid_action_mask_and_objects(env, visible_object, vocab_obj, vocab):
    # we will first filter out all the interact actions that are not available
    def filter_objects(condition):
        condition_input = condition
        if condition == "toggleon" or condition == "toggleoff":
            condition = "toggleable"

        if condition == "openable" or condition == "closeable":
            condition = "openable"

        visible_candidate_objects = [
            obj for obj in visible_object if obj[condition] == True
        ]

        candidate_obj_type = [
            vis_obj["objectId"].split("|")[0] for vis_obj in visible_candidate_objects
        ]

        remove_indicies = []

        if condition_input == "toggleon":
            if "Faucet" in candidate_obj_type:
                # SinkBasin: Sink|+03.08|+00.89|+00.09|SinkBasin
                visible_object_name = [
                    obj["objectId"].split("|")[-1] for obj in visible_object
                ]
                if "SinkBasin" not in visible_object_name:
                    remove_indicies.append(candidate_obj_type.index("Faucet"))

            for i, obj in enumerate(visible_candidate_objects):
                if (
                    obj["isToggled"] == True
                    and obj["objectId"].split("|")[0] in candidate_obj_type
                ):
                    remove_indicies.append(i)

        elif condition_input == "toggleoff":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["isToggled"] == False:
                    remove_indicies.append(i)

        elif condition_input == "openable":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["isOpen"] == True or obj["isToggled"] == True:
                    remove_indicies.append(i)

        elif condition_input == "closeable":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["isOpen"] == False:
                    remove_indicies.append(i)

        elif condition_input == "receptacle":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["openable"] == True and obj["isOpen"] == False:
                    remove_indicies.append(i)

        elif condition_input == "sliceable":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["isSliced"] == True:
                    remove_indicies.append(i)

        remove_indicies = set(remove_indicies)
        filtered_candidate_obj_type = [
            j for i, j in enumerate(candidate_obj_type) if i not in remove_indicies
        ]
        filtered_visible_candidate_objects = [
            j
            for i, j in enumerate(visible_candidate_objects)
            if i not in remove_indicies
        ]

        candidate_obj_type_id = [
            vocab_obj.word2index(candidate_obj_type_use)
            for candidate_obj_type_use in filtered_candidate_obj_type
            if candidate_obj_type_use in vocab_obj.to_dict()["index2word"]
        ]
        candidate_obj_type_id = np.array(list(set(candidate_obj_type_id)))
        return filtered_visible_candidate_objects, candidate_obj_type_id

    pickupable_object_names, pickupable_objects = filter_objects("pickupable")
    openable_object_names, openable_objects = filter_objects("openable")
    sliceable_object_names, sliceable_objects = filter_objects("sliceable")
    closeable_object_names, closeable_objects = filter_objects("closeable")
    receptacle_object_names, receptacle_objects = filter_objects("receptacle")

    toggleon_object_names, toggleon_objects = filter_objects("toggleon")
    toggleoff_object_names, toggleoff_objects = filter_objects("toggleoff")

    # generate invalid mask
    invalid_action_mask = []
    if (
        len(pickupable_objects) == 0
        or len(env.last_event.metadata["inventoryObjects"]) > 0
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("PickupObject") - 2)
    if len(openable_objects) == 0:
        invalid_action_mask.append(vocab["action_low"].word2index("OpenObject") - 2)
    if (
        len(sliceable_objects) == 0
        or len(env.last_event.metadata["inventoryObjects"]) == 0
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("SliceObject") - 2)
    if len(closeable_objects) == 0:
        invalid_action_mask.append(vocab["action_low"].word2index("CloseObject") - 2)
    if (
        len(receptacle_objects) == 0
        or len(env.last_event.metadata["inventoryObjects"]) == 0
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("PutObject") - 2)
    if len(toggleon_objects) == 0:
        invalid_action_mask.append(vocab["action_low"].word2index("ToggleObjectOn") - 2)
    if len(toggleoff_objects) == 0:
        invalid_action_mask.append(
            vocab["action_low"].word2index("ToggleObjectOff") - 2
        )
    if (
        len(env.last_event.metadata["inventoryObjects"]) > 0
        and env.last_event.metadata["inventoryObjects"][0]["objectId"].split("|")[0]
        not in knives
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("SliceObject") - 2)

    # <<stop>> action needs to be invalid
    invalid_action_mask.append(vocab["action_low"].word2index("<<stop>>") - 2)
    invalid_action_mask = list(set(invalid_action_mask))

    ret_dict = dict(
        pickupable_object_names=pickupable_object_names,
        pickupable_objects=pickupable_objects,
        openable_object_names=openable_object_names,
        openable_objects=openable_objects,
        sliceable_object_names=sliceable_object_names,
        sliceable_objects=sliceable_objects,
        closeable_object_names=closeable_object_names,
        closeable_objects=closeable_objects,
        receptacle_object_names=receptacle_object_names,
        receptacle_objects=receptacle_objects,
        toggleon_object_names=toggleon_object_names,
        toggleon_objects=toggleon_objects,
        toggleoff_object_names=toggleoff_object_names,
        toggleoff_objects=toggleoff_objects,
    )
    return invalid_action_mask, ret_dict


def generate_video(
    value_predictions,
    str_act,
    video_frames,
    env_rewards,
    primitive_skill_types=None,
    log_debugging_info=False,
):
    value_font = ImageFont.truetype("FreeMono.ttf", 20)
    action_font = ImageFont.truetype("FreeMono.ttf", 14)
    gif_logs = []
    for frame_number in range(len(video_frames)):
        img = video_frames[frame_number]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        if frame_number != 0:
            if log_debugging_info:
                if len(env_rewards) > 0:
                    reward_log = env_rewards[frame_number - 1]
                    draw.text(
                        (1, 260),
                        "Reward: %.1f" % (reward_log),
                        fill=(255, 255, 255),
                        font=value_font,
                    )
                    return_log = sum(env_rewards[0:frame_number])
                    draw.text(
                        (150, 260),
                        "Return: %.1f" % (return_log),
                        fill=(255, 255, 255),
                        font=value_font,
                    )

        if frame_number != len(video_frames) - 1:
            if log_debugging_info:
                if len(str_act) > 0:
                    action_log, object_log = (
                        str_act[frame_number]["action"],
                        str_act[frame_number]["object"],
                    )
                    draw.text(
                        (1, 1),
                        f"Action: {action_log}\nObject: {str(object_log)}",
                        fill=(255, 255, 255),
                        font=action_font,
                    )
                    if primitive_skill_types is not None:
                        draw.text(
                            (1, 31),
                            f"Skill: {primitive_skill_types[frame_number]}",
                            fill=(255, 255, 255),
                            font=action_font,
                        )

        if log_debugging_info:
            if len(value_predictions) != 0:
                value_log = value_predictions[frame_number]
                draw.text(
                    (1, 280),
                    "Value: %.3f" % (value_log),
                    fill=(255, 255, 255),
                    font=value_font,
                )
        log_images = np.array(img)
        gif_logs.append(log_images)

    video_frames = np.asarray(gif_logs)
    video_frames = np.transpose(video_frames, (0, 3, 1, 2))
    return video_frames


def add_prefix_to_skill_string(strings: list):
    prefix_skill_list = []
    for skill in strings:
        # get svo tuples
        # svo_tuples = findSVOs(nlp(process_skill_strings([f"I {generation}"])[0]))
        svo_tuples = textacy.extract.subject_verb_object_triples(nlp(f"I {skill}"))
        # turn it back into a sentence without the added I
        svo_tuples = list(svo_tuples)
        if len(svo_tuples) > 0:
            prefix_skill_list.append(f"{svo_tuples[0].verb[0].text.upper()}: {skill}")
        else:
            prefix_skill_list.append(skill)
    return prefix_skill_list


def process_skill_strings(strings):
    # process strings to all be proper sentences with punctuation and capitalization.
    if not isinstance(strings, list):
        strings = [strings]
    processed_strings = []
    for string in strings:
        if isinstance(string, list):
            # artifact of bug in the data
            string = string[0]
        string = string.strip().lower()
        string = re.sub(" +", " ", string)  # remove extra spaces
        if len(string) > 0 and string[-1] not in ["?", ".", "!"]:
            string = string + "."
        processed_strings.append(string.capitalize())
    return processed_strings


def compute_distance(agent_position, object):
    # computes xyz distance to an object
    agent_location = np.array(
        [agent_position["x"], agent_position["y"], agent_position["z"]]
    )
    object_location = np.array(
        [object["position"]["x"], object["position"]["y"], object["position"]["z"]]
    )

    distance = np.linalg.norm(agent_location - object_location)

    return distance


def compute_visibility_based_on_distance(agent_position, object, visibility_distance):
    # directly rewritten from C++ code here https://github.com/allenai/ai2thor/blob/f39ae981646d689047ba7006cb9c1dc507a247ff/unity/Assets/Scripts/BaseFPSAgentController.cs#L2628
    # used to figure out if an object is visible
    is_visible = True
    x_delta = object["position"]["x"] - agent_position["x"]
    y_delta = object["position"]["y"] - agent_position["y"]
    z_delta = object["position"]["z"] - agent_position["z"]
    if abs(x_delta) > visibility_distance:
        is_visible = False
    elif abs(y_delta) > visibility_distance:
        is_visible = False
    elif abs(z_delta) > visibility_distance:
        is_visible = False
    elif (
        x_delta * x_delta + z_delta * z_delta
        > visibility_distance * visibility_distance
    ):
        is_visible = False
    return is_visible


def mask_and_resample(action_probs, action_mask, deterministic, take_rand_action):
    # mask the action probabilities with the action mask (don't allow invalid actions)
    # then use those probabilities to produce an action
    action_probs[0, action_mask] = 0
    if torch.all(action_probs[0] == 0):
        # set the indicies NOT in action mask to 0
        action_mask_complement = np.ones(action_probs.shape[1], dtype=bool)
        action_mask_complement[action_mask] = False
        action_probs[0, action_mask_complement] = 1
    logprobs = torch.log(action_probs)
    logprobs[0, action_mask] = -100
    if deterministic:
        chosen_action = torch.argmax(action_probs)
    else:
        dist = torch.distributions.Categorical(logits=logprobs)
        chosen_action = dist.sample()
    if take_rand_action:
        action_mask_complement = np.ones(action_probs.shape[1], dtype=bool)
        # anything that doesn't get masked out by action_mask is in action_mask_complement
        action_mask_complement[action_mask] = False
        # set uniform probability for all valid actions
        # logprobs[0, action_mask_complement] = 0
        action_probs[0, action_mask_complement] = 1
        # sample uniformly
        dist = torch.distributions.Categorical(action_probs)
        chosen_action = dist.sample()
    return chosen_action


def get_action_from_agent(
    model,
    feat,
    vocab,
    vocab_obj,
    env,
    deterministic,
    epsilon,
    ret_value,
):
    take_rand_action = random.random() < epsilon

    action_out, object_pred_id, value = model.step(feat, ret_value=ret_value)

    action_out = torch.softmax(action_out, dim=1)

    object_pred_prob = torch.softmax(object_pred_id, dim=1)

    agent_position = env.last_event.metadata["agent"]["position"]

    visible_object = [
        obj
        for obj in env.last_event.metadata["objects"]
        if (
            obj["visible"] == True
            and compute_visibility_based_on_distance(
                agent_position, obj, VISIBILITY_DISTANCE
            )
        )
    ]
    invalid_action_mask, ret_dict = generate_invalid_action_mask_and_objects(
        env, visible_object, vocab_obj, vocab
    )
    # choose the action after filtering with the mask
    chosen_action = mask_and_resample(
        action_out, invalid_action_mask, deterministic, take_rand_action
    )
    string_act = vocab["action_low"].index2word(chosen_action + 2)
    assert string_act != "<<stop>>", breakpoint()
    if string_act not in interactive_actions:
        return string_act, None, value
    object_pred_prob = object_pred_prob.squeeze(0).cpu().detach().numpy()
    # otherwise, we need to choose the closest visible object for our action
    string_act_to_object_list_map = dict(
        PickupObject=(
            ret_dict["pickupable_object_names"],
            ret_dict["pickupable_objects"],
        ),
        OpenObject=(ret_dict["openable_object_names"], ret_dict["openable_objects"]),
        SliceObject=(ret_dict["sliceable_object_names"], ret_dict["sliceable_objects"]),
        CloseObject=(ret_dict["closeable_object_names"], ret_dict["closeable_objects"]),
        PutObject=(ret_dict["receptacle_object_names"], ret_dict["receptacle_objects"]),
        ToggleObjectOn=(
            ret_dict["toggleon_object_names"],
            ret_dict["toggleon_objects"],
        ),
        ToggleObjectOff=(
            ret_dict["toggleoff_object_names"],
            ret_dict["toggleoff_objects"],
        ),
    )

    candidate_object_names, candidate_object_ids = string_act_to_object_list_map[
        string_act
    ]
    prob_dict = {}
    for id in candidate_object_ids:
        if take_rand_action:
            prob_dict[id] = 1
        else:
            prob_dict[id] = object_pred_prob[id]
    prob_value = prob_dict.values()
    if deterministic:
        max_prob = max(prob_value)
        choose_id = [k for k, v in prob_dict.items() if v == max_prob][0]
    else:
        # sample from the object prob distribution
        object_probs = torch.tensor(list(prob_value), dtype=torch.float32)
        if torch.all(object_probs == 0):
            object_probs = torch.ones_like(object_probs)
        choose_id = torch.multinomial(object_probs, 1)[0].item()
        choose_id = list(prob_dict.keys())[choose_id]

    # choose the closest object
    object_type = vocab_obj.index2word(choose_id)
    candidate_objects = [
        obj
        for obj in candidate_object_names
        if obj["objectId"].split("|")[0] == object_type
    ]
    # object type
    agent_position = env.last_event.metadata["agent"]["position"]
    min_distance = float("inf")
    for ava_object in candidate_objects:
        obj_agent_dist = compute_distance(agent_position, ava_object)
        if obj_agent_dist < min_distance:
            min_distance = obj_agent_dist
            output_object = ava_object["objectId"]
    return string_act, output_object, value


def cleanup_mp(task_queue, processes):
    # generate termination signal for each worker
    for _ in range(len(processes)):
        task_queue.put(None)

    # wait for workers to terminate
    for worker in processes:
        worker.join()


def make_primitive_annotation_eval_dataset(eval_list: list[dict]):
    """
    Make a dataset for evaluation of primitive annotations, used for SayCan
    """
    new_eval_dataset = []
    for eval_dict in eval_list:
        eval_dict_copy = eval_dict.copy()
        annotations = []
        for primitive_skill in eval_dict_copy["primitive_skills"]:
            annotations.append(primitive_skill["annotations"])
        annotations = process_skill_strings(annotations)
        eval_dict_copy["annotation"] = " ".join(annotations)
        new_eval_dataset.append(eval_dict_copy)
    return new_eval_dataset


def generate_primitive_skill_list_from_eval_skill_info_list(
    primitive_eval_skill_info_list,
):
    primitive_skills_to_use = []
    for skill_info in primitive_eval_skill_info_list:
        primitive_skills_to_use.extend(
            [primitive_skill for primitive_skill in skill_info["primitive_skills"]]
        )
    for primitive_skill in primitive_skills_to_use:
        primitive_skill["api_actions"] = primitive_skill[
            "api_action"
        ]  # relabeling since online_reward.py expects api_actions

    def tuplify_dict_of_dicts(d):
        to_tuplify = []
        for k in sorted(d):
            if isinstance(d[k], dict):
                to_tuplify.append((k, tuplify_dict_of_dicts(d[k])))
            elif isinstance(d[k], list):
                inner_tuplify = []
                for item in d[k]:
                    if isinstance(item, list):
                        inner_tuplify.append(tuple(item))
                    else:
                        inner_tuplify.append(item)
                to_tuplify.append(tuple(inner_tuplify))
            else:
                to_tuplify.append((k, d[k]))
        return tuple(to_tuplify)

    # now remove duplicate primitive skills which is a list of dicts of inner dicts
    primitive_skill_set = set()
    unique_primitive_skills_to_use = []
    for primitive_skill in primitive_skills_to_use:
        if tuplify_dict_of_dicts(primitive_skill) not in primitive_skill_set:
            primitive_skill_set.add(tuplify_dict_of_dicts(primitive_skill))
            unique_primitive_skills_to_use.append(primitive_skill)
    return unique_primitive_skills_to_use


def make_color_coded_primitive_scatter_plot(
    flattened_values,
    flattened_skill_names,
    x_label,
    title,
    composite_value_masks=[],
    flattened_valid_masks=[],
    filter_invalid=True,
):
    value_counting = dict()
    for i in range(len(flattened_values)):
        value = flattened_values[i]
        skill_name = flattened_skill_names[i]
        valid = (
            1 if len(flattened_valid_masks) == 0 else float(flattened_valid_masks[i])
        )
        is_composite = (
            0 if len(composite_value_masks) == 0 else float(composite_value_masks[i])
        )
        # if skill_name.lower() == "gotolocation":
        #    skill = random.uniform(0.0, 1.0)
        #    point = np.array([value, skill]).reshape(1, -1)
        if skill_name.lower() == "pickupobject":
            skill = random.uniform(0, 1.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "putobject":
            skill = random.uniform(1.0, 2.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "toggleobject":
            skill = random.uniform(2.0, 3.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "sliceobject":
            skill = random.uniform(3.0, 4.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "heatobject":
            skill = random.uniform(4.0, 5.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "coolobject":
            skill = random.uniform(5.0, 6.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "cleanobject":
            skill = random.uniform(6.0, 7.0)
            point = np.array([value, skill, valid, is_composite]).reshape(1, -1)
        elif skill_name.lower() == "composite":
            skill = random.uniform(7.0, 8.0)
            point = np.array([value, skill, valid, True]).reshape(1, -1)
        if skill_name not in value_counting:
            value_counting[skill_name] = point
        else:
            value_counting[skill_name] = np.concatenate(
                (value_counting[skill_name], point), axis=0
            )
    return make_scatter_plot(value_counting, plot_color, x_label, title, filter_invalid)


def make_scatter_plot(
    key_to_value: dict,
    color_map: dict,
    x_label: str,
    title: str,
    filter_invalid: bool,
):
    fig, ax = plt.subplots()
    # first, aggregate all values to get the percentiles
    all_values = []
    for key in key_to_value:
        if filter_invalid:
            valid_mask = key_to_value[key][:, 2].astype(bool)
            all_values.append(key_to_value[key][valid_mask, 0])
        else:
            all_values.append(key_to_value[key][:, 0])
    all_values = np.concatenate(all_values, axis=0)
    percentiles = np.percentile(all_values, [10, 25, 50, 75, 90])
    # then, plot the percentiles with vertical dotted lines
    for i in range(len(percentiles)):
        ax.axvline(percentiles[i], color="black", linestyle="dotted", alpha=0.5)
    # then, plot the actual data
    for skill_name in key_to_value:
        color = color_map[skill_name]
        data = key_to_value[skill_name]
        value = data[:, 0]
        skill = data[:, 1]
        valid = data[:, 2].astype(bool)
        is_composite = data[:, 3].astype(bool)
        # scatter plot all valid primitive skills
        ax.scatter(
            value[valid & ~is_composite],
            skill[valid & ~is_composite],
            label=skill_name,
            c=color,
            s=10,
            alpha=0.3,
        )
        # scatter plot all invalid skills with marker x
        ax.scatter(
            value[~valid & ~is_composite],
            skill[~valid & ~is_composite],
            c=color,
            s=10,
            alpha=0.3,
            marker="x",
        )
        # scatter plot all valid composite skills with more alpha
        ax.scatter(
            value[valid & is_composite],
            skill[valid & is_composite],
            c=color,
            s=10,
            alpha=1.0,
            marker="*",
        )
        # scatter plot copmosite and invalid skills with marker x and more alpha
        ax.scatter(
            value[~valid & is_composite],
            skill[~valid & is_composite],
            c=color,
            s=10,
            alpha=1.0,
            marker="X",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Skill")
    ax.legend(bbox_to_anchor=(0.05, 1.0))
    ax.set_title(title)
    plt.tight_layout()
    ret_wandb_image = wandb.Image(fig)
    plt.close()
    plt.cla()
    plt.clf()
    return ret_wandb_image


def log_rollout_metrics(
    rollout_returns,
    successes,
    extra_info,
    rollout_gifs,
    video_captions,
    composite_skill_list: list[dict],
    config,
):
    # aggregate metrics
    rollout_metrics = dict(
        average_return=np.mean(rollout_returns),
        success=np.mean(successes),
    )
    if "new_skill_values" in extra_info:
        # make wandb histogram for values
        flattened_values = [
            item for sublist in extra_info["new_skill_values"] for item in sublist
        ]
        flattened_valid_masks = [
            item for sublist in extra_info["valid_masks"] for item in sublist
        ]
        flattened_composite_masks = [
            item for sublist in extra_info["is_composite"] for item in sublist
        ]

        # make scatter plot for values
        flattened_skill_names = [
            item
            for sublist in extra_info["new_skill_sampled_types"]
            for item in sublist
        ]
        rollout_metrics["new_skill_values_dist"] = (
            make_color_coded_primitive_scatter_plot(
                flattened_values,
                flattened_skill_names,
                "Value",
                "Skill Value Scatterplot",
                composite_value_masks=flattened_composite_masks,
                flattened_valid_masks=flattened_valid_masks,
                filter_invalid=True,
            )
        )
        # make scatter plot for llm probs
    flattened_llm_probs = []
    flattened_corresponding_values_for_llm_probs = []
    flattened_corresponding_valid_masks_for_llm_probs = []
    flattened_corresponding_composite_masks_for_llm_probs = []
    flattened_llm_skill_names = []
    if "new_skill_llm_probs" in extra_info:
        # make wandb histogram for llm probs
        for i, sublist in enumerate(extra_info["new_skill_llm_probs"]):
            if sublist is not None:
                flattened_llm_probs.extend(sublist)
                flattened_llm_skill_names.extend(
                    extra_info["new_skill_sampled_types"][i]
                )
                flattened_corresponding_values_for_llm_probs.extend(
                    extra_info["new_skill_values"][i]
                )
                flattened_corresponding_valid_masks_for_llm_probs.extend(
                    extra_info["valid_masks"][i]
                )
                # bug here with index error, i don't know why
                # flattened_corresponding_composite_masks_for_llm_probs.extend(
                #    extra_info["is_composite"][i]
                # )
        if len(flattened_llm_probs) > 0:
            # rollout_metrics["new_skill_llm_prob_hist"] = wandb.Histogram(
            #    flattened_llm_probs,
            # )
            rollout_metrics["new_skill_llm_dist"] = (
                make_color_coded_primitive_scatter_plot(
                    flattened_llm_probs,
                    flattened_llm_skill_names,
                    "LLM Probability",
                    "Skill LLM Prob Scatterplot",
                    # composite_value_masks=flattened_corresponding_composite_masks_for_llm_probs,
                    flattened_valid_masks=flattened_corresponding_valid_masks_for_llm_probs,
                    filter_invalid=True,
                )
            )
        # replace with this in extra_info because of the nones
        extra_info["new_skill_llm_probs"] = flattened_llm_probs
        # plot bar chart of sampled new skill types llm prob distribution
        skill_type_to_llm_probs = defaultdict(list)
        for skill_type, llm_probs in zip(
            flattened_llm_skill_names, flattened_llm_probs
        ):
            skill_type_to_llm_probs[skill_type].append(llm_probs)
        for primitive_skill_type in primitive_skill_types:
            if len(skill_type_to_llm_probs[primitive_skill_type]) > 0:
                skill_type_to_llm_probs[primitive_skill_type] = np.mean(
                    skill_type_to_llm_probs[primitive_skill_type]
                )
            else:
                skill_type_to_llm_probs[primitive_skill_type] = 0
        # plot bar chart of sampled new skill types llm prob distribution
        skill_probability_dist = torch.softmax(
            (
                torch.tensor(flattened_corresponding_values_for_llm_probs)
                / config.value_sampling_temp
            )
            ** (1 - config.llm_logprob_weight)
            * (torch.tensor(flattened_llm_probs) / config.llm_sampling_temp)
            ** config.llm_logprob_weight,
            dim=0,
        ).numpy()
        skill_probability_dist[
            ~np.array(flattened_corresponding_valid_masks_for_llm_probs).astype(
                bool
            )
        ] = 0
        skill_type_to_vf_times_llm_probs = defaultdict(list)
        for skill_type, vf_times_llm_probs in zip(
            flattened_llm_skill_names, skill_probability_dist
        ):
            skill_type_to_vf_times_llm_probs[skill_type].append(vf_times_llm_probs)
        for primitive_skill_type in primitive_skill_types:
            if len(skill_type_to_vf_times_llm_probs[primitive_skill_type]) > 0:
                skill_type_to_vf_times_llm_probs[primitive_skill_type] = np.mean(
                    skill_type_to_vf_times_llm_probs[primitive_skill_type]
                )
            else:
                skill_type_to_vf_times_llm_probs[primitive_skill_type] = 0
        if len(flattened_llm_skill_names) > 0:
            rollout_metrics["new_skill_values_times_llm_prob_dist"] = (
                make_color_coded_primitive_scatter_plot(
                    skill_probability_dist,
                    flattened_llm_skill_names,
                    f"Skill Value^({1 - config.llm_logprob_weight:0.2f}) * LLM Prob^{config.llm_logprob_weight:0.2f}, Value Temp: {config.value_sampling_temp:0.2f} LLM Temp: {config.llm_sampling_temp:0.2f}",
                    "Skill Value * LLM Prob Scatterplot",
                    composite_value_masks=flattened_corresponding_composite_masks_for_llm_probs,
                    flattened_valid_masks=flattened_corresponding_valid_masks_for_llm_probs,
                    filter_invalid=True,
                )
            )
    num_primitive_skills_attempted = extra_info["num_primitive_skills_attempted"]
    # these are None if no new skills were sampled, otherwise one less than num_primitive_skills_attempted
    # generate per-length return and subgoal success data
    per_number_return = defaultdict(list)
    per_number_success = defaultdict(list)
    for i, num_attempts in enumerate(num_primitive_skills_attempted):
        per_number_return[num_attempts].append(rollout_returns[i])
        per_number_success[num_attempts].append(successes[i])
    for num_attempts, returns in per_number_return.items():
        # log the averages
        rollout_metrics[f"length_{num_attempts}_return"] = np.mean(returns)
        rollout_metrics[f"length_{num_attempts}_success"] = np.mean(
            per_number_success[num_attempts]
        )
        # log a histogram to wandb
        rollout_metrics[f"length_{num_attempts}_return_distribution"] = make_bar_chart(
            values=np.array(returns),
            labels=list(range(len(returns))),
            title=f"Length {num_attempts} Return Distribution",
            xlabel="Which Task",
            ylabel="Return",
            ylim=None,
        )
        rollout_metrics[f"length_{num_attempts}_success_dist"] = make_bar_chart(
            values=np.array(per_number_success[num_attempts]),
            labels=list(range(len(per_number_success[num_attempts]))),
            title=f"Length {num_attempts} Success Distribution",
            xlabel="Which Task",
            ylabel="Success",
            ylim=None,
        )
    if "first_skill_length" in extra_info:
        first_skill_lengths = extra_info["first_skill_length"]
        second_skill_lengths = extra_info["second_skill_length"]
        first_lengths_to_returns = defaultdict(list)
        first_lengths_to_successes = defaultdict(list)
        second_lengths_to_returns = defaultdict(list)
        second_lengths_to_successes = defaultdict(list)
        for i, (first_skill_length, second_skill_length) in enumerate(
            zip(first_skill_lengths, second_skill_lengths)
        ):
            first_lengths_to_returns[first_skill_length].append(
                extra_info["first_skill_return"][i]
            )
            first_lengths_to_successes[first_skill_length].append(
                extra_info["first_skill_success"][i]
            )
            if second_skill_length is not None:
                second_lengths_to_returns[second_skill_length].append(
                    extra_info["second_skill_return"][i]
                )
                second_lengths_to_successes[second_skill_length].append(
                    extra_info["second_skill_success"][i]
                )
        for first_skill_length, returns in first_lengths_to_returns.items():
            rollout_metrics[f"first_skill_length_{first_skill_length}_return"] = (
                np.mean(returns)
            )
            rollout_metrics[f"first_skill_length_{first_skill_length}_success"] = (
                np.mean(first_lengths_to_successes[first_skill_length])
            )
        # also do overall success
        rollout_metrics["first_skill_success"] = np.mean(
            extra_info["first_skill_success"]
        )
        for second_skill_length, returns in second_lengths_to_returns.items():
            rollout_metrics[f"second_skill_length_{second_skill_length}_return"] = (
                np.mean(returns)
            )
            rollout_metrics[f"second_skill_length_{second_skill_length}_success"] = (
                np.mean(second_lengths_to_successes[second_skill_length])
            )
        # also do overall success
        if len(second_lengths_to_returns) > 0:
            rollout_metrics["second_skill_success"] = np.mean(
                [np.mean(v) for v in second_lengths_to_returns.values()]
            )
        extra_info.pop("first_skill_length")
        extra_info.pop("first_skill_return")
        extra_info.pop("first_skill_success")
        extra_info.pop("second_skill_length")
        extra_info.pop("second_skill_return")
        extra_info.pop("second_skill_success")

    # log composite skill original and new names to wandb table
    composite_skill_name_data = []
    for skill_dict in composite_skill_list:
        skill = skill_dict["skill"]
        composed_primitives = " ".join(skill.primitive_instructions_to_compose)
        joined_name = skill.composite_language_instruction
        llm_prob = skill.llm_prob
        scene_index = skill_dict["scene_index"]
        length = skill.num_skills
        composite_skill_name_data.append(
            [composed_primitives, joined_name, llm_prob, scene_index, length]
        )
    table = wandb.Table(
        columns=["Primitives", "Name", "LLM Prob", "Scene Index", "Length"],
        data=composite_skill_name_data,
    )
    rollout_metrics["composite_skill_table"] = table

    for key, value in extra_info.items():
        if key == "primitive_skill_types":
            counted_skills = Counter(value)
            data = np.array(
                [counted_skills[skill_type] for skill_type in primitive_skill_types]
            )
            data = data / data.sum()
            # plt bar chart because wandb bar chart doesn't update
            rollout_metrics[f"used_primitive_skill_dist"] = make_bar_chart(
                values=data,
                labels=primitive_skill_types,
                title="Used Primitive Skill Distribution",
                xlabel="Skill Type",
                ylabel="Frequency",
                ylim=(0, 1),
            )
        elif key == "new_skill_sampled_types":
            # plot bar chart of sampled new skill types frequency distribution
            # then plot bar chart of their average values, llm probs, and the product of the two
            vf_values = extra_info["new_skill_values"]
            frequency_counter = Counter()

            skill_type_to_average_value = defaultdict(list)
            # compute average value and llm prob for each skill type
            for i, sample in enumerate(value):
                for j, skill_type in enumerate(sample):
                    skill_type_to_average_value[skill_type].append(vf_values[i][j])
                sample_counter = Counter(sample)
                frequency_counter.update(sample_counter)
            for key in primitive_skill_types:
                if len(skill_type_to_average_value[key]) > 0:
                    skill_type_to_average_value[key] = np.mean(
                        skill_type_to_average_value[key]
                    )
                else:
                    skill_type_to_average_value[key] = 0

            # plot bar chart of sampled new skill types frequency distribution
            freq_data = np.array(
                [frequency_counter[skill_type] for skill_type in primitive_skill_types]
            )
            freq_data = freq_data / freq_data.sum()
            rollout_metrics[f"sampled_primitive_skill_distribution"] = make_bar_chart(
                values=freq_data,
                labels=primitive_skill_types,
                title="Sampled Primitive Skill Distribution",
                xlabel="Skill Type",
                ylabel="Frequency",
                ylim=(0, 1),
            )

        else:
            if len(value) > 0 and not isinstance(value[0], str):
                rollout_metrics[f"{key} Mean"] = np.mean(value)
                rollout_metrics[f"{key} Min"] = np.mean(np.min(value, axis=-1))
                rollout_metrics[f"{key} Max"] = np.mean(np.max(value, axis=-1))
            else:
                print(f"warning: {key} has no values")

    if len(rollout_gifs) > 0:
        # sort both rollout_gifs and video_captions by the caption so that we have a consistent ordering
        rollout_gifs, video_captions = zip(
            *sorted(zip(rollout_gifs, video_captions), key=lambda x: x[1])
        )
        for i, (gif, caption) in enumerate(zip(rollout_gifs, video_captions)):
            rollout_metrics["videos_%d" % i] = wandb.Video(
                gif, caption=caption, fps=3, format="mp4"
            )
    return rollout_metrics


def make_bar_chart(values, labels, title, xlabel, ylabel, ylim):
    """
    make a bar chart from a list of values and labels
    """

    # plt.figure(figsize=(10, 5))
    # plt.bar(labels, values)
    plt.bar(range(len(values)), values, tick_label=labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    ax = plt.gca()
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45)
    plt.tight_layout()
    ret_wandb_image = wandb.Image(plt)
    plt.close()
    plt.cla()
    plt.clf()
    return ret_wandb_image
