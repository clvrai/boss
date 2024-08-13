import os
import sys
import progressbar
import string

if "ALFRED_ROOT" not in os.environ:
    os.environ["ALFRED_ROOT"] = "/home/jzhang96/ALFRED_jiahui"
import os
import sys

sys.path.append(os.path.join(os.environ["ALFRED_ROOT"]))
sys.path.append(os.path.join(os.environ["ALFRED_ROOT"], "gen"))
sys.path.append(os.path.join(os.environ["ALFRED_ROOT"], "env"))

import json
import glob
import os
import constants
import cv2
import shutil
import numpy as np
import argparse
import threading
import time
import copy
import random
from boss.alfred.gen.utils.video_util import VideoSaver
from boss.alfred.gen.utils.py_util import walklevel
from boss.alfred.env.online_thor_env import OnlineThorEnv

if not os.path.exists("../direct_graph/"):
    os.mkdir("../direct_graph/")
    os.mkdir("../direct_graph/fail/")
    os.mkdir("../direct_graph/success/")
    os.mkdir("../direct_graph/success/moveable/")
    os.mkdir("../direct_graph/success/non_moveable/")
    os.mkdir("../direct_graph/fail/moveable/")
    os.mkdir("../direct_graph/fail/non_moveable/")
    os.mkdir("../direct_graph/fail/moveable/valid/")
    os.mkdir("../direct_graph/fail/moveable/invalid/")
    os.mkdir("../direct_graph/fail/non_moveable/valid/")
    os.mkdir("../direct_graph/fail/non_moveable/invalid/")
moveable_object = json.loads(open("../scene_sampling/moveable_object.json").read())
non_moveable_object = json.loads(
    open("../scene_sampling/non_moveable_object.json").read()
)

TRAJ_DATA_JSON_FILENAME = "traj_data.json"
AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"

ORIGINAL_IMAGES_FORLDER = "raw_images"
HIGH_RES_IMAGES_FOLDER = "high_res_images"
DEPTH_IMAGES_FOLDER = "depth_images"
INSTANCE_MASKS_FOLDER = "instance_masks"

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

render_settings = dict()
render_settings["renderImage"] = True
render_settings["renderDepthImage"] = False
render_settings["renderObjectImage"] = False
render_settings["renderClassImage"] = False

video_saver = VideoSaver()


# f = open('with_goto_log_not_use_origin_pose_video_0.85.log', 'a')
# sys.stdout = f


def compute_distance_2D(agent_position, object):

    agent_location = np.array([agent_position["x"], agent_position["y"]])
    object_location = np.array([object["position"]["x"], object["position"]["y"]])

    distance = np.linalg.norm(agent_location - object_location)

    return distance


def get_image_index(save_path):
    return len(glob.glob(save_path + "/*.png")) + len(glob.glob(save_path + "/*.jpg"))


def save_image_with_delays(env, action, save_path, direction=constants.BEFORE):
    if not args.only_reward_relabel:
        im_ind = get_image_index(save_path)
        counts = constants.SAVE_FRAME_BEFORE_AND_AFTER_COUNTS[action["action"]][
            direction
        ]
        for i in range(counts):
            save_image(env.last_event, save_path)
            env.noop()
        return im_ind


def save_image(event, save_path):
    if not args.only_reward_relabel:
        # rgb
        rgb_save_path = os.path.join(save_path, HIGH_RES_IMAGES_FOLDER)
        rgb_image = event.frame[:, :, ::-1]

        # depth
        if render_settings["renderDepthImage"]:
            depth_save_path = os.path.join(save_path, DEPTH_IMAGES_FOLDER)
            depth_image = event.depth_frame
            depth_image = depth_image * (255 / 10000)
            depth_image = depth_image.astype(np.uint8)

        # masks
        mask_save_path = os.path.join(save_path, INSTANCE_MASKS_FOLDER)
        mask_image = event.instance_segmentation_frame

        # dump images
        im_ind = get_image_index(rgb_save_path)
        cv2.imwrite(rgb_save_path + "/%09d.png" % im_ind, rgb_image)
        if render_settings["renderDepthImage"]:
            cv2.imwrite(depth_save_path + "/%09d.png" % im_ind, depth_image)
        cv2.imwrite(mask_save_path + "/%09d.png" % im_ind, mask_image)

        return im_ind


def save_images_in_events(events, root_dir):
    for event in events:
        save_image(event, root_dir)


def clear_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def augment_traj(env, json_file):
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)

    # make directories
    root_dir = json_file.replace(TRAJ_DATA_JSON_FILENAME, "")

    orig_images_dir = os.path.join(root_dir, ORIGINAL_IMAGES_FORLDER)
    high_res_images_dir = os.path.join(root_dir, HIGH_RES_IMAGES_FOLDER)
    depth_images_dir = os.path.join(root_dir, DEPTH_IMAGES_FOLDER)
    instance_masks_dir = os.path.join(root_dir, INSTANCE_MASKS_FOLDER)
    augmented_json_file = os.path.join(root_dir, AUGMENTED_TRAJ_DATA_JSON_FILENAME)

    if not os.path.exists(augmented_json_file):
        # fresh images list
        if not args.only_reward_relabel:
            traj_data["images"] = list()

            clear_and_create_dir(high_res_images_dir)
            clear_and_create_dir(depth_images_dir)
            clear_and_create_dir(instance_masks_dir)

        # scene setup
        scene_num = traj_data["scene"]["scene_num"]
        object_poses = traj_data["scene"]["object_poses"]
        object_toggles = traj_data["scene"]["object_toggles"]
        dirty_and_empty = traj_data["scene"]["dirty_and_empty"]

        # reset
        scene_name = "FloorPlan%d" % scene_num
        env.test_reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        env.step(dict(traj_data["scene"]["init_action"]))
        # print("Task: %s" % (traj_data['template']['task_desc']))

        # setup task
        # env.set_task(traj_data, args, reward_type="dense")
        rewards = []

        def generate_skill_info(traj_data, index):
            prev_hl_index = traj_data["plan"]["high_pddl"][index]["high_idx"]
            skill_info_for_online_env = {}
            current_api_actions = [
                action["discrete_action"]["action"]
                for action in traj_data["plan"]["low_actions"]
                if action["high_idx"] == prev_hl_index
            ]
            skill_info_for_online_env["api_actions"] = current_api_actions
            skill_info_for_online_env["planner_action"] = traj_data["plan"][
                "high_pddl"
            ][index]["planner_action"]
            skill_info_for_online_env["discrete_action"] = traj_data["plan"][
                "high_pddl"
            ][index]["discrete_action"]
            skill_info_for_online_env["annotations"] = traj_data["turk_annotations"][
                "anns"
            ][0]["high_descs"][index]
            skill_info_for_online_env["args_object"] = []
            skill_info_for_online_env["args_object"].append("None")
            if skill_info_for_online_env["discrete_action"]["action"] == "GotoLocation":
                skill_info_for_online_env["args_object"].append(
                    skill_info_for_online_env["discrete_action"]["args"][0]
                )

            return skill_info_for_online_env

        prev_hl_action = traj_data["plan"]["high_pddl"][0]
        prev_skill_info = generate_skill_info(traj_data, 0)
        set_subskill_success = env.set_subskill_type(prev_skill_info)
        if not set_subskill_success:
            print(f"Failed to set subskill type, {prev_skill_info}")
            return
        curr_hl_rewards = []
        log_to_video = []
        receptacle_names = []
        non_moveable_agent_object_graph_distance = []
        moveable_agent_object_graph_distance = []
        receptacle_num = []
        obj_in_ann = []
        rec_in_ann = []
        object_move = []
        skill_valid = []
        org_tar_pose = []
        close_tar_pose = []
        agent_object_distance = []
        gt_pose = []
        target_obj_pose = []
        close_object = []
        candidate_objects = []
        current_pose = []

        for ll_idx, ll_action in enumerate(traj_data["plan"]["low_actions"]):
            # next cmd under the current hl_action
            cmd = ll_action["api_action"]
            hl_action = traj_data["plan"]["high_pddl"][ll_action["high_idx"]]
            if prev_hl_action != hl_action:
                # if (
                #     curr_hl_rewards[-1] < 1 or sum(curr_hl_rewards) > 1
                # ) and "GotoLocation" != prev_hl_action["discrete_action"]["action"]:

                if curr_hl_rewards[-1] < 1 or sum(curr_hl_rewards) > 1:

                    ll_actions = [
                        action["api_action"]["action"]
                        for action in traj_data["plan"]["low_actions"][:ll_idx]
                        if action["high_idx"] == prev_hl_action["high_idx"]
                    ]
                    video_name = prev_skill_info["annotations"]
                    video_name = video_name.translate(
                        str.maketrans("", "", string.punctuation)
                    )
                    # sub_goal_save_path = os.path.join("../move_1_nonmove_0", video_name.replace(" ", "_"))+".mp4"

                    if prev_skill_info["args_object"][-1] in moveable_object:
                        if skill_valid[0]:
                            sub_goal_save_path = (
                                os.path.join(
                                    "../direct_graph",
                                    "fail",
                                    "moveable",
                                    "valid",
                                    video_name.replace(" ", "_"),
                                )
                                + ".mp4"
                            )
                        else:
                            sub_goal_save_path = (
                                os.path.join(
                                    "../direct_graph",
                                    "fail",
                                    "moveable",
                                    "invalid",
                                    video_name.replace(" ", "_"),
                                )
                                + ".mp4"
                            )
                    else:
                        if skill_valid[0]:
                            sub_goal_save_path = (
                                os.path.join(
                                    "../direct_graph",
                                    "fail",
                                    "non_moveable",
                                    "valid",
                                    video_name.replace(" ", "_"),
                                )
                                + ".mp4"
                            )
                        else:
                            sub_goal_save_path = (
                                os.path.join(
                                    "../direct_graph",
                                    "fail",
                                    "non_moveable",
                                    "invalid",
                                    video_name.replace(" ", "_"),
                                )
                                + ".mp4"
                            )
                    writer = cv2.VideoWriter(
                        sub_goal_save_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        5,
                        (300, 300),
                        True,
                    )

                    for frame_num in range(len(log_to_video)):
                        frame = log_to_video[frame_num]
                        frame = frame[:, :, [2, 1, 0]]
                        reward_for_frame = curr_hl_rewards[frame_num]

                        frame = np.ascontiguousarray(frame, dtype=np.uint8)
                        non_graph_dis = non_moveable_agent_object_graph_distance[
                            frame_num
                        ]
                        mov_graph_dis = moveable_agent_object_graph_distance[frame_num]
                        agent_obj_dis = agent_object_distance[frame_num]
                        # rec_num = receptacle_num[frame_num]
                        # obj_in_annotation = obj_in_ann[frame_num]
                        # rec_in_annotation = rec_in_ann[frame_num]
                        object_moveable = object_move[frame_num]
                        valid = skill_valid[frame_num]
                        dataset_pose = gt_pose[frame_num]
                        tar_obj_pose = target_obj_pose[frame_num]
                        close_obj = close_object[frame_num]
                        cand_obj = candidate_objects[frame_num]
                        curr_pose = current_pose[frame_num]
                        # org_tar = org_tar_pose[frame_num]
                        # close_tar = close_tar_pose[frame_num]

                        frame = cv2.putText(
                            img=frame,
                            text="reward %.3f" % (reward_for_frame),
                            org=(1, 295),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text="agent_obj_dis: %s" % (agent_obj_dis),
                            org=(1, 275),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text=("Args: " + prev_skill_info["args_object"][-1]),
                            org=(1, 285),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text="non_graph_dis: %s" % (non_graph_dis),
                            org=(170, 285),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text=("mov_graph_dis: %s" % str(mov_graph_dis)),
                            org=(170, 295),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text="object_mov: %s" % (object_moveable),
                            org=(170, 275),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text="gt_pose: %s" % (dataset_pose),
                            org=(1, 265),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )
                        frame = cv2.putText(
                            img=frame,
                            text=("tar_pose: %s" % tar_obj_pose),
                            org=(160, 265),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text=("close_obj: %s" % close_obj),
                            org=(170, 255),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(255, 255, 255),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text=("valid_skill: %s" % valid),
                            org=(1, 255),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text=("Cand obj: %s" % cand_obj),
                            org=(1, 245),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text=("curr pose: %s" % curr_pose),
                            org=(1, 235),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        writer.write(frame)
                    print("Saving video: %s" % sub_goal_save_path)
                    writer.release()

                    print(
                        f"Previous HL Action: {prev_hl_action['discrete_action']}, Rewards: {curr_hl_rewards}, Language Instruction: {traj_data['turk_annotations']['anns'][0]['high_descs'][prev_hl_action['high_idx']]}, Actions: {ll_actions}"
                    )
                    raise Exception("Unmatched HL Action")

                elif prev_skill_info["discrete_action"]["action"] == "GotoLocation":

                    ll_actions = [
                        action["api_action"]["action"]
                        for action in traj_data["plan"]["low_actions"][:ll_idx]
                        if action["high_idx"] == prev_hl_action["high_idx"]
                    ]
                    video_name = prev_skill_info["annotations"]
                    video_name = video_name.translate(
                        str.maketrans("", "", string.punctuation)
                    )
                    # sub_goal_save_path = os.path.join("../move_1_nonmove_0", video_name.replace(" ", "_"))+".mp4"

                    if prev_skill_info["args_object"][-1] in moveable_object:

                        sub_goal_save_path = (
                            os.path.join(
                                "../direct_graph",
                                "success",
                                "moveable",
                                video_name.replace(" ", "_"),
                            )
                            + ".mp4"
                        )
                    else:

                        sub_goal_save_path = (
                            os.path.join(
                                "../direct_graph",
                                "success",
                                "non_moveable",
                                video_name.replace(" ", "_"),
                            )
                            + ".mp4"
                        )

                    writer = cv2.VideoWriter(
                        sub_goal_save_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        5,
                        (300, 300),
                        True,
                    )

                    for frame_num in range(len(log_to_video)):
                        frame = log_to_video[frame_num]
                        frame = frame[:, :, [2, 1, 0]]
                        reward_for_frame = curr_hl_rewards[frame_num]

                        frame = np.ascontiguousarray(frame, dtype=np.uint8)
                        non_graph_dis = non_moveable_agent_object_graph_distance[
                            frame_num
                        ]
                        mov_graph_dis = moveable_agent_object_graph_distance[frame_num]
                        agent_obj_dis = agent_object_distance[frame_num]
                        # rec_num = receptacle_num[frame_num]
                        # obj_in_annotation = obj_in_ann[frame_num]
                        # rec_in_annotation = rec_in_ann[frame_num]
                        object_moveable = object_move[frame_num]
                        valid = skill_valid[frame_num]
                        dataset_pose = gt_pose[frame_num]
                        tar_obj_pose = target_obj_pose[frame_num]
                        close_obj = close_object[frame_num]
                        cand_obj = candidate_objects[frame_num]
                        curr_pose = current_pose[frame_num]
                        # org_tar = org_tar_pose[frame_num]
                        # close_tar = close_tar_pose[frame_num]

                        frame = cv2.putText(
                            img=frame,
                            text="reward %.3f" % (reward_for_frame),
                            org=(1, 295),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text="agent_obj_dis: %s" % (agent_obj_dis),
                            org=(1, 275),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text=("Args: " + prev_skill_info["args_object"][-1]),
                            org=(1, 285),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text="non_graph_dis: %s" % (non_graph_dis),
                            org=(170, 285),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text=("mov_graph_dis: %s" % str(mov_graph_dis)),
                            org=(170, 295),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text="object_mov: %s" % (object_moveable),
                            org=(170, 275),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text="gt_pose: %s" % (dataset_pose),
                            org=(1, 265),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(255, 255, 255),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )
                        frame = cv2.putText(
                            img=frame,
                            text=("tar_pose: %s" % tar_obj_pose),
                            org=(160, 265),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text=("close_obj: %s" % close_obj),
                            org=(170, 255),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text=("valid_skill: %s" % valid),
                            org=(1, 255),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text=("Cand obj: %s" % cand_obj),
                            org=(1, 245),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        frame = cv2.putText(
                            img=frame,
                            text=("curr pose: %s" % curr_pose),
                            org=(1, 235),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.36,
                            color=(200, 200, 200),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=False,
                        )

                        writer.write(frame)
                    print("Saving video: %s" % sub_goal_save_path)
                    writer.release()

                prev_hl_action = hl_action
                prev_skill_info = generate_skill_info(traj_data, ll_action["high_idx"])
                set_subskill_success = env.set_subskill_type(prev_skill_info)
                if not set_subskill_success:
                    print(f"Failed to set subskill type, {prev_skill_info}")
                    return
                curr_hl_rewards = []
                log_to_video = []
                receptacle_names = []
                non_moveable_agent_object_graph_distance = []
                moveable_agent_object_graph_distance = []
                receptacle_num = []
                obj_in_ann = []
                rec_in_ann = []
                object_move = []
                skill_valid = []
                org_tar_pose = []
                close_tar_pose = []
                agent_object_distance = []
                gt_pose = []
                target_obj_pose = []
                close_object = []
                candidate_objects = []
                current_pose = []

            # remove unnecessary keys
            cmd = {
                k: cmd[k]
                for k in [
                    "action",
                    "objectId",
                    "receptacleObjectId",
                    "placeStationary",
                    "forceAction",
                ]
                if k in cmd
            }

            if "MoveAhead" in cmd["action"]:
                if args.smooth_nav:
                    save_image(env.last_event, root_dir)
                    events = env.smooth_move_ahead(cmd, render_settings)
                    # save_images_in_events(events, root_dir)
                    event = events[-1]
                else:
                    save_image(env.last_event, root_dir)
                    event = env.step(cmd)

            elif "Rotate" in cmd["action"]:
                if args.smooth_nav:
                    save_image(env.last_event, root_dir)
                    events = env.smooth_rotate(cmd, render_settings)
                    save_images_in_events(events, root_dir)
                    event = events[-1]
                else:
                    save_image(env.last_event, root_dir)
                    event = env.step(cmd)

            elif "Look" in cmd["action"]:
                if args.smooth_nav:
                    save_image(env.last_event, root_dir)
                    events = env.smooth_look(cmd, render_settings)
                    save_images_in_events(events, root_dir)
                    event = events[-1]
                else:
                    save_image(env.last_event, root_dir)
                    event = env.step(cmd)

            # handle the exception for CoolObject tasks where the actual 'CoolObject' action is actually 'CloseObject'
            # TODO: a proper fix for this issue
            elif (
                "CloseObject" in cmd["action"]
                and "CoolObject" in hl_action["planner_action"]["action"]
                and "OpenObject"
                in traj_data["plan"]["low_actions"][ll_idx + 1]["api_action"]["action"]
            ):
                if args.time_delays:
                    cool_action = hl_action["planner_action"]
                    save_image_with_delays(
                        env, cool_action, save_path=root_dir, direction=constants.BEFORE
                    )
                    event = env.step(cmd)
                    save_image_with_delays(
                        env, cool_action, save_path=root_dir, direction=constants.MIDDLE
                    )
                    save_image_with_delays(
                        env, cool_action, save_path=root_dir, direction=constants.AFTER
                    )
                else:
                    save_image(env.last_event, root_dir)
                    event = env.step(cmd)

            else:
                if args.time_delays:
                    save_image_with_delays(
                        env, cmd, save_path=root_dir, direction=constants.BEFORE
                    )
                    event = env.step(cmd)
                    save_image_with_delays(
                        env, cmd, save_path=root_dir, direction=constants.MIDDLE
                    )
                    save_image_with_delays(
                        env, cmd, save_path=root_dir, direction=constants.AFTER
                    )
                else:
                    save_image(env.last_event, root_dir)
                    event = env.step(cmd)
            # update last event with the original intended action

            if args.smooth_nav:
                env.last_event.metadata["lastAction"] = cmd["action"]

            # update image list
            new_img_idx = get_image_index(high_res_images_dir)
            last_img_idx = len(traj_data["images"])
            num_new_images = new_img_idx - last_img_idx
            if not args.only_reward_relabel:
                for j in range(num_new_images):
                    traj_data["images"].append(
                        {
                            "low_idx": ll_idx,
                            "high_idx": ll_action["high_idx"],
                            "image_name": "%09d.png" % int(last_img_idx + j),
                        }
                    )

            if not event.metadata["lastActionSuccess"]:
                raise Exception(
                    "Replay Failed: %s" % (env.last_event.metadata["errorMessage"])
                )

            rew = env.test_get_done()
            rewards.append(rew)
            curr_hl_rewards.append(rew)
            log_to_video.append(np.uint8(event.frame))

            non_moveable_agent_object_graph_distance.append(
                env._current_action.non_moveable_agent_object_graph_distance
            )
            moveable_agent_object_graph_distance.append(
                env._current_action.moveable_agent_object_graph_distance
            )
            agent_object_distance.append(env._current_action.agent_object_distance)
            object_move.append(env._current_action.object_move)
            skill_valid.append(env._current_action.valid_skill)
            gt_pose.append(env._current_action.gt_pose)
            target_obj_pose.append(env._current_action.target_obj_pose)
            close_object.append(env._current_action.close_obj)
            candidate_objects.append(env._current_action.cand_obj)
            current_pose.append(env._current_action.current_pose)

        # save 10 frames in the end as per the training data
        # for _ in range(10):
        #    save_image(env.last_event, root_dir)

        # store color to object type dictionary
        color_to_obj_id_type = {}
        all_objects = env.last_event.metadata["objects"]
        for color, object_id in env.last_event.color_to_object_id.items():
            for obj in all_objects:
                if object_id == obj["objectId"]:
                    color_to_obj_id_type[str(color)] = {
                        "objectID": obj["objectId"],
                        "objectType": obj["objectType"],
                    }

        augmented_traj_data = copy.deepcopy(traj_data)
        augmented_traj_data["scene"]["color_to_object_type"] = color_to_obj_id_type
        augmented_traj_data["task"] = {
            "rewards": rewards,
            "reward_upper_bound": sum(rewards),
        }

        # check if number of new images is the same as the number of original images
        if args.only_reward_relabel:
            if args.smooth_nav and args.time_delays:
                orig_action_count = len(traj_data["plan"]["low_actions"])
                new_action_count = len(rewards)
                print(
                    "Original Action Count %d, New Action Count %d"
                    % (orig_action_count, new_action_count)
                )
                if orig_action_count != new_action_count:
                    raise Exception(
                        "WARNING: the augmented sequence length doesn't match the original"
                    )

        # save video
        if not args.only_reward_relabel:
            images_path = os.path.join(high_res_images_dir, "*.png")
            video_save_path = os.path.join(high_res_images_dir, "high_res_video.mp4")
            video_saver.save(images_path, video_save_path)


def run():
    """
    replay loop
    """
    # start THOR env
    env = OnlineThorEnv()

    skipped_files = []
    traj_list_length = len(traj_list)

    while len(traj_list) > 0:
        lock.acquire()
        # print(len(traj_list))
        json_file = traj_list.pop()
        lock.release()

        # print("Augmenting: " + json_file)
        try:
            augment_traj(env, json_file)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Error: " + repr(e))
            print("Skipping " + json_file)
            skipped_files.append(json_file)

    env.stop()
    print("Finished.")

    # skipped files
    if len(skipped_files) > 0:
        print("Skipped Files:")
        print(skipped_files)
        print("number of skipped files: ", len(skipped_files))


traj_list = []
lock = threading.Lock()

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/2.1.0")
parser.add_argument("--smooth_nav", dest="smooth_nav", action="store_true")
parser.add_argument("--time_delays", dest="time_delays", action="store_true")
parser.add_argument("--shuffle", dest="shuffle", action="store_true")
parser.add_argument("--num_threads", type=int, default=1)
parser.add_argument("--only_reward_relabel", action="store_true")
parser.add_argument(
    "--reward_config", type=str, default="../models/config/rewards.json"
)
args = parser.parse_args()

# make a list of all the traj_data json files
for dir_name, subdir_list, file_list in walklevel(args.data_path, level=3):
    if "trial_" in dir_name:
        json_file = os.path.join(dir_name, TRAJ_DATA_JSON_FILENAME)
        if not os.path.isfile(json_file) or "tests" in dir_name:
            continue
        traj_list.append(json_file)

# random shuffle
if args.shuffle:
    random.shuffle(traj_list)

# start threads
threads = []
for n in range(args.num_threads):
    thread = threading.Thread(target=run)
    threads.append(thread)
    thread.start()
    time.sleep(1)
