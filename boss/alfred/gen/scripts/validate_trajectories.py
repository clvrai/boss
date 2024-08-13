import os
import sys

if "ALFRED_ROOT" not in os.environ:
    os.environ["ALFRED_ROOT"] = "/home/jesse/ALFRED_jiahui"
import os
import sys

sys.path.append(os.path.join(os.environ["ALFRED_ROOT"]))
sys.path.append(os.path.join(os.environ["ALFRED_ROOT"], "gen"))

import json
import os
import shutil
import argparse
import threading
import time
import copy
import random
from boss.alfred.gen.utils.video_util import VideoSaver
from boss.alfred.gen.utils.py_util import walklevel
from boss.alfred.env.thor_env import ThorEnv


TRAJ_DATA_JSON_FILENAME = "traj_data.json"
AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"

ORIGINAL_IMAGES_FORLDER = "raw_images"
HIGH_RES_IMAGES_FOLDER = "high_res_images"
DEPTH_IMAGES_FOLDER = "depth_images"
INSTANCE_MASKS_FOLDER = "instance_masks"

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

render_settings = dict()
render_settings["renderImage"] = False
render_settings["renderDepthImage"] = False
render_settings["renderObjectImage"] = False
render_settings["renderClassImage"] = False

video_saver = VideoSaver()


def augment_traj(env, json_file):
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)

    # fresh images list
    traj_data["images"] = list()

    # scene setup
    scene_num = traj_data["scene"]["scene_num"]
    object_poses = traj_data["scene"]["object_poses"]
    object_toggles = traj_data["scene"]["object_toggles"]
    dirty_and_empty = traj_data["scene"]["dirty_and_empty"]

    # reset
    scene_name = "FloorPlan%d" % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)
    env.step(dict(traj_data["scene"]["init_action"]))

    # setup task
    env.set_task(traj_data, args, reward_type="dense")
    rewards = []
    prev_hl_action = traj_data["plan"]["high_pddl"][0]
    for ll_idx, ll_action in enumerate(traj_data["plan"]["low_actions"]):
        # next cmd under the current hl_action
        cmd = ll_action["api_action"]
        hl_action = traj_data["plan"]["high_pddl"][ll_action["high_idx"]]
        if prev_hl_action != hl_action:
            if rewards[-1] < 2:
                ll_actions = [
                    action["api_action"]["action"]
                    for action in traj_data["plan"]["low_actions"][:ll_idx]
                    if action["high_idx"] == prev_hl_action["high_idx"]
                ]
                print(
                    f"Previous HL Action: {prev_hl_action['discrete_action']}, Rewards: {rewards}, Language Instruction: {traj_data['turk_annotations']['anns'][0]['high_descs'][prev_hl_action['high_idx']]}, Actions: {ll_actions}"
                )
                # raise Exception("New HL Action")
            prev_hl_action = hl_action
            rewards = []

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
                events = env.smooth_move_ahead(cmd, render_settings)
                event = events[-1]
            else:
                event = env.step(cmd)

        elif "Rotate" in cmd["action"]:
            if args.smooth_nav:
                events = env.smooth_rotate(cmd, render_settings)
                event = events[-1]
            else:
                event = env.step(cmd)

        elif "Look" in cmd["action"]:
            if args.smooth_nav:
                events = env.smooth_look(cmd, render_settings)
                event = events[-1]
            else:
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
                event = env.step(cmd)
            else:
                event = env.step(cmd)

        else:
            if args.time_delays:
                event = env.step(cmd)
            else:
                event = env.step(cmd)
        # update last event with the original intended action
        # if args.smooth_nav:
        #    env.last_event.metadata["lastAction"] = cmd["action"]

        # update image list
        if not event.metadata["lastActionSuccess"]:
            raise Exception(
                "Replay Failed: %s" % (env.last_event.metadata["errorMessage"])
            )

        reward, _ = env.get_transition_reward()
        rewards.append(reward)

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


def run():
    """
    replay loop
    """
    # start THOR env
    env = ThorEnv(player_screen_width=IMAGE_WIDTH, player_screen_height=IMAGE_HEIGHT)

    skipped_files = []

    while len(traj_list) > 0:
        lock.acquire()
        json_file = traj_list.pop()
        lock.release()

        print("Augmenting: " + json_file)
        try:
            # if (
            #    json_file
            #    # == "/home/jesse/ALFRED_jiahui/data/json_2.1.0/train/pick_two_obj_and_place-Pen-None-Drawer-307/trial_T20190909_055453_986512/traj_data.json"
            #    # == "/home/jesse/ALFRED_jiahui/data/json_2.1.0/train/pick_clean_then_place_in_recep-Spatula-None-CounterTop-13/trial_T20190909_023039_986825/traj_data.json"
            #    # == "/home/jesse/ALFRED_jiahui/data/json_2.1.0/train/pick_two_obj_and_place-Pen-None-Drawer-307/trial_T20190909_055453_986512/traj_data.json"
            #    # == "/home/jesse/ALFRED_jiahui/data/json_2.1.0/train/pick_two_obj_and_place-Laptop-None-Bed-311/trial_T20190907_074201_784114/traj_data.json"
            #    == " /home/jesse/ALFRED_jiahui/data/json_2.1.0/train/pick_two_obj_and_place-BreadSliced-None-Fridge-11/trial_T20190908_132037_877612/traj_data.json"
            # ):
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


traj_list = []
lock = threading.Lock()

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/2.1.0")
parser.add_argument("--smooth_nav", dest="smooth_nav", action="store_true")
parser.add_argument("--time_delays", dest="time_delays", action="store_true")
parser.add_argument("--shuffle", dest="shuffle", action="store_true")
parser.add_argument("--num_threads", type=int, default=1)
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
