import numpy as np
from boss.utils.utils import compute_distance
from boss.alfred.gen.utils.game_util import (
    get_object,
    get_objects_of_type,
    object_type_equals,
)
import json
from boss.alfred.gen.constants import RECEPTACLES
import os

json_paths = os.path.join(os.environ["BOSS"], "boss/rollouts/")

moveable_object = json.loads(
    open(os.path.join(json_paths, "moveable_object.json")).read()
)
non_moveable_object = json.loads(
    open(os.path.join(json_paths, "non_moveable_object.json")).read()
)
RECEPTACLES = list(RECEPTACLES)
for i in range(len(RECEPTACLES)):
    RECEPTACLES[i] = RECEPTACLES[i].lower()


def get_receptacle_object_from_skill_info(skill_info, state_metadata):
    return get_objects_of_type(
        skill_info["planner_action"]["coordinateReceptacleObjectId"][0], state_metadata
    )[0]


def remove_slice_postfix(inventory_id):
    # sometimes when picking up to inventory the item gets a new dumb name etension (eg apple | | | | applesliced)
    # so we need to remove the postfix
    split_id = inventory_id.split("|")
    if len(split_id) == 5:
        return "|".join(split_id[:-1])
    return inventory_id


def object_id_to_type(object_id):
    split_id = object_id.split("|")
    if len(split_id) == 4:
        return split_id[0]
    elif len(split_id) == 5:
        # special case for Sink|BS|BS|BS|SinkBasin
        # or Knife|BS|BS|BS|ButterKnife
        return split_id[-1]


def get_closest_obj_of_type(obj_type, metadata):
    objects_of_right_type = get_objects_of_type(obj_type, metadata)
    agent_position = metadata["agent"]["position"]
    distances = [
        (obj, compute_distance(agent_position, obj)) for obj in objects_of_right_type
    ]
    return min(distances, key=lambda x: x[1])[0]


def compute_distance_2D(agent_position, object):

    agent_location = np.array([agent_position["x"], agent_position["z"]])
    object_location = np.array([object["position"]["x"], object["position"]["z"]])

    distance = np.linalg.norm(agent_location - object_location)

    return distance


def convertTuple(tup):
    # initialize an empty string
    out_str = ""
    for item in tup:
        out_str = out_str + "|" + str(item)
    return out_str


class BaseAction(object):
    """
    base class for API actions
    """

    def check_valid_skill(self, state, **kwargs):
        return True

    def __init__(self, gt_graph, env, skill_dict):
        # self.gt_graph = gt_graph  # for navigation
        self.env = env
        self.skill_dict = skill_dict
        self.gt_graph = gt_graph
        self.close_object_required = (
            "api_actions" in skill_dict and "CloseObject" in skill_dict["api_actions"]
        )

        self.args = None
        if skill_dict["discrete_action"] == "GotoLocation":
            self.args = skill_dict["discrete_action"]["args"][0]
        # self.rewards = rewards
        # self.strict = strict

    def get_reward(self, state, **kwargs):
        reward, done = self.rewards["neutral"], True
        return reward, done


class PickupObjectAction(BaseAction):
    """
    PickupObject
    """

    def check_valid_skill(self, state):

        self.valid_check = self.check_object(state)
        return True

    def check_object(self, state):
        self.close_object_required = False
        if "CloseObject" in self.skill_dict["api_actions"]:
            self.close_object_required = True

        self.target_object_type = self.skill_dict["planner_action"][
            "coordinateObjectId"
        ][0]
        parent_receptacle_type = None
        # it says "Id" but it's actually a type lol
        if "coordinateReceptacleObjectId" in self.skill_dict["planner_action"]:
            parent_receptacle_type = self.skill_dict["planner_action"][
                "coordinateReceptacleObjectId"
            ][0]

        self.object_parent_map = {}
        objects_of_right_type = get_objects_of_type(
            self.target_object_type, state.metadata
        )

        if len(objects_of_right_type) == 0:
            return False

        for object in objects_of_right_type:
            # for all objects that are of the right type, check if they are pickupable
            if object["pickupable"] and not object["isPickedUp"]:
                parent_receptacles = object["parentReceptacles"]
                if parent_receptacles is None:
                    parent_receptacles = (
                        []
                    )  # for compatibility with below list comprehensions
                # if closing an object is required in this skill, then check if the object is in a receptacle and that receptacle is closeable/openable
                object_valid = False
                if (
                    self.close_object_required
                    and object["objectType"].lower() != "laptop"
                ):
                    valid_parents = [
                        parent_receptacle
                        for parent_receptacle in parent_receptacles
                        if get_object(parent_receptacle, state.metadata)["openable"]
                        and object_type_equals(
                            object_id_to_type(parent_receptacle), parent_receptacle_type
                        )
                    ]
                    if len(valid_parents) > 0:
                        object_valid = True
                # otherwise, just check the parent receptacles
                else:
                    # NEW: ignore the receptacle check, because this causes mismatch between train and eval
                    object_valid = True
                    valid_parents = []
                    # valid_parents = [
                    #    parent_receptacle
                    #    for parent_receptacle in parent_receptacles
                    #    if object_type_equals(
                    #        object_id_to_type(parent_receptacle), parent_receptacle_type
                    #    )
                    # ]
                    ## if parent receptacle exists and valid parents exist, then the object is valid
                    # if parent_receptacle_type is not None and len(valid_parents) > 0:
                    #    object_valid = True
                    ## if parent receptacle is none and also this object has no parent receptacle, then the object is valid
                    # elif (
                    #    parent_receptacle_type is None and len(parent_receptacles) == 0
                    # ):
                    #    object_valid = True
                # if the parent receptacles are the same, then the object is valid for this skill
                # final check is that the object ID's match
                if (
                    object_valid
                    or object["objectId"]
                    == self.skill_dict["planner_action"]["objectId"]
                ):
                    self.object_parent_map[object["objectId"]] = valid_parents

        return len(self.object_parent_map) > 0

    def get_reward(self, state, **kwargs):
        reward, done = 0, False

        if not self.valid_check:
            return reward, done
        else:
            inventory_objects = state.metadata["inventoryObjects"]

            # check inventory for the correct object
            for inventory_object in inventory_objects:
                # if the inventory object is in the pre-defined object parent map, check more conditions to see if it's correct
                processed_id = remove_slice_postfix(inventory_object["objectId"])
                if processed_id in self.object_parent_map:
                    # if we need to close an object, then we need to check that each parent recep is closed
                    if self.close_object_required:
                        reward = 1
                        # sometimes parent receptacles is none when setting up for some reason, so re-check it now

                        #
                        """
                            here the error is
                            if len(self.object_parent_map[inventory_object["objectId"]]) == 0:
                            KeyError: 'Lettuce|+01.03|+01.21|+01.16|LettuceSliced_1'
                            Error: KeyError('Lettuce|+01.03|+01.21|+01.16|LettuceSliced_1')
                        """

                        if inventory_object["objectType"].lower() == "laptop":
                            parent_status = (
                                "parentReceptacles" not in inventory_object.keys()
                                or inventory_object["parentReceptacles"]
                                or len(inventory_object["parentReceptacles"]) == 0
                            )
                            if (
                                "isOpen" in inventory_object
                                and inventory_object["isOpen"]
                                and parent_status
                            ):
                                reward = 1
                                done = True
                            elif "isOpen" not in inventory_object and parent_status:
                                reward = 1
                                done = True

                        elif len(self.object_parent_map[processed_id]) == 0:
                            if "parentReceptacles" in inventory_object.keys():
                                parent_receptacles = inventory_object[
                                    "parentReceptacles"
                                ]
                                if (
                                    parent_receptacles is None
                                    or len(parent_receptacles) == 0
                                ):
                                    reward = 0
                                else:
                                    self.object_parent_map[processed_id] = (
                                        parent_receptacles
                                    )
                            else:
                                reward = 0
                        # check if each parent receptacle is closed
                        if inventory_object["objectType"].lower() != "laptop":
                            for parent_recep_id in self.object_parent_map[processed_id]:
                                parent_object = get_object(
                                    parent_recep_id, state.metadata
                                )
                                if not parent_object["isOpen"]:
                                    reward *= 1
                                else:
                                    reward *= 0
                            done = reward == 1
                        if done:
                            break
                    else:
                        reward = 1
                        done = True
                        break

        return reward, done


class PutObjectAction(BaseAction):
    """
    PutObject
    """

    def check_valid_skill(self, state):
        self.valid_check = self.check_object(state)
        return True

    def check_object(self, state):
        self.close_object_required = False
        if "CloseObject" in self.skill_dict["api_actions"]:
            self.close_object_required = True

        self.object_type = self.skill_dict["planner_action"]["coordinateObjectId"][0]
        self.receptacle_arg = self.skill_dict["planner_action"][
            "coordinateReceptacleObjectId"
        ][0]
        scene_objects = state.metadata["objects"]
        inventory_objects = state.metadata["inventoryObjects"]

        self.valid_objects = set()

        self.picked_up_objects = set()

        # for inv_obj in inventory_objects:
        for inv_obj in scene_objects:
            # obj = get_object(remove_slice_postfix(inv_obj["objectId"]), state.metadata)
            obj = get_object(inv_obj["objectId"], state.metadata)
            if (
                object_type_equals(obj["objectType"], self.object_type)
                # and obj["isPickedUp"]
            ):
                self.valid_objects.add(obj["objectId"])
            elif object_type_equals(obj["objectId"].split("|")[0], self.object_type):
                self.valid_objects.add(obj["objectId"])

        for inv_obj in inventory_objects:
            self.picked_up_objects.add(inv_obj["objectId"])

        # return early to avoid another loop
        # if len(self.valid_objects) == 0:
        #    return False

        # holds valid receptacles and the number of objects of the correct type they contain
        self.valid_receptacles = set()

        # make sure target object is in scene
        for object in scene_objects:
            if object_type_equals(object["objectType"], self.receptacle_arg):
                valid = False
                # if this object is openable and we need to close an object then it's valid
                if self.close_object_required:
                    if object["openable"]:
                        valid = True
                else:
                    # otherwise it's valid as long as the type matches
                    valid = True
                if valid:
                    self.valid_receptacles.add(object["objectId"])
        return len(self.valid_receptacles) > 0

    def get_reward(
        self,
        state,
        **kwargs,
    ):
        reward, done = 0, False
        if not self.valid_check:
            return reward, done
        else:
            # check inventory for the correct object
            inventory_objects = state.metadata["inventoryObjects"]
            for inventory_object in inventory_objects:
                self.picked_up_objects.add(inventory_object["objectId"])
            # check against pre-defined valid objects
            for valid_object_id in self.valid_objects:
                valid_object = get_object(valid_object_id, state.metadata)

                # check to make sure we have picked it up already. This prevents the case
                # in which an object is already in the thing we are trying to put it in
                # and we get a reward for that without doing anything.
                if valid_object["objectId"] not in self.picked_up_objects:
                    continue
                parent_receptacle_ids = valid_object["parentReceptacles"]

                if parent_receptacle_ids is None:
                    parent_receptacle_ids = []
                for parent_recep_id in parent_receptacle_ids:
                    # if in pre-defined valid object receptacles, then posibly get reward
                    if parent_recep_id in self.valid_receptacles:
                        # if we don't need to close the object then the object has been placed successfully
                        if not self.close_object_required:
                            return 1, True
                        else:
                            # if we need to close the object, then check if the parent receptacle is closed
                            parent_object = get_object(parent_recep_id, state.metadata)
                            if (
                                parent_object["openable"]
                                and not parent_object["isOpen"]
                            ):
                                return 1, True
        return reward, done


class ToggleObjectAction(BaseAction):
    """
    ToggleObjectOn, ToggleObjectOff
    """

    valid_actions = {
        "ToggleObjectOn",
        "ToggleObjectOff",
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Teleport",
        "TeleportFull",
    }

    def check_valid_skill(self, state):
        self.valid_check = self.check_object(state)
        return True

    def check_object(self, state):

        self.target_object_type = self.skill_dict["planner_action"][
            "coordinateObjectId"
        ][0]

        objs = get_objects_of_type(self.target_object_type, state.metadata)
        self.valid_target_object_ids = set()
        for obj in objs:
            if obj["toggleable"]:
                if "ToggleObjectOn" in self.skill_dict["api_actions"]:
                    if not obj["isToggled"]:
                        self.valid_target_object_ids.add(obj["objectId"])
                elif "ToggleObjectOff" in self.skill_dict["api_actions"]:
                    if obj["isToggled"]:
                        self.valid_target_object_ids.add(obj["objectId"])
        return len(self.valid_target_object_ids) > 0

    def get_reward(self, state, **kwargs):
        reward, done = 0, False
        if not self.valid_check:
            return reward, done
        else:

            for valid_object_id in self.valid_target_object_ids:
                target_object = get_object(valid_object_id, state.metadata)
                if target_object is not None:
                    is_target_toggled = target_object["isToggled"]
                    if "ToggleObjectOn" in self.skill_dict["api_actions"]:
                        if is_target_toggled:
                            return 1, True
                    elif "ToggleObjectOff" in self.skill_dict["api_actions"]:
                        if not is_target_toggled:
                            return 1, True
            return reward, done


class SliceObjectAction(BaseAction):
    """
    SliceObject
    """

    valid_actions = {
        "SliceObject",
        "OpenObject",
        "CloseObject",
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Teleport",
        "TeleportFull",
    }

    def check_valid_skill(self, state):
        self.valid_check = self.check_object(state)
        return True

    def check_object(self, state):
        # TODO: if the closeObject action is in the set of api actions,
        # then we need to check if object in receptacle and that receptacle is openable.
        # then, we memorize the receptacle and the object to be sliced.

        # SLICE is special because not everything has receptacles

        # TODO: In general, memorize object ID and their associated receptacles
        # to make sure we're not picking up random objects not on that receptacle.

        self.close_object_required = False
        if "CloseObject" in self.skill_dict["api_actions"]:
            self.close_object_required = True
        self.target_object_type = self.skill_dict["planner_action"][
            "coordinateObjectId"
        ][0]

        parent_receptacle_type = None
        # it says "Id" but it's actually a type lol
        if "coordinateReceptacleObjectId" in self.skill_dict["planner_action"]:
            parent_receptacle_type = self.skill_dict["planner_action"][
                "coordinateReceptacleObjectId"
            ][0]

        self.receptacle_object = None

        self.object_parent_map = {}
        objects_of_right_type = get_objects_of_type(
            self.target_object_type, state.metadata
        )

        if len(objects_of_right_type) == 0:
            return False

        for object in objects_of_right_type:
            # for all objects that are of the right type, check if they are sliceable and they haven't already been sliced
            if object["sliceable"] and not object["isSliced"]:
                parent_receptacles = object["parentReceptacles"]
                if parent_receptacle_type is None:
                    parent_receptacles = (
                        []
                    )  # for compatibility with below list comprehensions
                # if closing an object is required in this skill, then check if the object is in a receptacle and that receptacle is closeable/openable
                object_valid = False
                if self.close_object_required:
                    valid_parents = [
                        parent_receptacle
                        for parent_receptacle in parent_receptacles
                        if parent_receptacle["openable"]
                        and object_type_equals(
                            object_id_to_type(parent_receptacle), parent_receptacle_type
                        )
                    ]
                    if len(valid_parents) > 0:
                        object_valid = True
                # otherwise, just check the parent receptacles
                else:
                    valid_parents = [
                        parent_receptacle
                        for parent_receptacle in parent_receptacles
                        if object_type_equals(
                            object_id_to_type(parent_receptacle), parent_receptacle_type
                        )
                    ]
                    # if parent receptacle exists and valid parents exist, then the object is valid
                    if parent_receptacle_type is not None and len(valid_parents) > 0:
                        object_valid = True
                    # if parent receptacle is none and also this object has no parent receptacle, then the object is valid
                    elif (
                        parent_receptacle_type is None and len(parent_receptacles) == 0
                    ):
                        object_valid = True
                # if the parent receptacles are the same, then the object is valid for this skill
                # final check is that the object ID's match
                if (
                    object_valid
                    or object["objectId"]
                    == self.skill_dict["planner_action"]["objectId"]
                ):
                    self.object_parent_map[object["objectId"]] = valid_parents
        return len(self.object_parent_map) > 0

    def get_reward(self, state, **kwargs):
        reward, done = 0, False
        if not self.valid_check:
            return reward, done
        else:
            # check object_parent_map for a sliced object
            for object_id, parent_receptacles in self.object_parent_map.items():
                obj = get_object(object_id, state.metadata)
                if obj["isSliced"]:
                    # if we need to close an object, then we need to check that each parent recep is closed
                    if self.close_object_required:
                        reward = 1
                        # sometimes parent receptacles is none when setting up for some reason, so re-check it now
                        if len(parent_receptacles) == 0:
                            parent_receptacles = obj["parentReceptacles"]
                            if (
                                parent_receptacles is None
                                or len(parent_receptacles) == 0
                            ):
                                reward = 0
                        # check if each parent receptacle is closed
                        for parent_recep_id in parent_receptacles:
                            parent_object = get_object(parent_recep_id, state.metadata)
                            if not parent_object["isOpen"]:
                                reward *= 1
                            else:
                                reward *= 0
                        done = reward == 1
                        if done:
                            break
                    else:
                        reward = 1
                        done = True
            return reward, done


class CleanObjectAction(BaseAction):
    """
    CleanObject
    """

    valid_actions = {
        "PutObject",
        "PickupObject",
        "ToggleObjectOn",
        "ToggleObjectOff",
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Teleport",
        "TeleportFull",
    }

    def check_valid_skill(self, state):
        self.valid_check = self.check_object(state)
        return True

    def check_object(self, state):
        target_object_type = self.skill_dict["planner_action"]["coordinateObjectId"][0]

        objs_of_right_type = get_objects_of_type(target_object_type, state.metadata)
        if len(objs_of_right_type) == 0:
            return False

        self.valid_objects = set()
        for obj in objs_of_right_type:
            if obj["dirtyable"] and obj["isDirty"]:
                self.valid_objects.add(obj["objectId"])
            elif (
                obj["objectType"].lower() == "egg"
                or obj["objectType"].lower() == "soapbar"
                or obj["objectType"].lower() == "lettuce"
                or obj["objectType"].lower() == "knife"
                or obj["objectType"].lower() == "spatula"
                or obj["objectType"].lower() == "apple"
                or obj["objectType"].lower() == "ladle"
                or obj["objectType"].lower() == "butterknife"
                or obj["objectType"].lower() == "tomato"
                or obj["objectType"].lower() == "fork"
                or obj["objectType"].lower() == "dishsponge"
                or obj["objectType"].lower() == "kettle"
                or obj["objectType"].lower() == "cup"
            ):
                # clean egg, egg is not dirtyable and isDirty is always false
                # clean soapbar, soapbar is not dirtyable and isDirty is always false
                self.valid_objects.add(obj["objectId"])
        # if len(self.valid_objects) > 0:
        #    # ADDED: check if the object is in the inventory
        #    inventory_objects = state.metadata["inventoryObjects"]
        #    for inventory_object in inventory_objects:
        #        if inventory_object["objectId"] in self.valid_objects:
        #            return True
        # return False
        return len(self.valid_objects) > 0

    def get_reward(
        self,
        state,
        **kwargs,
    ):
        # check object is clean
        # check faucet is closed
        # check hold clean object

        reward, done = 0, False
        if not self.valid_check:
            return reward, done
        else:
            cleaned_valid_objects = set()
            for object_id in self.env.cleaned_objects:
                # obj = get_object(object_id, state.metadata)
                if object_id in self.valid_objects:
                    cleaned_valid_objects.add(object_id)

            if len(cleaned_valid_objects) == 0:
                return 0, False

            # check that closest faucet is off
            closest_faucet = get_closest_obj_of_type("Faucet", state.metadata)
            if closest_faucet is not None:
                if closest_faucet["isToggled"]:
                    return 0, False

            # check that we're holding the clean object
            inventory_objects = state.metadata["inventoryObjects"]
            if len(inventory_objects) > 0:
                for inventory_object in inventory_objects:
                    if (
                        remove_slice_postfix(inventory_object["objectId"])
                        in cleaned_valid_objects
                    ):
                        return 1, True
                    # else:
                    #     import pdb; pdb.set_trace()
            return reward, done


class HeatObjectAction(BaseAction):
    """
    HeatObject
    """

    # add microwave oven is closed after heating
    # add object is holding after heating
    # only at end state

    valid_actions = {
        "OpenObject",
        "CloseObject",
        "PickupObject",
        "PutObject",
        "ToggleObjectOn",
        "ToggleObjectOff",
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Teleport",
        "TeleportFull",
    }

    def check_valid_skill(self, state):
        self.valid_check = self.check_object(state)
        return True

    def check_object(self, state):
        self.target_object_type = self.skill_dict["discrete_action"]["args"][
            0
        ].capitalize()
        scene_objects = state.metadata["objects"]

        self.valid_heatable_objects = set()
        for object in scene_objects:
            if (
                self.target_object_type in object["objectType"]
            ):  # for cases like Apple and AppleSliced
                self.valid_heatable_objects.add(object["objectId"])
        # if len(self.valid_heatable_objects) > 0:
        #    # ADDED: check if the object is in the inventory
        #    inventory_objects = state.metadata["inventoryObjects"]
        #    for inventory_object in inventory_objects:
        #        if (
        #            remove_slice_postfix(inventory_object["objectId"])
        #            in self.valid_heatable_objects
        #        ):
        #            return True
        # return False
        return len(self.valid_heatable_objects) > 0

    def get_reward(
        self,
        state,
        **kwargs,
    ):
        reward, done = 0, False
        if not self.valid_check:
            return reward, done
        else:
            valid_heated_objects = self.env.heated_objects.intersection(
                self.valid_heatable_objects
            )
            if len(valid_heated_objects) == 0:
                return 0, False

            # check microwave is closed
            closest_microwave = get_closest_obj_of_type("Microwave", state.metadata)
            is_microwave_open = closest_microwave["isOpen"]

            # check holding object
            inventory_objects = state.metadata["inventoryObjects"]
            holding_heated_object = False
            if len(inventory_objects) > 0:
                for inv_object in inventory_objects:
                    inv_object_id = inv_object[
                        "objectId"
                    ]  # no remove_slice_postfix here because inventory and env heated objects somehow matches?????
                    if inv_object_id in valid_heated_objects:
                        holding_heated_object = True
                        break
                    else:
                        holding_heated_object = False

            if holding_heated_object and not is_microwave_open:
                reward = 1
                done = True
            return reward, done


class CoolObjectAction(BaseAction):
    """
    CoolObject
    """

    # add fridge is closed after cooling
    # add object is holding after cooling
    # only at end state

    valid_actions = {
        "OpenObject",
        "CloseObject",
        "PickupObject",
        "PutObject",
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Teleport",
        "TeleportFull",
    }

    def check_valid_skill(self, state):
        self.valid_check = self.check_object(state)
        return True

    def check_object(self, state):
        self.target_object_type = self.skill_dict["discrete_action"]["args"][0]

        """
        Here error
        Traceback (most recent call last):                                                                                                                          
        File "/home/jzhang96/ALFRED_jiahui/gen/scripts/check_online_trajectory_reward_correspondence.py", line 365, in run                                        
            augment_traj(env, json_file)                                                                                                                            
        File "/home/jzhang96/ALFRED_jiahui/gen/scripts/check_online_trajectory_reward_correspondence.py", line 186, in augment_traj                               
            raise Exception("Unmatched HL Action")                                                                                                                  
        Exception: Unmatched HL Action
        Error: Exception('Unmatched HL Action')
        Skipping /home/jzhang96/ALFRED_jiahui/data/json_2.1.0/valid_unseen/pick_cool_then_place_in_recep-BreadSliced-None-Microwave-10/trial_T20190906_224843_443882
        /traj_data.json
        """
        scene_objects = state.metadata["objects"]
        # problem cool object only show fridge as target object. But we cannot distinguish bread and bread sliced.
        if self.target_object_type == "bread":
            for object in scene_objects:
                if "BreadSliced" in object["objectType"]:
                    self.target_object_type = "BreadSliced"
                    break

        elif self.target_object_type == "tomato":
            for object in scene_objects:
                if "TomatoSliced" in object["objectType"]:
                    self.target_object_type = "TomatoSliced"
                    break

        elif self.target_object_type == "apple":
            for object in scene_objects:
                if "AppleSliced" in object["objectType"]:
                    self.target_object_type = "AppleSliced"
                    break

        elif self.target_object_type == "lettuce":
            for object in scene_objects:
                if "LettuceSliced" in object["objectType"]:
                    self.target_object_type = "LettuceSliced"
                    break

        elif self.target_object_type == "potato":
            for object in scene_objects:
                if "PotatoSliced" in object["objectType"]:
                    self.target_object_type = "PotatoSliced"
                    break

        object_in_scene = False

        inventory_objects = state.metadata["inventoryObjects"]

        for object in scene_objects:
            # ADDED: check if the object is in the inventory
            if object_type_equals(object["objectType"], self.target_object_type):
                # for inventory_object in inventory_objects:
                #    if remove_slice_postfix(
                #        inventory_object["objectId"]
                #    ) == remove_slice_postfix(object["objectId"]):
                #        object_in_scene = True
                #        break
                object_in_scene = True
                break
        return object_in_scene

    def get_reward(
        self,
        state,
        **kwargs,
    ):
        # check inventory object is cooled and we're holding it
        reward, done = 0, False
        if not self.valid_check:
            return reward, done
        else:
            inventory_objects = state.metadata["inventoryObjects"]
            holding_cool_object = False
            if len(inventory_objects) > 0:
                for inventory_object in inventory_objects:
                    # cooled + holding
                    if (
                        object_type_equals(
                            inventory_object["objectType"], self.target_object_type
                        )
                        and inventory_object["objectId"] in self.env.cooled_objects
                    ):
                        holding_cool_object = True
                        break

            if not holding_cool_object:
                return 0, False

            # check fridge is closed
            fridge_object = get_closest_obj_of_type("Fridge", state.metadata)
            is_fridge_closed = not fridge_object["isOpen"]
            if is_fridge_closed:
                reward = 1
                done = True

            return reward, done


def get_action(graph, env, skill_dict):
    action_type = skill_dict["discrete_action"]["action"]
    action_type_str = action_type + "Action"

    if action_type_str in globals():
        action = globals()[action_type_str]
        return action(graph, env, skill_dict)
    else:
        raise Exception("Invalid action_type %s" % action_type_str)
