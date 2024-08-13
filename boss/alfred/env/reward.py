from boss.alfred.gen.utils.game_util import get_object, get_objects_of_type

# modified from the original ALFRED code to support modified rewards


# doesn't matter which slice you pick up
def remove_slice_postfix(object_id):
    return object_id.split("Sliced")[0]


class BaseAction(object):
    """
    base class for API actions
    """

    def __init__(self, gt_graph, env, rewards, strict=True):
        self.gt_graph = gt_graph  # for navigation
        self.env = env
        self.rewards = rewards
        self.strict = strict

    def get_reward(self, state, prev_state, expert_plan, goal_idx, expert_ll_actions):
        reward, done = self.rewards["neutral"], True
        return reward, done


class GotoLocationAction(BaseAction):
    """
    MoveAhead, Rotate, Lookup
    """

    valid_actions = {
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Teleport",
        "TeleportFull",
    }

    def get_reward(self, state, prev_state, expert_plan, goal_idx, expert_ll_actions):
        if state.metadata["lastAction"] not in self.valid_actions:
            # print(state.metadata["lastAction"])
            reward, done = self.rewards["invalid_action"], False
            return reward, done

        subgoal = expert_plan[goal_idx]["planner_action"]
        curr_pose = state.pose_discrete
        prev_pose = prev_state.pose_discrete
        tar_pose = tuple([int(i) for i in subgoal["location"].split("|")[1:]])

        prev_actions, _ = self.gt_graph.get_shortest_path(prev_pose, tar_pose)
        curr_actions, _ = self.gt_graph.get_shortest_path(curr_pose, tar_pose)

        prev_distance = len(prev_actions)
        curr_distance = len(curr_actions)
        reward = (prev_distance - curr_distance) * 0.2  # distance reward factor?

        # [DEPRECATED] Old criteria which requires the next subgoal object to be visible
        # Consider navigation a success if we can see the target object in the next step from here.
        # assert len(expert_plan) > goal_idx + 1
        # next_subgoal = expert_plan[goal_idx + 1]['planner_action']
        # next_goal_object = get_object(next_subgoal['objectId'], state.metadata)
        # done = (next_goal_object['visible'] and curr_distance < self.rewards['min_reach_distance'])

        done = curr_distance < self.rewards["min_reach_distance"]

        if done:
            reward += self.rewards["positive"]

        return reward, done


class PickupObjectAction(BaseAction):
    """
    PickupObject
    """

    valid_actions = {
        "MoveAhead",
        "PickupObject",
        "OpenObject",
        "CloseObject",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Teleport",
        "TeleportFull",
    }

    def get_reward(self, state, prev_state, expert_plan, goal_idx, expert_ll_actions):
        if state.metadata["lastAction"] not in self.valid_actions:
            reward, done = self.rewards["invalid_action"], False
            return reward, done

        subgoal = expert_plan[goal_idx]["planner_action"]
        reward, done = self.rewards["neutral"], False
        inventory_objects = state.metadata["inventoryObjects"]

        current_relevant_actions = [
            action for action in expert_ll_actions if action["high_idx"] == goal_idx
        ]
        api_actions = set(
            [action["api_action"]["action"] for action in current_relevant_actions]
        )
        close_in_goal = "CloseObject" in api_actions
        if len(inventory_objects) and not self.env.picked_up:
            inv_object_id = state.metadata["inventoryObjects"][0]["objectId"]
            goal_object_id = subgoal["objectId"]

            inv_object_id = remove_slice_postfix(inv_object_id)
            goal_object_id = remove_slice_postfix(goal_object_id)

            reward, done = (
                (self.rewards["positive"], not close_in_goal)
                if inv_object_id == goal_object_id
                else (self.rewards["negative"], False)
            )
            if inv_object_id == goal_object_id:
                self.env.picked_up = True
        elif len(inventory_objects) and self.env.picked_up:
            inv_object_id = state.metadata["inventoryObjects"][0]["objectId"]
            goal_object_id = subgoal["objectId"]

            inv_object_id = remove_slice_postfix(inv_object_id)
            goal_object_id = remove_slice_postfix(goal_object_id)
            last_action_is_closed = state.metadata["lastAction"] == "CloseObject"
            object_closed = not get_object(
                current_relevant_actions[-1]["api_action"]["objectId"], state.metadata
            )["isOpen"]

            reward, done = (
                (self.rewards["positive"], True)
                if inv_object_id == goal_object_id
                and last_action_is_closed
                and object_closed
                else (self.rewards["negative"], False)
            )
        if done:
            self.env.picked_up = False
        return reward, done


class PutObjectAction(BaseAction):
    """
    PutObject
    """

    valid_actions = {
        "PutObject",
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

    def get_reward(self, state, prev_state, expert_plan, goal_idx, expert_ll_actions):
        if state.metadata["lastAction"] not in self.valid_actions:
            reward, done = self.rewards["invalid_action"], False
            return reward, done
        subgoal = expert_plan[goal_idx]["planner_action"]
        current_relevant_actions = [
            action for action in expert_ll_actions if action["high_idx"] == goal_idx
        ]
        api_actions = set(
            [action["api_action"]["action"] for action in current_relevant_actions]
        )
        close_in_goal = "CloseObject" in api_actions
        reward, done = self.rewards["neutral"], False
        target_object_id = subgoal["objectId"]
        recep_object = get_object(subgoal["receptacleObjectId"], state.metadata)
        if (
            recep_object is not None
            and not self.env.put_in_recep
            and state.metadata["lastAction"] == "PutObject"
        ):
            is_target_in_recep = target_object_id in recep_object["receptacleObjectIds"]
            reward, done = (
                (self.rewards["positive"], not close_in_goal)
                if is_target_in_recep
                else (self.rewards["negative"], False)
            )
            if is_target_in_recep:
                self.env.put_in_recep = True
            # print(
            #    f"Target in receptacle: {is_target_in_recep}, recep_objects: {recep_object['receptacleObjectIds']}"
            # )
            object = get_object(target_object_id, state.metadata)
            # print(f"Object: {object['name']}, Position: {object['position']}")
        elif recep_object is not None and self.env.put_in_recep:
            is_target_in_recep = target_object_id in recep_object["receptacleObjectIds"]
            # print(
            #    f"Target in receptacle: {is_target_in_recep}, recep_objects: {recep_object['receptacleObjectIds']}"
            # )
            object = get_object(target_object_id, state.metadata)
            # print(f"Object: {object['name']}, Position: {object['position']}")
            last_action_is_close = state.metadata["lastAction"] == "CloseObject"
            object_closed = not get_object(
                current_relevant_actions[-1]["api_action"]["objectId"], state.metadata
            )["isOpen"]
            reward, done = (
                (self.rewards["positive"], True)
                if is_target_in_recep and object_closed and last_action_is_close
                else (self.rewards["negative"], False)
            )
        if done:
            self.env.put_in_recep = False
        return reward, done


class OpenObjectAction(BaseAction):
    """
    OpenObject
    """

    valid_actions = {
        "OpenObject",
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Teleport",
        "TeleportFull",
    }

    def get_reward(self, state, prev_state, expert_plan, goal_idx, expert_ll_actions):
        if state.metadata["lastAction"] not in self.valid_actions:
            reward, done = self.rewards["invalid_action"], False
            return reward, done

        subgoal = expert_plan[goal_idx]["planner_action"]
        reward, done = self.rewards["neutral"], False
        target_recep = get_object(subgoal["objectId"], state.metadata)
        if target_recep is not None:
            is_target_open = target_recep["isOpen"]
            reward, done = (
                (self.rewards["positive"], True)
                if is_target_open
                else (self.rewards["negative"], False)
            )
        return reward, done


class CloseObjectAction(BaseAction):
    """
    CloseObject
    """

    valid_actions = {
        "CloseObject",
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "Teleport",
        "TeleportFull",
    }

    def get_reward(self, state, prev_state, expert_plan, goal_idx, expert_ll_actions):
        if state.metadata["lastAction"] not in self.valid_actions:
            reward, done = self.rewards["invalid_action"], False
            return reward, done

        subgoal = expert_plan[goal_idx]["planner_action"]
        reward, done = self.rewards["negative"], False
        target_recep = get_object(subgoal["objectId"], state.metadata)
        if target_recep is not None:
            is_target_closed = not target_recep["isOpen"]
            reward, done = (
                (self.rewards["positive"], True)
                if is_target_closed
                else (self.rewards["negative"], False)
            )
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

    def get_reward(self, state, prev_state, expert_plan, goal_idx, expert_ll_actions):
        if state.metadata["lastAction"] not in self.valid_actions:
            reward, done = self.rewards["invalid_action"], False
            return reward, done

        subgoal = expert_plan[goal_idx]["planner_action"]
        reward, done = self.rewards["neutral"], False
        target_toggle = get_object(subgoal["objectId"], state.metadata)
        if target_toggle is not None:
            is_target_toggled = target_toggle["isToggled"]
            reward, done = (
                (self.rewards["positive"], True)
                if is_target_toggled
                else (self.rewards["negative"], False)
            )
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

    def get_reward(self, state, prev_state, expert_plan, goal_idx, expert_ll_actions):
        if state.metadata["lastAction"] not in self.valid_actions:
            reward, done = self.rewards["invalid_action"], False
            return reward, done

        subgoal = expert_plan[goal_idx]["planner_action"]
        reward, done = self.rewards["neutral"], False
        target_object = get_object(subgoal["objectId"], state.metadata)
        current_relevant_actions = [
            action for action in expert_ll_actions if action["high_idx"] == goal_idx
        ]
        api_actions = set(
            [action["api_action"]["action"] for action in current_relevant_actions]
        )
        close_in_goal = "CloseObject" in api_actions
        if target_object is not None and not self.env.sliced:
            is_target_sliced = target_object["isSliced"]
            reward, done = (
                (self.rewards["positive"], not close_in_goal)
                if is_target_sliced
                else (self.rewards["negative"], False)
            )
            if is_target_sliced:
                self.env.sliced = True
        elif target_object is not None and self.env.sliced:
            is_target_sliced = target_object["isSliced"]
            last_action_is_close = state.metadata["lastAction"] == "CloseObject"
            object_closed = not get_object(
                current_relevant_actions[-1]["api_action"]["objectId"], state.metadata
            )["isOpen"]
            reward, done = (
                (self.rewards["positive"], True)
                if is_target_sliced and object_closed and last_action_is_close
                else (self.rewards["negative"], False)
            )
        if done:
            self.env.sliced = False
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

    def get_reward(self, state, prev_state, expert_plan, goal_idx, expert_ll_actions):
        if state.metadata["lastAction"] not in self.valid_actions:
            reward, done = self.rewards["invalid_action"], False
            return reward, done

        subgoal = expert_plan[goal_idx]["planner_action"]
        reward, done = self.rewards["neutral"], False
        clean_object = get_object(subgoal["cleanObjectId"], state.metadata)

        # reward for putting the dirty object in the sink
        sink_basin = get_objects_of_type("SinkBasin", state.metadata)
        if (
            len(sink_basin)
            and state.metadata["lastAction"] == "PutObject"
            and not self.env.put_in_sink_reward
        ):
            if clean_object is not None:
                # :4 because the object could be sliced or something while in the sink, so get name and first 3 location tags
                for sink in sink_basin:
                    objects_in_sink_basin = [
                        "|".join(obj.split("|")[:4])
                        for obj in sink["receptacleObjectIds"]
                    ]
                    is_target_in_sink = (
                        clean_object["objectId"] in objects_in_sink_basin
                    )
                    if is_target_in_sink:
                        reward, done = self.rewards["positive"], False
                        self.env.put_in_sink_reward = True
                        break
        # reward for washing the dirty object that's in the sink basin
        elif (
            len(sink_basin)
            and state.metadata["lastAction"] == "ToggleObjectOn"
            and clean_object["objectId"] in self.env.cleaned_objects
            and not self.env.wash_in_sink_reward
        ):
            if clean_object is not None:
                is_target_cleaned = not clean_object["isDirty"]
                if is_target_cleaned:
                    reward, done = self.rewards["positive"], False
                    self.env.wash_in_sink_reward = True
        # reward for turning off the faucet after cleaning the dirty object
        elif (
            len(sink_basin)
            and state.metadata["lastAction"] == "ToggleObjectOff"
            and clean_object["objectId"] in self.env.cleaned_objects
            and not self.env.toggle_faucet_off_reward
        ):
            faucet_object = get_objects_of_type("Faucet", state.metadata)
            if len(faucet_object):
                for faucet in faucet_object:
                    is_faucet_toggled_off = not faucet["isToggled"]
                    if is_faucet_toggled_off:
                        break
            if clean_object is not None and is_faucet_toggled_off:
                reward, done = self.rewards["positive"], False
                self.env.toggle_faucet_off_reward = True
        # reward for picking up the object
        elif (
            clean_object is not None and state.metadata["lastAction"] == "PickupObject"
        ):
            is_obj_clean = clean_object["objectId"] in self.env.cleaned_objects
            reward, done = (
                (self.rewards["positive"], True)
                if is_obj_clean
                else (self.rewards["negative"], False)
            )
        else:
            reward, done = self.rewards["negative"], False
        return reward, done


class HeatObjectAction(BaseAction):
    """
    HeatObject
    """

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

    def get_reward(self, state, prev_state, expert_plan, goal_idx, expert_ll_actions):
        if state.metadata["lastAction"] not in self.valid_actions:
            reward, done = self.rewards["invalid_action"], False
            return reward, done

        reward, done = self.rewards["negative"], False
        # next_put_goal_idx = (
        #    goal_idx + 2
        # )  # (+1) GotoLocation -> (+2) PutObject (get the objectId from the PutObject action)
        next_put_goal_idx = (
            goal_idx + 1
        )  # (+1) PutObject (get the objectId from the PutObject action)
        if next_put_goal_idx < len(expert_plan):
            heat_object_id = expert_plan[next_put_goal_idx]["planner_action"][
                "objectId"
            ]
            heat_object = get_object(heat_object_id, state.metadata)
            is_obj_hot = heat_object["objectId"] in self.env.heated_objects
            inventory_objects = state.metadata["inventoryObjects"]
            if len(inventory_objects):
                inv_object_id = state.metadata["inventoryObjects"][0]["objectId"]

            # check that the MICROWAVE was the one that was opened
            microwaves = get_objects_of_type("Microwave", state.metadata)

            for microwave in microwaves:
                is_target_in_microwave = (
                    heat_object["objectId"] in microwave["receptacleObjectIds"]
                )
                if is_target_in_microwave:
                    break
            is_microwave_on = microwave["isToggled"]
            is_microwave_opened = microwave["isOpen"]
            # reward for putting the object in the microwave
            if (
                state.metadata["lastAction"] == "PutObject"
                and not self.env.put_in_microwave_reward
            ):
                if is_target_in_microwave:
                    reward, done = self.rewards["positive"], False
                    self.env.put_in_microwave_reward = True
            # reward for heating the object
            elif (
                state.metadata["lastAction"] == "ToggleObjectOn"
                and not self.env.heat_object_reward
            ):
                if is_microwave_on and is_target_in_microwave:
                    reward, done = self.rewards["positive"], False
                    self.env.heat_object_reward = True
                else:
                    reward, done = self.rewards["negative"], False
            # reward for turning off the microwave
            elif (
                state.metadata["lastAction"] == "ToggleObjectOff"
                and not self.env.toggle_microwave_off_reward
            ):
                is_microwave_off = not microwave["isToggled"]
                if is_obj_hot and is_microwave_off:
                    reward, done = self.rewards["positive"], False
                    self.env.toggle_microwave_off_reward = True
            # reward for opening the microwave door while not holding the correct object
            # elif (
            #    state.metadata["lastAction"] == "OpenObject"
            #    and is_target_in_microwave
            #    and not self.env.open_microwave_door_with_obj_in_it_reward
            #    and not is_microwave_on
            # ):
            #    reward, done = self.rewards["positive"], False
            #    self.env.open_microwave_door_with_obj_in_it_reward = True
            ## reward for taking the object out of the microwave
            # elif (
            #    state.metadata["lastAction"] == "PickupObject"
            #    and is_obj_hot
            #    and not is_microwave_on
            #    and not self.env.take_out_microwave_reward
            # ):
            #    reward, done = self.rewards["positive"], False
            #    self.env.take_out_microwave_reward = True
            # reward for finally closing the microwave door while holding the hot object
            elif (
                state.metadata["lastAction"] == "CloseObject"
                and is_obj_hot
                and len(inventory_objects)
            ):
                if len(inventory_objects):
                    inv_object_id = state.metadata["inventoryObjects"][0]["objectId"]
                    if inv_object_id == heat_object_id:
                        reward, done = self.rewards["positive"], True
            else:
                reward, done = self.rewards["negative"], False
        return reward, done


class CoolObjectAction(BaseAction):
    """
    CoolObject
    """

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

    def get_reward(self, state, prev_state, expert_plan, goal_idx, expert_ll_actions):
        if state.metadata["lastAction"] not in self.valid_actions:
            reward, done = self.rewards["invalid_action"], False
            return reward, done

        reward, done = self.rewards["negative"], False
        subgoal = expert_plan[goal_idx]["planner_action"]
        # next_put_goal_idx = (
        #    goal_idx + 2
        # )  # (+1) GotoLocation -> (+2) PutObject (get the objectId from the PutObject action)
        next_put_goal_idx = (
            goal_idx + 1
        )  # (+1) PutObject (get the objectId from the PutObject action)
        inventory_objects = state.metadata["inventoryObjects"]
        if next_put_goal_idx < len(expert_plan):
            cool_object_id = expert_plan[next_put_goal_idx]["planner_action"][
                "objectId"
            ]
            cool_object = get_object(cool_object_id, state.metadata)
            is_obj_cool = cool_object["objectId"] in self.env.cooled_objects

            # TODO(mohit): support dense rewards for all subgoals
            # intermediate reward if object is cooled
            if is_obj_cool and not self.env.cooled_reward:
                self.env.cooled_reward = True
                reward, done = self.rewards["positive"], False

            # intermediate reward for opening fridge after object is cooled
            elif is_obj_cool and state.metadata["lastAction"] == "OpenObject":
                target_recep = get_object(subgoal["objectId"], state.metadata)
                if target_recep is not None and not self.env.reopen_reward:
                    if (
                        target_recep["isOpen"]
                        and target_recep["objectType"] == "Fridge"
                    ):
                        self.env.reopen_reward = True
                        reward, done = self.rewards["positive"], False

            # intermediate reward for picking up cooled object after reopening fridge
            elif is_obj_cool and state.metadata["lastAction"] == "PickupObject":
                if len(inventory_objects):
                    inv_object_id = state.metadata["inventoryObjects"][0]["objectId"]
                    if inv_object_id == cool_object_id:
                        reward, done = (
                            self.rewards["positive"],
                            False,
                        )
            # final reward for closing fridge after object is cooled and while holding object
            elif is_obj_cool and state.metadata["lastAction"] == "CloseObject":
                is_fridge_closed = not get_objects_of_type("Fridge", state.metadata)[0][
                    "isOpen"
                ]
                if len(inventory_objects) and is_fridge_closed:
                    inv_object_id = state.metadata["inventoryObjects"][0]["objectId"]
                    if inv_object_id == cool_object_id:
                        reward, done = self.rewards["positive"], True

        return reward, done


def get_action(action_type, gt_graph, env, reward_config, strict):
    action_type_str = action_type + "Action"

    if action_type_str in globals():
        action = globals()[action_type_str]
        return action(gt_graph, env, reward_config[action_type_str], strict)
    else:
        raise Exception("Invalid action_type %s" % action_type_str)
