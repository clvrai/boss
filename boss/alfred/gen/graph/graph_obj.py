import os
import random
import time

import networkx as nx
import numpy as np

import boss.alfred.gen.constants as constants

# import boss.alfred.gen.constants
# import constants
from boss.alfred.gen.utils import game_util

MAX_WEIGHT_IN_GRAPH = 1e5
PRED_WEIGHT_THRESH = 10
EPSILON = 1e-4

# Direction: 0: north, 1: east, 2: south, 3: west


class Graph(object):
    def __init__(self, use_gt=False, construct_graph=True, scene_id=None, debug=False):
        t_start = time.time()

        self.construct_graph = construct_graph
        """
        event = env.step({'action': 'GetReachablePositions'})
        new_reachable_positions = event.metadata['reachablePositions']
        points = []
        for point in new_reachable_positions:
            xx = int(point['x'] / constants.AGENT_STEP_SIZE)
            yy = int(point['z'] / constants.AGENT_STEP_SIZE)
            points.append([xx, yy])

        self.points = np.array(points, dtype=np.int32)
        self.points = self.points[np.lexsort(self.points.T)]
        """

        self.scene_id = scene_id
        self.points = np.load(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                "layouts",
                "FloorPlan%s-layout.npy" % self.scene_id,
            )
        )

        self.points /= constants.AGENT_STEP_SIZE
        self.points = np.round(self.points).astype(np.int32)
        self.org_points = self.points.copy() * 0.25

        self.xMin = self.points[:, 0].min() - constants.SCENE_PADDING * 2
        self.yMin = self.points[:, 1].min() - constants.SCENE_PADDING * 2
        self.xMax = self.points[:, 0].max() + constants.SCENE_PADDING * 2
        self.yMax = self.points[:, 1].max() + constants.SCENE_PADDING * 2
        self.memory = np.zeros(
            (self.yMax - self.yMin + 1, self.xMax - self.xMin + 1), dtype=np.float32
        )
        self.gt_graph = None
        self.shortest_paths = {}
        self.shortest_paths_unweighted = {}
        self.use_gt = use_gt
        self.impossible_spots = set()
        self.updated_weights = {}
        self.prev_navigable_locations = None

        if self.use_gt:
            self.memory[:] = MAX_WEIGHT_IN_GRAPH
            self.memory[
                self.points[:, 1] - self.yMin, self.points[:, 0] - self.xMin
            ] = (1 + EPSILON)
        else:
            self.memory[:] = 1
            self.memory[:, : int(constants.SCENE_PADDING * 1.5)] = MAX_WEIGHT_IN_GRAPH
            self.memory[: int(constants.SCENE_PADDING * 1.5), :] = MAX_WEIGHT_IN_GRAPH
            self.memory[:, -int(constants.SCENE_PADDING * 1.5) :] = MAX_WEIGHT_IN_GRAPH
            self.memory[-int(constants.SCENE_PADDING * 1.5) :, :] = MAX_WEIGHT_IN_GRAPH

        if self.gt_graph is None:
            self.gt_graph = nx.DiGraph()
            if self.construct_graph:
                for yy in np.arange(self.yMin, self.yMax + 1):
                    for xx in np.arange(self.xMin, self.xMax + 1):
                        weight = self.memory[yy - self.yMin, xx - self.xMin]
                        for direction in range(4):
                            node = (xx, yy, direction)
                            back_direction = (direction + 2) % 4
                            back_node = (xx, yy, back_direction)
                            self.gt_graph.add_edge(
                                node, (xx, yy, (direction + 1) % 4), weight=1
                            )
                            self.gt_graph.add_edge(
                                node, (xx, yy, (direction - 1) % 4), weight=1
                            )
                            forward_node = None
                            if direction == 0 and yy != self.yMax:
                                forward_node = (xx, yy + 1, back_direction)
                            elif direction == 1 and xx != self.xMax:
                                forward_node = (xx + 1, yy, back_direction)
                            elif direction == 2 and yy != self.yMin:
                                forward_node = (xx, yy - 1, back_direction)
                            elif direction == 3 and xx != self.xMin:
                                forward_node = (xx - 1, yy, back_direction)
                            if forward_node is not None:
                                self.gt_graph.add_edge(
                                    forward_node, back_node, weight=weight
                                )

        self.initial_memory = self.memory.copy()
        self.debug = debug
        if self.debug:
            print("Graph construction time %.3f" % (time.time() - t_start))

    def clear(self):
        self.shortest_paths = {}
        self.shortest_paths_unweighted = {}
        self.impossible_spots = set()
        self.prev_navigable_locations = None

        if self.use_gt:
            self.memory[:] = self.initial_memory
        else:
            self.memory[:] = 1
            self.memory[:, : int(constants.SCENE_PADDING * 1.5)] = MAX_WEIGHT_IN_GRAPH
            self.memory[: int(constants.SCENE_PADDING * 1.5), :] = MAX_WEIGHT_IN_GRAPH
            self.memory[:, -int(constants.SCENE_PADDING * 1.5) :] = MAX_WEIGHT_IN_GRAPH
            self.memory[-int(constants.SCENE_PADDING * 1.5) :, :] = MAX_WEIGHT_IN_GRAPH

        if self.construct_graph:
            for (nodea, nodeb), original_weight in self.updated_weights.items():
                self.gt_graph[nodea][nodeb]["weight"] = original_weight
        self.updated_weights = {}

    @property
    def image(self):
        return self.memory[:, :].astype(np.uint8)

    def check_graph_memory_correspondence(self):
        # graph sanity check
        if self.construct_graph:
            for yy in np.arange(self.yMin, self.yMax + 1):
                for xx in np.arange(self.xMin, self.xMax + 1):
                    for direction in range(4):
                        back_direction = (direction + 2) % 4
                        back_node = (xx, yy, back_direction)
                        if direction == 0 and yy != self.yMax:
                            assert (
                                abs(
                                    self.gt_graph[(xx, yy + 1, back_direction)][
                                        back_node
                                    ]["weight"]
                                    - self.memory[
                                        int(yy - self.yMin), int(xx - self.xMin)
                                    ]
                                )
                                < 0.0001
                            )
                        elif direction == 1 and xx != self.xMax:
                            assert (
                                abs(
                                    self.gt_graph[(xx + 1, yy, back_direction)][
                                        back_node
                                    ]["weight"]
                                    - self.memory[
                                        int(yy - self.yMin), int(xx - self.xMin)
                                    ]
                                )
                                < 0.0001
                            )
                        elif direction == 2 and yy != self.yMin:
                            assert (
                                abs(
                                    self.gt_graph[(xx, yy - 1, back_direction)][
                                        back_node
                                    ]["weight"]
                                    - self.memory[
                                        int(yy - self.yMin), int(xx - self.xMin)
                                    ]
                                )
                                < 0.0001
                            )
                        elif direction == 3 and xx != self.xMin:
                            assert (
                                abs(
                                    self.gt_graph[(xx - 1, yy, back_direction)][
                                        back_node
                                    ]["weight"]
                                    - self.memory[
                                        int(yy - self.yMin), int(xx - self.xMin)
                                    ]
                                )
                                < 0.0001
                            )
            print("\t\t\tgraph tested successfully")

    def update_graph(self, graph_patch, pose):
        graph_patch, curr_val = graph_patch
        curr_val = np.array(curr_val)
        # Rotate the array to get its global coordinate frame orientation.
        rotation = int(pose[2])
        assert rotation in {0, 1, 2, 3}, "rotation was %s" % str(rotation)

        if rotation != 0:
            graph_patch = np.rot90(graph_patch, rotation)
        # Shift offsets to global coordinate frame.
        if rotation == 0:
            x_min = pose[0] - int(constants.STEPS_AHEAD / 2)
            y_min = pose[1] + 1
        elif rotation == 1:
            x_min = pose[0] + 1
            y_min = pose[1] - int(constants.STEPS_AHEAD / 2)
        elif rotation == 2:
            x_min = pose[0] - int(constants.STEPS_AHEAD / 2)
            y_min = pose[1] - constants.STEPS_AHEAD
        elif rotation == 3:
            x_min = pose[0] - constants.STEPS_AHEAD
            y_min = pose[1] - int(constants.STEPS_AHEAD / 2)
        else:
            raise Exception("Invalid pose direction")
        if self.construct_graph:
            for yi, yy in enumerate(range(y_min, y_min + constants.STEPS_AHEAD)):
                for xi, xx in enumerate(range(x_min, x_min + constants.STEPS_AHEAD)):
                    self.update_weight(xx, yy, graph_patch[yi, xi, 0])
            self.update_weight(pose[0], pose[1], curr_val[0])

    def get_graph_patch(self, pose):
        rotation = int(pose[2])
        assert rotation in {0, 1, 2, 3}

        if rotation == 0:
            x_min = pose[0] - int(constants.STEPS_AHEAD / 2)
            y_min = pose[1] + 1
        elif rotation == 1:
            x_min = pose[0] + 1
            y_min = pose[1] - int(constants.STEPS_AHEAD / 2)
        elif rotation == 2:
            x_min = pose[0] - int(constants.STEPS_AHEAD / 2)
            y_min = pose[1] - constants.STEPS_AHEAD
        elif rotation == 3:
            x_min = pose[0] - constants.STEPS_AHEAD
            y_min = pose[1] - int(constants.STEPS_AHEAD / 2)
        else:
            raise Exception("Invalid pose direction")
        x_min -= self.xMin
        y_min -= self.yMin

        graph_patch = self.memory[
            y_min : y_min + constants.STEPS_AHEAD, x_min : x_min + constants.STEPS_AHEAD
        ].copy()

        if rotation != 0:
            graph_patch = np.rot90(graph_patch, -rotation)

        return graph_patch, self.memory[pose[1] - self.yMin, pose[0] - self.xMin].copy()

    def add_impossible_spot(self, spot):
        self.update_weight(spot[0], spot[1], MAX_WEIGHT_IN_GRAPH)
        self.impossible_spots.add(spot)

    def update_weight(self, xx, yy, weight):
        if (xx, yy) not in self.impossible_spots:
            if self.construct_graph:
                for direction in range(4):
                    node = (xx, yy, direction)
                    self.update_edge(node, weight)
            self.memory[yy - self.yMin, xx - self.xMin] = weight
            self.shortest_paths = {}

    def update_edge(self, pose, weight):
        rotation = int(pose[2])
        assert rotation in {0, 1, 2, 3}

        (xx, yy, direction) = pose
        back_direction = (direction + 2) % 4
        back_pose = (xx, yy, back_direction)
        if direction == 0 and yy != self.yMax:
            forward_pose = (xx, yy + 1, back_direction)
        elif direction == 1 and xx != self.xMax:
            forward_pose = (xx + 1, yy, back_direction)
        elif direction == 2 and yy != self.yMin:
            forward_pose = (xx, yy - 1, back_direction)
        elif direction == 3 and xx != self.xMin:
            forward_pose = (xx - 1, yy, back_direction)
        else:
            raise NotImplementedError("Unknown direction")
        if (forward_pose, back_pose) not in self.updated_weights:
            self.updated_weights[(forward_pose, back_pose)] = self.gt_graph[
                forward_pose
            ][back_pose]["weight"]
        self.gt_graph[forward_pose][back_pose]["weight"] = weight

    def get_shortest_path(self, pose, goal_pose):
        assert pose[2] in {0, 1, 2, 3}
        assert goal_pose[2] in {0, 1, 2, 3}

        # Store horizons for possible final look correction.
        curr_horizon = int(pose[3])
        goal_horizon = int(goal_pose[3])

        pose = tuple(int(pp) for pp in pose[:3])
        goal_pose = tuple(int(pp) for pp in goal_pose[:3])

        try:
            assert (
                self.construct_graph
            ), "Graph was not constructed, cannot get shortest path."
            assert pose in self.gt_graph, "start point not in graph"
            assert goal_pose in self.gt_graph, "start point not in graph"
        except Exception as ex:
            print("pose", pose, "goal_pose", goal_pose)
            raise ex

        if (pose, goal_pose) not in self.shortest_paths:
            path = nx.astar_path(
                self.gt_graph,
                pose,
                goal_pose,
                heuristic=lambda nodea, nodeb: (
                    abs(nodea[0] - nodeb[0])
                    + abs(nodea[1] - nodeb[1])
                    + abs(nodea[2] - nodeb[2])
                ),
                weight="weight",
            )
            for ii, pp in enumerate(path):
                self.shortest_paths[(pp, goal_pose)] = path[ii:]
        path = self.shortest_paths[(pose, goal_pose)]
        max_point = 1
        for ii in range(len(path) - 1):
            weight = self.gt_graph[path[ii]][path[ii + 1]]["weight"]
            if path[ii][:2] != path[ii + 1][:2]:
                if (
                    abs(
                        self.memory[
                            path[ii + 1][1] - self.yMin, path[ii + 1][0] - self.xMin
                        ]
                        - weight
                    )
                    > 0.001
                ):
                    print(
                        self.memory[
                            path[ii + 1][1] - self.yMin, path[ii + 1][0] - self.xMin
                        ],
                        weight,
                    )
                    raise AssertionError("weights do not match")
            if weight >= PRED_WEIGHT_THRESH:
                break
            max_point += 1
        path = path[:max_point]

        actions = [
            Graph.get_plan_move(path[ii], path[ii + 1]) for ii in range(len(path) - 1)
        ]
        Graph.horizon_adjust(actions, path, curr_horizon, goal_horizon)

        return actions, path

    def get_shortest_path_unweighted(self, pose, goal_pose):
        assert pose[2] in {0, 1, 2, 3}
        assert goal_pose[2] in {0, 1, 2, 3}
        curr_horizon = int(pose[3])
        goal_horizon = int(goal_pose[3])
        pose = tuple(int(pp) for pp in pose[:3])
        goal_pose = tuple(int(pp) for pp in goal_pose[:3])

        try:
            assert (
                self.construct_graph
            ), "Graph was not constructed, cannot get shortest path."
            assert pose in self.gt_graph, "start point not in graph"
            assert goal_pose in self.gt_graph, "start point not in graph"
        except Exception as ex:
            print("pose", pose, "goal_pose", goal_pose)
            raise ex

        if (pose, goal_pose) not in self.shortest_paths_unweighted:
            # TODO: swap this out for astar (might be get_shortest_path tho) and update heuristic to account for
            # TODO: actual number of turns.
            path = nx.shortest_path(self.gt_graph, pose, goal_pose)
            for ii, pp in enumerate(path):
                self.shortest_paths_unweighted[(pp, goal_pose)] = path[ii:]
        path = self.shortest_paths_unweighted[(pose, goal_pose)]

        actions = [
            Graph.get_plan_move(path[ii], path[ii + 1]) for ii in range(len(path) - 1)
        ]
        Graph.horizon_adjust(actions, path, curr_horizon, goal_horizon)
        return actions, path

    def update_map(self, env):
        event = env.step({"action": "GetReachablePositions"})
        new_reachable_positions = event.metadata["reachablePositions"]
        new_memory = np.full_like(self.memory[:, :], MAX_WEIGHT_IN_GRAPH)
        if self.construct_graph:
            for point in new_reachable_positions:
                xx = int(point["x"] / constants.AGENT_STEP_SIZE)
                yy = int(point["z"] / constants.AGENT_STEP_SIZE)
                new_memory[yy - self.yMin, xx - self.xMin] = 1 + EPSILON
        changed_locations = np.where(
            np.logical_xor(
                self.memory[:, :] == MAX_WEIGHT_IN_GRAPH,
                new_memory == MAX_WEIGHT_IN_GRAPH,
            )
        )
        for location in zip(*changed_locations):
            self.update_weight(
                location[1] + self.xMin, location[0] + self.yMin, 1 + EPSILON
            )

    def navigate_to_goal(self, game_state, start_pose, end_pose):
        # Look down

        self.update_map(game_state.env)

        start_angle = start_pose[3]
        if start_angle > 180:
            start_angle -= 360
        if start_angle != 45:  # pitch angle
            # Perform initial tilt to get to 45 degrees.
            tilt_pose = [pp for pp in start_pose]
            tilt_pose[3] = 45
            tilt_actions, _ = self.get_shortest_path(start_pose, tilt_pose)
            for action in tilt_actions:
                game_state.step(action)
            start_pose = tuple(tilt_pose)

        actions, path = self.get_shortest_path(start_pose, end_pose)
        while len(actions) > 0:
            for ii, (action, pose) in enumerate(zip(actions, path)):
                game_state.step(action)
                event = game_state.env.last_event
                last_action_success = event.metadata["lastActionSuccess"]

                if not last_action_success:
                    # Can't traverse here, make sure the weight is correct.
                    if action["action"].startswith("Look") or action[
                        "action"
                    ].startswith("Rotate"):
                        raise Exception(
                            "Look action failed %s" % event.metadata["errorMessage"]
                        )
                    self.add_impossible_spot(path[ii + 1])
                    break
            pose = game_util.get_pose(event)
            actions, path = self.get_shortest_path(pose, end_pose)
        print("nav done")

    @staticmethod
    def get_plan_move(pose0, pose1):
        if (pose0[2] + 1) % 4 == pose1[2]:
            action = {"action": "RotateRight"}
        elif (pose0[2] - 1) % 4 == pose1[2]:
            action = {"action": "RotateLeft"}
        else:
            action = {"action": "MoveAhead", "moveMagnitude": constants.AGENT_STEP_SIZE}
        return action

    @staticmethod
    def horizon_adjust(actions, path, hor0, hor1):
        if hor0 < hor1:
            for _ in range((hor1 - hor0) // constants.AGENT_HORIZON_ADJ):
                actions.append({"action": "LookDown"})
                path.append(path[-1])
        elif hor0 > hor1:
            for _ in range((hor0 - hor1) // constants.AGENT_HORIZON_ADJ):
                actions.append(
                    {
                        "action": "LookUp",
                    }
                )
                path.append(path[-1])

    def __get_nav_graph_point(self, thor_x, thor_z, exclude_points=None):
        """
        Get the index in the navigation graph nearest to the given x,z coord in AI2-THOR coordinates
        :param thor_x: x coordinate on AI2-THOR floor plan
        :param thor_z: z coordinate on AI2-THOR floor plan
        :param exclude_points: Any navigation graph points that cannot be used
        :param get_closest: If false, instead of returning closest navigation graph point, do nucleus sampling around
        the coordinate; if True return the closest navigation graph point
        """
        # if self.navigation_graph is None:
        #     self.__generate_navigation_graph()
        t_point = nearest_t_d = None
        distances = []
        for idx in range(len(self.points)):
            if exclude_points is not None and idx in exclude_points:
                distances.append(float("inf"))
                continue
            d = np.abs(self.points[idx, 0] - thor_x) + np.abs(
                self.points[idx, 1] - thor_z
            )
            distances.append(d)
            if t_point is None or d < nearest_t_d:
                t_point = idx
                nearest_t_d = d
        return t_point

    def __get_nav_graph_rot(self, thor_x, thor_z, thor_facing_x, thor_facing_z):
        """
        Get the cardinal direction to rotate to, to be facing (thor_facing_x, thor_facing_x=z) when standing at
        (thor_x, thor_z)
        :param thor_x: x Coordinate on Ai2-THOR floor plan where agent is standing
        :param thor_z: z Coordinate on Ai2-THOR floor plan where agent is standing
        :param thor_facing_x: x Coordinate on Ai2-THOR floor plan where agent is desired to face
        :param thor_facing_z: z Coordinate on Ai2-THOR floor plan where agent is desired to face
        """
        # Determine target rotation.
        if np.abs(thor_x - thor_facing_x) > np.abs(
            thor_z - thor_facing_z
        ):  # Difference is greater in the x direction.
            if thor_x - thor_facing_x > 0:  # Destination to the x left
                t_rot = (-1, 0)
            else:
                t_rot = (1, 0)
        else:  # Difference is greater in the z direction
            if thor_z - thor_facing_z > 0:  # Destination to the z above
                t_rot = (0, -1)
            else:
                t_rot = (0, 1)

        return t_rot

    def __get_nearest_object_face_to_position(self, obj, pos):
        """
        Examine the AI2-THOR property 'axisAlignedBoundingBox'['cornerPoints'] and return the pose closest to target
        pose specified in param pos
        :param obj: the object to examine the faces of
        :param pos: the target position to get near
        """
        coords = ["x", "y", "z"]
        if obj["pickupable"]:
            # For pickupable objects we don't actually need to examine corner points and doing so sometimes causes
            # errors with clones
            return obj["position"]
        if "axisAlignedBoundingBox" not in obj:
            print("No bounding box for object", obj)
        xzy_obj_face = {
            c: obj["axisAlignedBoundingBox"]["cornerPoints"][
                self.__argmin(
                    [
                        np.abs(
                            obj["axisAlignedBoundingBox"]["cornerPoints"][pidx][
                                coords.index(c)
                            ]
                            - pos[c]
                        )
                        for pidx in range(
                            len(obj["axisAlignedBoundingBox"]["cornerPoints"])
                        )
                    ]
                )
            ][coords.index(c)]
            for c in coords
        }
        return xzy_obj_face

    def __get_y_rot_from_xz(self, x, z):
        """
        Given (x, z) norm rotation (e.g., (0, 1)), return the closest degrees in [270, 180, 90, 0] matching.
        """
        s_rot_to_dir_degree = {(-1, 0): 270, (0, -1): 180, (1, 0): 90, (0, 1): 0}
        return s_rot_to_dir_degree[(x, z)]

    def aim_camera_at_object(self, obj, camera_id=0):
        """
        Position camera specified by camera_id such that object obj is visible; Used to set target object view for
        TEACh data collection interface
        :param obj: Object to face - an element of the output of get_objects()
        :param camera_id: A valid camera ID
        """
        nav_point_idx = self.__get_nav_graph_point(
            obj["position"]["x"], obj["position"]["z"]
        )
        face_obj_rot = self.__get_nav_graph_rot(
            self.points[nav_point_idx, 0],
            self.points[nav_point_idx, 1],
            obj["position"]["x"],
            obj["position"]["z"],
        )

        # Calculate the angle at which to look at the object to center it.
        # We look from the head height of the agent [https://github.com/allenai/ai2thor/issues/266]
        # Head gaze is the hypotenuse of a right triangle whose legs are the xz (floor) distance to the obj and the
        # difference in gaze versus object height.
        # To get the object 'face' instead of center (which could be out of frame, especially for large objects like
        # drawers and cabinets), we decide the x,z,y position of the obj as the min distance to its corners.
        xzy_obj_face = self.__get_nearest_object_face_to_position(
            obj, self.points[nav_point_idx]
        )
        xz_dist = np.sqrt(
            np.power(xzy_obj_face["x"] - self.points[nav_point_idx, 0], 2)
            + np.power(xzy_obj_face["z"] - self.points[nav_point_idx, 1], 2)
        )
        y_diff = 1.8 - xzy_obj_face["y"]
        theta = (
            np.arctan(y_diff / xz_dist) * 180.0 / np.pi
            if not np.isclose(xz_dist, 0)
            else 0
        )

        continuous_rot = self.__get_y_rot_from_xz(face_obj_rot[0], face_obj_rot[1])
        # action = dict(
        #     action="UpdateThirdPartyCamera",
        #     thirdPartyCameraId=camera_id,
        #     rotation=dict(x=theta,
        #                   y=continuous_rot,
        #                   z=0),
        #     position=dict(x=self.points[nav_point_idx, 0],
        #                   y=1.8,
        #                   z=self.points[nav_point_idx, 1]),
        # )

        # self.controller.step(action)
        point_0 = self.points[nav_point_idx, 0]
        point_1 = self.points[nav_point_idx, 1]
        return point_0, point_1, int(continuous_rot / 90), (round(theta / 15) * 15)

    def aim_camera_x_y_z(self, obj):
        """
        Position camera specified by camera_id such that object obj is visible; Used to set target object view for
        TEACh data collection interface
        :param obj: Object to face - an element of the output of get_objects()
        :param camera_id: A valid camera ID
        """
        nav_point_idx = self.__get_nav_graph_point(
            obj["position"]["x"], obj["position"]["z"]
        )
        face_obj_rot = self.__get_nav_graph_rot(
            self.points[nav_point_idx, 0],
            self.points[nav_point_idx, 1],
            obj["position"]["x"],
            obj["position"]["z"],
        )

        continuous_rot = self.__get_y_rot_from_xz(face_obj_rot[0], face_obj_rot[1])

        point_0 = self.points[nav_point_idx, 0]
        point_1 = self.points[nav_point_idx, 1]
        return point_0, point_1, int(continuous_rot / 90), 45


# if __name__ == "__main__":
#     # Test graphs
#     env = game_util.create_env()
#     scenes = sorted(constants.TRAIN_SCENE_NUMBERS + constants.TEST_SCENE_NUMBERS)
#     while True:
#         scene_id = random.choice(scenes)
#         graph = Graph(use_gt=True, construct_graph=True, scene_id=scene_id)
#         game_util.reset(
#             env,
#             scene_id,
#             render_image=False,
#             render_depth_image=False,
#             render_class_image=False,
#             render_object_image=False,
#         )
#         num_points = len(graph.points)
#         point1 = random.randint(0, num_points - 1)
#         point2 = point1
#         while point2 == point1:
#             point2 = random.randint(0, num_points)
#         point1 = graph.points[point1]
#         point2 = graph.points[point2]
#         start_pose = (point1[0], point1[1], random.randint(0, 3), 0)
#         end_pose = (point2[0], point2[1], random.randint(0, 3), 0)
#         agent_height = env.last_event.metadata["agent"]["position"]["y"]
#         action = {
#             "action": "TeleportFull",
#             "x": start_pose[0] * constants.AGENT_STEP_SIZE,
#             "y": agent_height,
#             "z": start_pose[1] * constants.AGENT_STEP_SIZE,
#             "rotateOnTeleport": True,
#             "rotation": start_pose[2],
#             "horizon": start_pose[3],
#         }
#         env.step(action)
#         actions, path = graph.get_shortest_path(start_pose, end_pose)
#         while len(actions) > 0:
#             for ii, (action, pose) in enumerate(zip(actions, path)):
#                 env.step(action)
#                 event = env.last_event
#                 last_action_success = event.metadata["lastActionSuccess"]

#                 if not last_action_success:
#                     # Can't traverse here, make sure the weight is correct.
#                     if action["action"].startswith("Look") or action[
#                         "action"
#                     ].startswith("Rotate"):
#                         raise Exception(
#                             "Look action failed %s" % event.metadata["errorMessage"]
#                         )
#                     graph.add_impossible_spot(path[ii + 1])
#                     break
#             pose = game_util.get_pose(event)
#             actions, path = graph.get_shortest_path(pose, end_pose)
#         if end_pose == pose:
#             print("made it")
#         else:
#             print("could not make it :(")
