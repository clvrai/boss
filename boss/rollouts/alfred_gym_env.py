from threading import Lock
import copy
from boss.alfred.env.thor_env import ThorEnv
import gym
from boss.utils.utils import AttrDict
from boss.models.large_language_model import LargeLanguageModel
from boss.rollouts.skill import Skill
from boss.utils.utils import (
    process_skill_strings,
    add_prefix_to_skill_string,
    generate_primitive_skill_list_from_eval_skill_info_list,
)
import os
import numpy as np
from boss.alfred.gen.graph import graph_obj
from boss.models.boss_model import ETIQLCritics
import random
from PIL import Image
from boss.rollouts.online_alfred_rewards import get_action
import torch
import json
from sentence_transformers.util import semantic_search
from torch.nn.utils.rnn import pad_sequence

DEFAULT_RENDER_SETTINGS = {
    "renderImage": True,
    "renderDepthImage": False,
    "renderClassImage": False,
    "renderObjectImage": False,
}

DEFAULT_TIME_LIMIT = 40
GYM_EVAL_STEP_RATIO = 5
DEFAULT_PATH = f"{os.environ['BOSS']}/boss/rollouts"
DATA_PATH = f'{os.environ["BOSS"]}/boss/alfred/data/json_2.1.0_merge_goto/preprocess'
REWARD_CONFIG_PATH = f"{os.environ['BOSS']}/boss/alfred/models/config/rewards.json"

vocab_path = f"{os.environ['BOSS']}/boss/models/low_level_actions.vocab"
vocab_obj_path = f"{os.environ['BOSS']}/boss/models/obj_cls.vocab"


# TODO: handle floorplan selection
class ALFREDRLBootstrappingEnv(gym.Env):
    """
    A wrapper that performs skill bootstrapping with the given environment
    """

    def __init__(
        self,
        eval_json: str,
        which_floorplan: int,
        lang_embedding_model=None,  # used for bootstrapping/saycan to embed language instructions for value func
        semantic_search_model=None,  # used for bootstrapping/saycan for language sentence matching
        use_llm: bool = True,  # use the large language model for bootstrapping
        device=None,  # used for bootstrapping/saycan for the value function
        rand_init: bool = False,  # randomly initialize the agent in a random spot
        value_func: ETIQLCritics = None,  # used for bootstrapping/saycan
        visual_preprocessor=None,  # used for bootstrapping/saycan
        llm_model: LargeLanguageModel = None,  # used for bootstrapping/saycan
        scene_type="valid_unseen",
        llm_logprob_weight=1.0,
        num_skills_to_sample=100,
        value_sampling_temp=1.0,
        llm_sampling_temp=1.0,
        obs_concat_length=1,  # number of consecutive frames to concatenate for bootstrapping model
        skill_match_with_dataset=True,
        num_skill_match_generations=10,
        use_value_func=True,
        use_value_for_next_skill=False,
        forced_max_skills_to_chain=None,
    ):
        super().__init__()
        self._num_subgoals = 0
        self._scene_list = None
        self._current_skill = None
        self._curr_subgoal_index = 0

        self.value_func = value_func
        if self.value_func is not None:
            assert device is not None, "Must provide a device for bootstrapping"
        if use_llm:
            assert (
                llm_model is not None
            ), "Must provide a LargeLanguageModel for bootstrapping"
        self.visual_preprocessor = visual_preprocessor
        self.lang_embedding_model = lang_embedding_model
        self.scene_type = scene_type
        self.device = device
        self.obs_buffer = None
        self.frame = None
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.llm_logprob_weight = llm_logprob_weight
        self._time_limit = DEFAULT_TIME_LIMIT
        self._action_info = None
        self._scene_file = None
        self._rand_init = rand_init
        self._num_skills_to_sample = num_skills_to_sample
        self._value_sampling_temp = value_sampling_temp
        self._llm_sampling_temp = llm_sampling_temp
        self.lock = (
            Lock()
        )  # thread-lock the VF to avoid running out of GPU mem across threads
        self.max_composite_skill_length = None
        self.num_skill_match_generations = num_skill_match_generations
        self.obs_concat_length = obs_concat_length
        self._composite_embeddings = None
        self._skill_match_with_dataset = skill_match_with_dataset
        self._primitive_annotation_embeddings = None
        self._next_skill_filtering_threshold = 0.9
        self._use_value_func = use_value_func
        self._use_value_for_next_skill = use_value_for_next_skill
        self.semantic_search_model = semantic_search_model
        self._eval_json = eval_json
        self._forced_max_skills_to_chain = forced_max_skills_to_chain
        self.task_args = AttrDict()
        self.task_args.reward_config = REWARD_CONFIG_PATH
        scene_file_name = os.path.join(
            DEFAULT_PATH, f"{self.scene_type}_scene_list.json"
        )
        self._thor_env = ThorEnv()
        # load scene json and grab scene_index
        with open(scene_file_name) as f:
            all_scenes = json.load(f)
        self._scene_file_name = scene_file_name

        # perform a bunch of extra setup stuff
        self._env_setup(which_floorplan)

        new_scene_list = []
        # filter out scenes that aren't in the task list
        # and also add extra info from the tasks for this scene list for bootstrapping
        # (this is a remnant of some older code and i'm afraid to touch it)
        for task in self._task_names:
            new_scene_list.extend([x for x in all_scenes if x["task"] == task["task"]])
            new_scene_list[-1]["starting_subgoal_id"] = task["starting_subgoal_id"]
            new_scene_list[-1]["repeat_id"] = task["repeat_id"]
        self._scene_list = new_scene_list
        assert len(self._scene_list) == len(self._task_pool)

        # defines the action vocabularies
        self.vocab = torch.load(vocab_path)
        self.vocab_obj = torch.load(vocab_obj_path)

    ###### USED FOR GYM ENVIRONMENT API AND EVALUATION ########
    @property
    def num_tasks(self):
        return len(self._task_pool)

    @property
    def num_subgoals_to_complete(self):
        return self._gym_num_subgoals_to_complete

    def _sample_task(
        self,
        specific_task=None,
    ):
        # sample a task from the task pool for gym reset
        if specific_task is None:
            specific_task = random.randint(0, len(self._task_pool) - 1)
        log = self._task_pool[specific_task]

        task = log["task"]
        REPEAT_ID = log["repeat_id"]
        eval_idx = log["subgoal_ids"][0]
        json_path = os.path.join(
            DATA_PATH, self.scene_type, task, "ann_%d.json" % REPEAT_ID
        )
        with open(json_path) as f:
            traj_data = json.load(f)
        return eval_idx, traj_data, REPEAT_ID, specific_task

    def reset(self, specific_task=None):
        # gym-compatible reset function
        first_subgoal_idx, traj_data, REPEAT_ID, specific_task = self._sample_task(
            specific_task
        )
        curr_task = self._task_pool[specific_task]
        self._gym_num_subgoals_to_complete = len(curr_task["primitive_skills"])
        expert_init_actions = [
            a["discrete_action"]
            for a in traj_data["plan"]["low_actions"]
            if a["high_idx"] < first_subgoal_idx
        ]
        num_primitive_steps_in_task = sum(
            [
                len(primitive_skill["api_action"])
                for primitive_skill in curr_task["primitive_skills"]
            ]
        )
        if specific_task is None:
            # training
            self.max_steps = self._gym_num_subgoals_to_complete * DEFAULT_TIME_LIMIT
        else:
            # evaluation
            self.max_steps = num_primitive_steps_in_task * GYM_EVAL_STEP_RATIO
        # scene setup
        scene_num = traj_data["scene"]["scene_num"]
        object_poses = traj_data["scene"]["object_poses"]
        dirty_and_empty = traj_data["scene"]["dirty_and_empty"]
        object_toggles = traj_data["scene"]["object_toggles"]

        scene_name = "FloorPlan%d" % scene_num
        self._thor_env.reset(scene_name)
        self._thor_env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        self._thor_env.step(dict(traj_data["scene"]["init_action"]))

        # setup task for reward
        self._thor_env.set_task(traj_data, self.task_args, reward_type="dense")

        t = 0

        # set the initial state of the environment to the target initial state of the sampled task
        while t < len(expert_init_actions):
            action = expert_init_actions[t]
            compressed_mask = (
                action["args"]["mask"] if "mask" in action["args"] else None
            )
            mask = (
                self._thor_env.decompress_mask(compressed_mask)
                if compressed_mask is not None
                else None
            )
            success, _, _, err, _ = self._thor_env.va_interact(
                action["action"], interact_mask=mask, smooth_nav=True, debug=False
            )
            t += 1
            if not success:
                print(
                    "Failed to execute expert action when initializing ALFRED env, retrying"
                )
                return self.reset(specific_task)
            _, _ = (
                self._thor_env.get_transition_reward()
            )  # advances the reward function
        curr_frame = np.uint8(self.last_event.frame)
        curr_lang_instruction = process_skill_strings(
            self._task_pool[specific_task]["annotation"]
        )[0]
        primitive_lang_instructions = process_skill_strings(
            [
                primitive_skill["annotations"]
                for primitive_skill in curr_task["primitive_skills"]
            ]
        )
        self._gym_primitive_lang_instructions = primitive_lang_instructions

        self._gym_lang_instruction = curr_lang_instruction
        self.obs = curr_frame
        self._gym_curr_subgoal_idx = first_subgoal_idx
        self._gym_first_subgoal_idx = first_subgoal_idx
        self._gym_curr_task_idx = specific_task
        self._gym_curr_step = 0
        # obs, info
        return self.obs, self._build_gym_info()

    def _build_gym_info(self):
        primitive_lang_instruction = None
        if self._gym_curr_subgoal_idx - self._gym_first_subgoal_idx < len(
            self._gym_primitive_lang_instructions
        ):
            primitive_lang_instruction = self._gym_primitive_lang_instructions[
                self._gym_curr_subgoal_idx - self._gym_first_subgoal_idx
            ]
        return AttrDict(
            task=self._gym_curr_task_idx,
            subgoal=self._gym_curr_subgoal_idx,
            timeout=self._gym_curr_step >= self.max_steps,
            lang_instruction=self._gym_lang_instruction,
            primitive_lang_instruction=primitive_lang_instruction,
        )

    @property
    def last_event(self):
        return self._thor_env.last_event

    @property
    def action_space(self):
        return gym.spaces.Dict(
            {
                "action": gym.spaces.Discrete(len(self.vocab) - 3),
                "object": gym.spaces.Discrete(len(self.vocab_obj)),
            }
        )

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)

    def step(self, action_dict: dict):
        # this is a gym step that actually follows the action space and observation space
        # action_dict is an ordered dict with keys "action" and "object"
        action, obj_action = action_dict["action"], action_dict["object"]
        # if action and obj_action are already given as strings, just use directly
        if not isinstance(action, str):
            action = self.vocab[action + 3]  # + 3 offset for the 3 unused actions
            obj_action = self.vocab_obj[obj_action]
        try:
            _, _ = self._thor_env.to_thor_api_exec(action, obj_action, smooth_nav=True)
        except Exception as e:
            # ignore failed action execution
            pass
        self._gym_curr_step += 1
        self.obs = np.uint8(self.last_event.frame)
        _, _ = self._thor_env.get_transition_reward()  # advances the reward function
        # subgoal_idx from thor_env starts at -1 and iterates up
        new_subgoal_idx = self._thor_env.get_subgoal_idx() + 1
        rew = 0
        if new_subgoal_idx > self._gym_curr_subgoal_idx:
            self._gym_curr_subgoal_idx = new_subgoal_idx
            rew = 1
        done = (
            new_subgoal_idx
            == self._gym_num_subgoals_to_complete + self._gym_first_subgoal_idx
        )
        info = self._build_gym_info()
        return self.obs, rew, done, info

    ###### END OF USED ONLY FOR GYM ENVIRONMENT API AND EVALUATION ########

    def _env_setup(self, which_floorplan):
        # list of all evaluation tasks
        with open(self._eval_json, "r") as f:
            eval_skill_info_list = json.load(f)

        # sort skill info list by num_primitive_skills, descending, for faster evaluation with multiple threads
        eval_skill_info_list.sort(
            key=lambda x: sum(
                [
                    len(primitive_skill["api_action"])
                    for primitive_skill in x["primitive_skills"]
                ]
            ),
            reverse=True,
        )

        # make sure all tasks have at least 2 primitive skills
        assert all([len(x["primitive_skills"]) >= 2 for x in eval_skill_info_list])
        with open(self._scene_file_name, "r") as f:
            all_scenes_json = json.load(f)
        floorplan_set = set()
        all_floorplans = [
            all_scenes_json[x["primitive_skills"][0]["scene_index"]]["scene_num"]
            for x in eval_skill_info_list
        ]
        unique_floorplans = [
            x
            for x in all_floorplans
            if not (x in floorplan_set or floorplan_set.add(x))
        ]
        assert which_floorplan < len(unique_floorplans)
        eval_skill_info_list = [
            x
            for x in eval_skill_info_list
            if all_scenes_json[x["primitive_skills"][0]["scene_index"]]["scene_num"]
            == unique_floorplans[which_floorplan]
        ]
        sorted_task_names = [
            dict(
                task=task["task"],
                starting_subgoal_id=min(task["subgoal_ids"]),
                repeat_id=task["repeat_id"],
            )
            for task in eval_skill_info_list
        ]
        primitive_skills_to_use = []
        task_lengths = []
        for task in eval_skill_info_list:
            primitive_skills_to_use.append(
                generate_primitive_skill_list_from_eval_skill_info_list([task])
            )
            if self._forced_max_skills_to_chain is not None:
                task_lengths.append(self._forced_max_skills_to_chain)
            else:
                task_lengths.append(len(task["primitive_skills"]))

        # aggregate all primitive skills for the initial skill library if they have the same floorplan
        floorplan_per_task = []
        for task in eval_skill_info_list:
            floorplan_per_task.append(
                all_scenes_json[task["primitive_skills"][0]["scene_index"]]["scene_num"]
            )
        aggregated_tasks = {}
        for task, fp in zip(eval_skill_info_list, floorplan_per_task):
            if fp not in aggregated_tasks:
                aggregated_tasks[fp] = copy.deepcopy(task)
            else:
                aggregated_tasks[fp]["primitive_skills"].extend(
                    task["primitive_skills"]
                )

        # now separate by floorplan
        all_primitive_skills_per_floorplan = {
            fp: generate_primitive_skill_list_from_eval_skill_info_list(
                [aggregated_tasks[fp]]
            )
            for fp in aggregated_tasks
        }
        # now add the primitive skills for each task
        primitive_skills_to_use = []
        for fp in floorplan_per_task:
            primitive_skills_to_use.append(all_primitive_skills_per_floorplan[fp])
        print(
            f"Evaluating on {len(sorted_task_names)} tasks. Total {[len(primitive_skills_to_use[i]) for i in range(len(primitive_skills_to_use))]} skills"
        )
        self._task_pool = eval_skill_info_list
        self._task_names = sorted_task_names
        # sets the max length for each task
        self._task_lengths = task_lengths
        # list of primitive skills to allow skill chaining with
        self._primitive_skills_to_use = primitive_skills_to_use

    ###### BELOW FOLLOWS MAINLY CODE FOR BOOTSTRAPPING ######

    def cleanup(self):
        self._thor_env.stop()

    def bootstrap_reset(
        self,
        composite_skill_list: list[
            dict
        ] = [],  # list of already learned composite skill objects to use for bootstrapping
    ):
        # grab scene info and sample scene if needed
        scene_index = random.randint(0, len(self._scene_list) - 1)

        self._scene_index = scene_index
        self.max_composite_skill_length = self._task_lengths[self._scene_index]

        self._scene_name = self.get_scene_name()
        self._gt_graph = graph_obj.Graph(
            use_gt=True,
            construct_graph=True,
            scene_id=self._scene_name.split("Plan")[1],
        )

        # we'll change the number of subgoals on the fly in training mode
        self._num_subgoals = float("inf")

        self._curr_subgoal_index = 0
        # initialize with the first action of this scene
        init_action = self.initialize(None)
        self._generate_obs(self.last_event.frame, reset_buffer=True)
        self.composite_skill_list = composite_skill_list
        self.new_composite_skill = None
        self.second_sampled_skill = None
        success = False
        while not success:
            self._current_skill = None
            extra_info = {}
            new_skill, values, skill_types = self.sample_new_skill(extra_info)
            self._current_skill = new_skill
            success = self._set_subskill_type(
                self._current_skill, self._curr_subgoal_index
            )
        self._set_current_language_embedding()
        # set time limit for each primitive skill in sampled skill
        # self._time_limit = self._current_skill.num_skills * DEFAULT_TIME_LIMIT
        self._time_limit = DEFAULT_TIME_LIMIT
        info = self.build_info(
            primitive_skill_type=self._action_info["planner_action"]["action"],
            values=values,
            scene_index=scene_index,
            sampled_skill_types=skill_types,
            init_action=init_action,
            **extra_info,
        )
        return self.obs_buffer, info

    def get_scene_name(self):
        if "floor_plan" in self._scene_list[self._scene_index]:
            return self._scene_list[self._scene_index]["floor_plan"]
        else:
            # this is evaling on the full json which doesn't have the same convention
            return f'FloorPlan{self._scene_list[self._scene_index]["primitive_skills"][0]["scene_index"]}'

    def sample_next_skill_with_llm(self, extra_info, threshold):
        # print("SAMPLING NEXT SKILL WITH SEMANTIC SEARCH")
        if self._current_skill.num_skills >= self.max_composite_skill_length:
            return None
        if threshold < 0:
            return None
        # semantic search, by first generating skills from LLM and then matching
        # first_skill_annotation = [self._current_skill.composite_language_instruction]
        first_skill_annotation = self._current_skill.primitive_instructions_to_compose
        # get primitive annotation embeddings
        valid_primitive_skills = [
            primitive_skill
            for primitive_skill in self._primitive_skills_to_use[self._scene_index]
            # # if primitive_skill["scene_index"] == self._scene_index
        ]
        valid_primitive_annotations = process_skill_strings(
            [skill["annotations"] for skill in valid_primitive_skills]
        )

        # add prefix like PLACE: Place the thing in the ...
        prefix_valid_primitive_annotations = add_prefix_to_skill_string(
            valid_primitive_annotations
        )
        # don't embed with prefix as we are using these for the actual skill descriptions and such
        primitive_embeddings = self.lang_embedding_model.encode(
            valid_primitive_annotations,
        )

        unique_corpus_ids = []
        # generate next skill prediction via LLM
        (
            next_generations,
            llm_logprobs,
        ) = self.llm_model.generate_next_skill_with_other_skills(
            [first_skill_annotation],
            [valid_primitive_annotations],
            self.num_skill_match_generations,
        )
        prefix_next_generations = add_prefix_to_skill_string(next_generations)
        # encode the embeddings with semantic model
        # encoded_generations = self.lang_embedding_model.encode(next_generations)

        # search with prefix added generations
        encoded_search_generations = self.semantic_search_model.encode(
            prefix_next_generations
        )
        search_annotation_search_embeddings = self.semantic_search_model.encode(
            prefix_valid_primitive_annotations
        )
        # searched_results gives us a list of lists of top k results for each generation
        searched = semantic_search(
            encoded_search_generations, search_annotation_search_embeddings, top_k=1
        )
        unique_corpus_ids = []
        kept_indicies = []
        for i, res in enumerate(searched):
            if (
                res[0]["score"] > threshold
                and res[0]["corpus_id"] not in unique_corpus_ids
            ):
                unique_corpus_ids.append(res[0]["corpus_id"])
                kept_indicies.append(i)

        if len(unique_corpus_ids) == 0:
            max_score = max([x["score"] for res in searched for x in res])
            # binary search to find new threshold
            new_threshold = min(
                threshold - 0.05, (threshold - max_score) / 2 + max_score
            )
            print(
                f"NO good matches, sampling again. Max score: {max_score}, new threshold: {new_threshold}"
            )
            return self.sample_next_skill_with_llm(extra_info, threshold=new_threshold)

        # grab the skills of the results
        id_to_skill_map = {}
        for corpus_id in unique_corpus_ids:
            id_to_skill_map[corpus_id] = valid_primitive_skills[corpus_id]
        # grab the primitive skill annotations corresponding to this
        searched_primitive_results = [
            [valid_primitive_annotations[corpus_id]]
            for corpus_id in unique_corpus_ids
            if corpus_id < len(valid_primitive_skills)
        ]
        searched_primitive_embeddings = [
            primitive_embeddings[corpus_id]
            for corpus_id in unique_corpus_ids
            if corpus_id < len(valid_primitive_annotations)
        ]
        searched_primitive_ids = [
            corpus_id
            for corpus_id in unique_corpus_ids
            if corpus_id < len(valid_primitive_annotations)
        ]
        # order the ids
        ordered_search_ids = searched_primitive_ids

        # add them up to be sent as logprob queries to the LLM
        if llm_logprobs is None:
            llm_logprobs = self.llm_model.query_logprobs_with_other_skills(
                first_skill_annotation,
                searched_primitive_results,
                valid_primitive_annotations,
            )
        else:
            llm_logprobs = llm_logprobs[kept_indicies]

        # ordered search embeddings as primitives first then composites next so same order as llm_logprobs, then make them tensors
        ordered_search_embeddings = searched_primitive_embeddings

        # now grab the actual skills and their values and check validity
        other_shape = [1] * len(self.obs_buffer.shape[1:])
        repeated_obs = self.obs_buffer.repeat(
            len(ordered_search_embeddings), *other_shape
        )
        if self._use_value_func:
            with self.lock:
                # ET transformer model
                ordered_search_embeddings = pad_sequence(
                    ordered_search_embeddings, batch_first=True, padding_value=0
                )
                value = self.value_func.get_value(
                    repeated_obs.to(self.device),
                    ordered_search_embeddings.to(self.device),
                )
        else:
            value = torch.ones(repeated_obs.shape[0])
        value = torch.clamp(value, min=0.0)  # clamp value to be at least 0

        # generate mask for valid skills, check each skill to see if valid
        valid_mask = []
        all_sampled_skill_objects = []
        for i, id in enumerate(ordered_search_ids):
            skill, embedding = id_to_skill_map[id], ordered_search_embeddings[i]
            if isinstance(skill, Skill):
                input_skill_object = skill
            else:
                input_skill_object = Skill([skill], embeddings=[embedding])
            all_sampled_skill_objects.append(input_skill_object)
            success = self._set_subskill_type(input_skill_object, 0)
            success = (
                # self._current_action.valid_check
                success
                and input_skill_object.num_skills + self._current_skill.num_skills
                <= self.max_composite_skill_length
            )
            valid_mask.append(success)

        extra_info["valid_mask"] = valid_mask
        extra_info["is_composite"] = [0] * len(searched_primitive_results)
        # boolean mask for invalid skills
        invalid_mask = ~torch.tensor(valid_mask, dtype=torch.bool)
        if torch.all(invalid_mask):
            # redo
            print("all invalid")
            return self.sample_next_skill_with_llm(
                extra_info, threshold=threshold - 0.05
            )

        # sort logprob indicies, descending
        sorted_indicies = torch.argsort(llm_logprobs, dim=-1, descending=True)
        sorted_indicies = sorted_indicies.cpu().numpy()
        # get the sentences corresponding to the sorted indicies, store the logprobs also
        sorted_sentences = [
            searched_primitive_results[index] for index in sorted_indicies
        ]
        sorted_probs = [llm_logprobs[index].exp().item() for index in sorted_indicies]

        print(
            f"Original Prompt: {self._current_skill.get_precomposed_language_instructions()}"
        )
        print(f"Top 3 LLM Preds: {sorted_sentences[:3]}, {sorted_probs[:3]}")
        print(f"Bottom 3 LLM Preds: {sorted_sentences[-3:]}, {sorted_probs[-3:]}")

        logvals = torch.log(value).cpu()
        # if logval is -inf since value can be 0 or negative, set it to -100
        logvals[torch.nonzero(torch.isnan(logvals))] = -100

        # set logprobs for invalid skills to -100
        logvals[invalid_mask] = -100
        llm_logprobs[invalid_mask] = -100
        value[invalid_mask] = 0

        # now we have a list of candidate skill sentences, grab their values
        skill_probability_dist = (logvals / self._value_sampling_temp) * (
            1 - self.llm_logprob_weight
        ) + (llm_logprobs / self._llm_sampling_temp) * self.llm_logprob_weight

        skill_index = (
            torch.distributions.Categorical(logits=skill_probability_dist)
            .sample()
            .item()
        )

        new_skill = all_sampled_skill_objects[skill_index]

        extra_info["sampled_skill_llm_prob"] = llm_logprobs[skill_index].exp().item()

        skill_types = [
            id_to_skill_map[skill]["planner_action"]["action"]
            for skill in searched_primitive_ids
        ]  # + ["Composite" for _ in searched_composite_ids]

        return new_skill, value.cpu(), llm_logprobs.exp(), skill_types

    def sample_new_skill(self, extra_info):
        # first sample primitive skills
        # then sample the same number of composite skills
        # then pass through vf
        num_current_skills = (
            0 if self._current_skill is None else self._current_skill.num_skills
        )

        if num_current_skills >= self.max_composite_skill_length:
            return None

        primitive_skills = [
            primitive_skill
            for primitive_skill in self._primitive_skills_to_use[self._scene_index]
        ]

        primitive_lang_annotations = process_skill_strings(
            [sample["annotations"] for sample in primitive_skills]
        )
        primitive_embeddings = self.lang_embedding_model.encode(
            primitive_lang_annotations,
        )
        valid_composite_skills = [
            skill
            for skill in self.composite_skill_list
            if skill["scene_index"] == self._scene_index
        ]
        composite_sample_indicies = random.sample(
            range(len(valid_composite_skills)),
            min(self._num_skills_to_sample, len(valid_composite_skills)),
        )
        # composite embeddings will be skill language embeddings of the entire composite skill's instruction
        composite_embeddings = []
        # composite_sample is a list of composite skills
        composite_sample = []
        if len(composite_sample_indicies) > 0:
            for index in composite_sample_indicies:
                sampled_composite_skill = valid_composite_skills[index]["skill"]
                composite_embeddings.append(
                    sampled_composite_skill.composite_language_embedding
                )
                composite_sample.append(sampled_composite_skill)
            sample_lang_embeddings = primitive_embeddings + composite_embeddings
        else:
            sample_lang_embeddings = primitive_embeddings
        all_sample_skills = primitive_skills + composite_sample

        use_vf_this_time = self._use_value_func and (
            self._current_skill is None or self._use_value_for_next_skill
        )

        with self.lock:
            # get the value of the current skill
            other_shape = [1] * len(self.obs_buffer.shape[1:])
            repeated_obs = self.obs_buffer.repeat(
                len(sample_lang_embeddings), *other_shape
            )
            if use_vf_this_time:
                sample_lang_embeddings = pad_sequence(
                    sample_lang_embeddings, batch_first=True, padding_value=0
                )
                value = self.value_func.get_value(
                    repeated_obs.to(self.device),
                    sample_lang_embeddings.to(self.device),
                )
            else:
                value = torch.ones(repeated_obs.shape[0])
        value = torch.clamp(
            value,
            min=0.0,
        )  # clamp value to be at least 0

        # generate mask for valid skills, check each skill to see if valid
        valid_mask = []
        for skill, embedding in zip(all_sample_skills, sample_lang_embeddings):
            # success = self._set_subskill_type(Skill([skill], embeddings=[embedding]), 0)
            input_skill_object = Skill([skill], embeddings=[embedding])
            success = self._set_subskill_type(input_skill_object, 0)
            success = (
                success
                and input_skill_object.num_skills + num_current_skills
                <= self.max_composite_skill_length
            )
            valid_mask.append(success)
        extra_info["valid_mask"] = valid_mask
        extra_info["is_composite"] = [0] * len(primitive_skills) + [1] * len(
            composite_sample
        )
        logvals = torch.log(value)
        # if logval is -inf since value can be 0 or negative, set it to -100
        logvals[torch.nonzero(torch.isnan(logvals))] = -100
        invalid_mask = ~torch.tensor(valid_mask, dtype=torch.bool)
        if torch.all(invalid_mask):
            return self.sample_new_skill(extra_info)
        value[invalid_mask] = 0
        logvals = logvals.cpu()

        skill_probability_dist = logvals / self._value_sampling_temp

        if torch.any(invalid_mask):
            skill_probability_dist[invalid_mask] = -100
        if torch.all(skill_probability_dist == -100):
            # redo
            return self.sample_new_skill(extra_info)

        skill_index = (
            torch.distributions.Categorical(logits=skill_probability_dist)
            .sample()
            .item()
        )
        if skill_index < len(primitive_skills):
            most_likely_skill = primitive_skills[skill_index : skill_index + 1]
            new_skill = Skill(
                most_likely_skill,
                embeddings=sample_lang_embeddings[skill_index : skill_index + 1],
            )
        else:
            skill = valid_composite_skills[
                composite_sample_indicies[skill_index - len(primitive_skills)]
            ]
            new_skill = skill["skill"]

        skill_types = [
            skill["planner_action"]["action"] for skill in primitive_skills
        ] + ["Composite" for _ in composite_sample]

        return new_skill, value.cpu(), skill_types

    def build_info(self, **kwargs):
        info = AttrDict(
            frame=self.frame,
            timeout=self._time_limit == 0,
            current_skill=self._current_skill,
            composite_skill=self.new_composite_skill,
            second_sampled_skill=self.second_sampled_skill,
            per_step_primitive_skill_type=(
                self._action_info["planner_action"]["action"]
                if self._action_info
                else None
            ),  # used for writing primitive skill type in video
            lang_embedding=self._curr_lang_embedding,
        )
        info.update(kwargs)
        return info

    def bootstrap_step(self, action_dict: dict):
        # this is a bootstrapping step just for skill bootstrapping
        # action_dict is an ordered dict with keys "action" and "object"
        action, obj_action = action_dict["action"], action_dict["object"]
        # execute action
        try:
            _, _ = self._thor_env.to_thor_api_exec(action, obj_action, smooth_nav=True)
        except:
            # exceptions can be thrown if the action turned out to be invalid; just ignore it
            self._time_limit -= 1
            self._generate_obs(self.last_event.frame, reset_buffer=False)
            return (
                self.obs_buffer,
                0,
                False,
                self.build_info(subgoal_finished=False, error=True),
            )

        # check if the subgoal is finished
        reward, sg_done = self._current_action.get_reward(self.last_event)
        done = False
        # if the current subgoal is done, check if we're done with the rollout or need to move on
        # to the next primitive skill, or sample a new primitive/composite skill
        extra_info = dict()
        if sg_done:
            self._curr_subgoal_index += 1
            done_with_rollout = self._curr_subgoal_index >= self._num_subgoals
            # if the current skill is max composite skill length then we're done, no need to sample a next skill.
            current_skill_is_max_len_and_done = (
                self._num_subgoals == self.max_composite_skill_length
            ) and done_with_rollout
            if done_with_rollout or current_skill_is_max_len_and_done:
                done = True
                self._generate_obs(self.last_event.frame, reset_buffer=False)
            else:
                # we're either setting or sampling a new skill, so extend the time limit back to the original
                self._time_limit = DEFAULT_TIME_LIMIT + 1
                # need to advance to the next primitive skill
                if self._curr_subgoal_index < self._current_skill.num_skills:
                    # reset the buffer when generating this new observation since we're onto the next primitive skill in this chain
                    self._generate_obs(self.last_event.frame, reset_buffer=True)
                    successful_subskill_set = self._set_subskill_type(
                        self._current_skill, self._curr_subgoal_index
                    )
                    if not successful_subskill_set:
                        # error
                        self._time_limit = 0
                        done = False
                        return (
                            self.obs_buffer,
                            reward,
                            done,
                            self.build_info(subgoal_finished=False, error=True),
                        )

                # current primitive/composite skill is done and we need to sample a new primitive/composite skill
                elif self._curr_subgoal_index == self._current_skill.num_skills:
                    # reset the buffer when generating this new observation because we're done with the current skill
                    self._generate_obs(self.last_event.frame, reset_buffer=True)
                    successful_subskill_set = False

                    while not successful_subskill_set:
                        if self.use_llm and self._skill_match_with_dataset:
                            ret = self.sample_next_skill_with_llm(
                                extra_info,
                                threshold=self._next_skill_filtering_threshold,
                            )
                        else:
                            ret = self.sample_new_skill(extra_info)
                        if ret is None:
                            # wasn't able to successfully sample a next skill, possibly due to composite skill length maximum.
                            self._time_limit = 0
                            done = True
                            return (
                                self.obs_buffer,
                                reward,
                                done,
                                self.build_info(subgoal_finished=True, error=True),
                            )
                        else:
                            next_skill, values, llm_probs, skill_types = ret
                        new_composite_skill = Skill(
                            [self._current_skill, next_skill],
                            embedding_model=self.lang_embedding_model,
                        )
                        successful_subskill_set = self._set_subskill_type(
                            new_composite_skill, self._curr_subgoal_index
                        )
                        if successful_subskill_set:
                            self._current_skill = new_composite_skill
                            self.new_composite_skill = new_composite_skill
                            self.second_sampled_skill = next_skill
                            self._num_subgoals = self._current_skill.num_skills
                            extra_info["values"] = values
                            extra_info["llm_probs"] = llm_probs
                            extra_info["sampled_skill_types"] = skill_types

                extra_info["primitive_skill_type"] = (
                    self._action_info["planner_action"]["action"],
                )
                self._set_current_language_embedding()
        else:
            # if subgoal is not done then generate the frame as normal without resetting the buffer
            self._generate_obs(self.last_event.frame, reset_buffer=False)

        self._time_limit -= 1
        info = self.build_info(subgoal_finished=sg_done, error=False)
        info.update(extra_info)
        return (
            self.obs_buffer,
            reward,
            done,
            info,
        )

    def _set_subskill_type(self, skill, index: int):
        if isinstance(skill, Skill):
            curr_primitive_skill = skill.get_skill_at_index(index)
        else:
            # it's a dict(primitive_skill_type: str, skill_embedding: lang)
            curr_primitive_skill = skill
        self._current_action = get_action(
            self._gt_graph, self._thor_env, curr_primitive_skill.skill_info
        )
        success = self._current_action.check_valid_skill(self.last_event)
        if success:
            self._action_info = curr_primitive_skill.skill_info
            # reset ThorEnv states so correct rewards can be computed after an action is completed. Heat/Cool rely on this.
            self._thor_env.reset_states()
        return success

    def _set_current_language_embedding(self):
        primitive_skill = self._current_skill.get_skill_at_index(
            self._curr_subgoal_index
        )
        self._curr_lang_embedding = primitive_skill.language_embedding.to(self.device)
        if len(self._curr_lang_embedding.shape) == 1:
            self._curr_lang_embedding = self._curr_lang_embedding.unsqueeze(0)

    def _generate_obs(self, frame, reset_buffer=False):
        self.frame = Image.fromarray(np.uint8(frame))
        processed_obs = self.visual_preprocessor.featurize([self.frame], batch=1).to(
            self.device
        )
        if reset_buffer:
            if len(processed_obs.shape) == 4:  # resnet image feature
                self.obs_buffer = processed_obs.unsqueeze(1)
            else:
                self.obs_buffer = processed_obs.repeat(1, self.obs_concat_length)
        else:
            self.obs_buffer = torch.cat(
                [self.obs_buffer, processed_obs.unsqueeze(1)], dim=1
            )
            self.obs_buffer = self.obs_buffer[
                :,
                -self.obs_concat_length :,
            ]

    def _generate_start_pose(self, graph, init_y):
        points = graph.org_points
        num_points = len(points)
        choose_point = random.randint(0, num_points - 1)
        point = points[choose_point]
        init_action = {}
        init_action["action"] = "TeleportFull"
        init_action["horizon"] = random.randint(2, 4) * 15
        init_action["rotateOnTeleport"] = True
        init_action["rotation"] = random.randint(0, 3) * 90
        init_action["x"] = point[0]
        init_action["y"] = init_y  # this changes from scene to scene, not sure why
        init_action["z"] = point[1]
        return init_action

    def initialize(self, specific_skill=None):
        # used to initialize to a specific skill for the bootstrap reset function
        self._thor_env.reset(self._scene_name)
        scene = self._scene_list[self._scene_index]
        self._thor_env.restore_scene(
            object_poses=scene["object_poses"],
            object_toggles=scene["object_toggles"],
            dirty_and_empty=scene["dirty_and_empty"],
        )
        if specific_skill is None:
            if self._rand_init:
                init_action = self._generate_start_pose(
                    self._gt_graph, scene["init_action"]["y"]
                )
            else:
                init_action = scene["init_action"]
        else:
            init_action = specific_skill["init_action"]
        self._thor_env.step(init_action)
        task = scene["task"]
        REPEAT_ID = scene["repeat_id"]
        starting_subgoal_id = scene["starting_subgoal_id"]
        json_path = os.path.join(
            DATA_PATH, self.scene_type, task, "ann_%d.json" % REPEAT_ID
        )
        with open(json_path) as f:
            traj_data = json.load(f)
        expert_init_actions = [
            a["discrete_action"]
            for a in traj_data["plan"]["low_actions"]
            if a["high_idx"] < starting_subgoal_id
        ]
        for t in range(len(expert_init_actions)):
            action = expert_init_actions[t]
            compressed_mask = (
                action["args"]["mask"] if "mask" in action["args"] else None
            )
            mask = (
                self._thor_env.decompress_mask(compressed_mask)
                if compressed_mask is not None
                else None
            )

            success, _, _, err, _ = self._thor_env.va_interact(
                action["action"], interact_mask=mask, smooth_nav=True, debug=False
            )
            if not success:
                print("Error: ", err, "Initializing with expert actions failed.")
                return self.initialize(specific_skill)
        return init_action
