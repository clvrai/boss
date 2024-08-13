from boss.models.large_language_model import LargeLanguageModel
from boss.utils.utils import add_prefix_to_skill_string, process_skill_strings

import random
import torch
from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer


class SaycanPlanner(LargeLanguageModel):
    prompt_start = "Robot: Hi there, I'm a robot operating in a house.\n"
    prompt_start += "Robot: You can ask me to do various tasks and I'll tell you the sequence of actions I would do to accomplish your task.\n"
    starter = "Human: How would you "
    prompt = prompt_start + starter
    prompt += "put the box with keys on the sofa next to the newspaper?\n"
    prompt += "Robot: "
    prompt += "1. Pick up the keys on the center table.\n"
    prompt += "2. Put the keys in the box.\n"
    prompt += "3. Pick up the box with keys.\n"
    prompt += "4. Put the box with keys on the sofa close to the newspaper.\n"

    prompt += starter + "cool a slice of lettuce and put it on the counter?\n"
    prompt += "Robot: "
    prompt += "1. Pick up the knife from in front of the tomato.\n"
    prompt += "2. Cut the lettuce on the counter.\n"
    prompt += "3. Set the knife down on the counter in front of the toaster.\n"
    prompt += "4. Pick up a slice of the lettuce from the counter.\n"
    prompt += "5. Put the lettuce slice in the refrigerator. take the lettuce slice out of the refrigerator.\n"
    prompt += "6. Set the lettuce slice on the counter in front of the toaster.\n"

    prompt += starter + "put the book on the table on the couch?\n"
    prompt += "Robot: "
    prompt += "1. Pick up the book on the table, in front of the chair.\n"
    prompt += "2. Place the book on the left cushion of the couch.\n"

    prompt += starter + "put the book on the table on the couch?\n"
    prompt += "Robot: "
    prompt += "1. Pick up the fork from the table.\n"
    prompt += "2. Put the fork in the sink and fill the sink with water, then empty the water from the sink and remove the fork.\n"
    prompt += "3. Put the fork in the drawer.\n"

    prompt += starter + "put two boxes of tissues on the barred rack?\n"
    prompt += "Robot: "
    prompt += "1. Take the box of tissues from the makeup vanity.\n"
    prompt += "2. Put the tissues on the barred rack.\n"
    prompt += "3. Take the box of tissues from the top of the toilet.\n"
    prompt += "4. Put the tissues on the barred rack.\n"

    prompt += starter + "put a heated glass from the sink onto the wooden rack?\n"
    prompt += "Robot: "
    prompt += "1. Pick up the glass from the sink.\n"
    prompt += "2. Heat the glass in the microwave.\n"
    prompt += "3. Put the glass on the wooden rack.\n"

    prompt += (
        starter + "look at the box from the far side of the bed under the lamp light?\n"
    )
    prompt += "Robot: "
    prompt += "1. Pick up the box from the far side of the bed.\n"
    prompt += "2. Hold the box and turn on the lamp.\n"

    prompt += starter
    prompt_mid_1 = "Robot: "
    prompt_mid_fn = lambda self, index, text: f"{index+1}. {text}\n"

    all_next_skill_prompt_start = prompt  # [: -len(starter)]
    all_next_skill_prompt_mid = "\nPredict the next skill correctly by choosing from the following next skills: "
    all_next_skill_aggregate_skills = (
        lambda self, text: f"{text.replace(text[-1], ';')}"
    )

    def __init__(self, config):
        config.llm_max_new_tokens = 30
        config.llm_next_skill_temp = 0.8
        config.llm_summary_temp = None
        super().__init__(config)
        self.threshold = 0.95
        print(f"SayCan Prompt:\n{self.prompt}")
        self.next_skill_top_p = 0.8
        self.skill_match_with_dataset = (
            config.skill_match_with_dataset  # this enables Saycan+P in the paper over just Saycan
        )
        self.lang_embedding_model = SentenceTransformer("all-mpnet-base-v2")

    def preprocess_llm_inputs_for_logprob(
        self,
        first_annotations: list[list[str]],
        second_annotations: list[str],
        high_level_tasks: list[str],
    ):
        modified_prompts_without_end = []
        only_part_twos = []
        for prompt_part_one, prompt_part_two, high_level_task in zip(
            first_annotations, second_annotations, high_level_tasks
        ):
            with_mid_annotations = [
                self.prompt_mid_fn(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)

            second_with_mid_annotations = self.prompt_mid_fn(next_i, prompt_part_two)
            high_level_task_question = high_level_task.lower()[:-1] + "?\n"
            modified_prompts_without_end.append(
                self.prompt
                + high_level_task_question  # task name
                + self.prompt_mid_1
                + "".join(with_mid_annotations)
                + second_with_mid_annotations
            )
            only_part_two = (
                # " " + prompt_part_two[0] + "\n" + second_with_mid_annotations[1:]
                # " " + prompt_part_two + "\n" + second_with_mid_annotations[4:] # get rid of the number and first letter (" 1. P" for example)
                second_with_mid_annotations[
                    2:
                ]  # get rid of the number (" 1." for example)
            )
            only_part_twos.append(only_part_two)
        all_tokenized_prompts_no_end = self.tokenizer(
            modified_prompts_without_end,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_second_parts = self.tokenizer(
            only_part_twos, padding=True, truncation=True, return_tensors="pt"
        )
        return (
            all_tokenized_prompts_no_end,
            tokenized_second_parts,
        )

    def _get_logprobs(
        self,
        first_annotation: list[str],
        sample_second_annotations: list[str],
        high_level_tasks: list[str],
    ):
        all_first_annotations = [first_annotation] * len(sample_second_annotations)
        high_level_tasks_repeated = high_level_tasks * len(sample_second_annotations)
        # use batch size
        for i in range(0, len(sample_second_annotations), self.llm_batch_size):
            (
                batch_tokenized_prompt_no_end,
                batch_tokenized_second_part,
            ) = self.preprocess_llm_inputs_for_logprob(
                all_first_annotations[i : i + self.llm_batch_size],
                sample_second_annotations[i : i + self.llm_batch_size],
                high_level_tasks_repeated[i : i + self.llm_batch_size],
            )
            batch_input_ids = batch_tokenized_prompt_no_end.input_ids
            batch_attention_mask = batch_tokenized_prompt_no_end.attention_mask
            batch_second_part_attention_mask = (
                batch_tokenized_second_part.attention_mask
            )
            logprobs = self._get_non_generated_logprobs_hf(
                batch_input_ids,
                batch_attention_mask,
                batch_second_part_attention_mask,
            )
            if i == 0:
                all_logprobs = logprobs.clone()
            else:
                all_logprobs = torch.cat((all_logprobs, logprobs), dim=0)
        return all_logprobs.cpu()

    def get_saycan_logprobs(
        self,
        first_annotation: list[str],
        sample_second_annotations: list[str],
        high_level_tasks: list[str],
    ):
        if self.skill_match_with_dataset:
            return self._get_logprobs_by_skill_match(
                first_annotation, sample_second_annotations, high_level_tasks
            )
        return self._get_logprobs(
            first_annotation, sample_second_annotations, high_level_tasks
        )

    def _get_logprobs_by_skill_match(
        self,
        first_annotation: list[str],
        sample_second_annotations: list[str],
        high_level_tasks: list[str],
    ):
        all_tokenized_prompts = (
            self.preprocess_llm_inputs_for_choosing_next_skill_generation(
                first_annotation, sample_second_annotations, high_level_tasks[0]
            )
        )
        found_next_skill = False
        threshold = self.threshold
        while not found_next_skill:
            next_skills, next_logprobs = self._generate_hf_text(
                all_tokenized_prompts,
                num_generations=5,
            )
            matched_skill_indicies, index_in_next_skill, new_threshold = (
                self.search_for_next_skill(
                    next_skills, sample_second_annotations, threshold
                )
            )
            if matched_skill_indicies is not None:
                found_next_skill = True
            threshold = new_threshold
        # now we have a list of matched skills, find logprob
        logprobs = torch.tensor([-float("inf")] * len(sample_second_annotations))
        matched_logprobs = [
            next_logprobs[index_in_next_skill[i]]
            for i in range(len(matched_skill_indicies))
        ]
        for i, index in enumerate(matched_skill_indicies):
            logprobs[index] = matched_logprobs[i]
        print(logprobs)
        return logprobs

    def preprocess_llm_inputs_for_choosing_next_skill_generation(
        self,
        first_annotations: list[str],
        all_possible_skills: list[str],
        high_level_task: str,
    ):
        modified_prompts_without_end = []
        # randomize the order of available next skills
        random.shuffle(all_possible_skills.copy())
        all_next_skill_sentence = (
            " ".join(
                self.all_next_skill_aggregate_skills(skill)
                for skill in process_skill_strings(all_possible_skills)
            )
            + "\n"
        )
        with_mid_annotations = [
            self.prompt_mid_fn(i, annotation)
            for i, annotation in enumerate(first_annotations)
        ]
        next_i = len(first_annotations)
        modified_prompts_without_end.append(
            self.all_next_skill_prompt_start
            + high_level_task.lower()[:-1]
            + "?\n"
            + self.all_next_skill_prompt_mid
            + all_next_skill_sentence
            + self.prompt_mid_1
            + "".join(with_mid_annotations)
            + str(next_i + 1)
            + "."
        )
        all_tokenized_prompts_no_end = self.tokenizer(
            modified_prompts_without_end,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return all_tokenized_prompts_no_end

    def search_for_next_skill(
        self,
        next_skill_candidates: list[str],  # comes from LLM
        all_skill_candidates: list[str],
        threshold,
    ):
        # add prefix like PLACE: Place the thing in the ...
        prefix_valid_primitive_annotations = add_prefix_to_skill_string(
            all_skill_candidates
        )

        unique_corpus_ids = []
        prefix_next_generations = add_prefix_to_skill_string(next_skill_candidates)

        # search with prefix-added annotations
        encoded_search_generations = self.lang_embedding_model.encode(
            prefix_next_generations
        )
        search_annotation_search_embeddings = self.lang_embedding_model.encode(
            prefix_valid_primitive_annotations
        )
        # searched_results gives us a list of lists of top k results for each generation
        searched = semantic_search(
            encoded_search_generations, search_annotation_search_embeddings, top_k=1
        )
        unique_corpus_ids = []
        index_in_next_skill_candidates = []
        for i, res in enumerate(searched):
            if (
                res[0]["score"] > threshold
                and res[0]["corpus_id"] not in unique_corpus_ids
            ):
                unique_corpus_ids.append(res[0]["corpus_id"])
                index_in_next_skill_candidates.append(i)
        # unique_corpus_ids = list(set(unique_corpus_ids))

        if len(unique_corpus_ids) == 0:
            max_score = max([x["score"] for res in searched for x in res])
            # binary search to find new threshold
            new_threshold = min(
                threshold - 0.05, (threshold - max_score) / 2 + max_score
            )
            print(
                f"NO good matches, sampling again. Max score: {max_score}, new threshold: {new_threshold}"
            )
            return None, None, new_threshold
        else:
            # return sorted(unique_corpus_ids), threshold
            return unique_corpus_ids, index_in_next_skill_candidates, threshold
