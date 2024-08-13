from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import requests
import torch
from threading import Lock

from boss.utils.utils import process_skill_strings


class LargeLanguageModel:
    starter = "Task Steps:\n"
    summary_start = "Instructions: give a high-level description for the following steps describing common household tasks.\n\n"
    summary_prompt_start = summary_start
    summary_prompt_start += starter
    summary_prompt_start += "1. Pick up the keys on the center table.\n"
    summary_prompt_start += "2. Put the keys in the box.\n"
    summary_prompt_start += "3. Pick up the box with keys.\n"
    summary_prompt_start += (
        "4. Put the box with keys on the sofa close to the newspaper.\n"
    )
    summary_prompt_start += (
        "Summary: Put the box with keys on the sofa next to the newspaper.\n"
    )

    summary_prompt_start += starter
    summary_prompt_start += "1. Pick up the knife from in front of the tomato.\n"
    summary_prompt_start += "2. Cut the lettuce on the counter.\n"
    summary_prompt_start += (
        "3. Set the knife down on the counter in front of the toaster.\n"
    )
    summary_prompt_start += "4. Pick up a slice of the lettuce from the counter.\n"
    summary_prompt_start += "5. Put the lettuce slice in the refrigerator. take the lettuce slice out of the refrigerator.\n"
    summary_prompt_start += (
        "6. Set the lettuce slice on the counter in front of the toaster.\n"
    )
    summary_prompt_start += (
        "Summary: Cool a slice of lettuce and put it on the counter.\n"
    )

    summary_prompt_start += starter
    summary_prompt_start += "1. Pick up the book on the table, in front of the chair.\n"
    summary_prompt_start += "2. Place the book on the left cushion of the couch.\n"
    summary_prompt_start += "Summary: Put the book on the table on the couch.\n"

    summary_prompt_start += starter
    summary_prompt_start += "1. Pick up the fork from the table.\n"
    summary_prompt_start += "2. Put the fork in the sink and fill the sink with water, then empty the water from the sink and remove the fork.\n"
    summary_prompt_start += "3. Put the fork in the drawer.\n"
    summary_prompt_start += (
        "Summary: Rinse the fork in the sink and then put it in a drawer.\n"
    )

    summary_prompt_start += starter
    summary_prompt_start += "1. Take the box of tissues from the makeup vanity.\n"
    summary_prompt_start += "2. Put the tissues on the barred rack.\n"
    summary_prompt_start += "3. Take the box of tissues from the top of the toilet.\n"
    summary_prompt_start += "4. Put the tissues on the barred rack.\n"
    summary_prompt_start += "Summary: Put two boxes of tissues on the barred rack.\n"

    summary_prompt_start += starter
    summary_prompt_start += "1. Pick up the glass from the sink.\n"
    summary_prompt_start += "2. Heat the glass in the microwave.\n"
    summary_prompt_start += "3. Put the glass on the wooden rack.\n"
    summary_prompt_start += (
        "Summary: Put a heated glass from the sink onto the wooden rack.\n"
    )

    summary_prompt_start += starter
    summary_prompt_start += "1. Pick up the box from the far side of the bed.\n"
    summary_prompt_start += "2. Hold the box and turn on the lamp.\n"
    summary_prompt_start += (
        "Summary: Look at the box from the far side of the bed under the lamp light.\n"
    )

    summary_prompt_start += starter
    summary_prompt_mid = lambda self, index, text: f"{index+1}. {text}\n"
    summary_prompt_end = "Summary:"

    all_next_skill_prompt_start = (
        "Examples of common household tasks and their descriptions: \n\n"
    )
    all_next_skill_prompt_start = summary_prompt_start.replace(
        summary_start, all_next_skill_prompt_start
    )
    # replace summary with task
    all_next_skill_prompt_start = all_next_skill_prompt_start.replace(
        "Summary: ", "Task: "
    )
    # remove the last starter
    all_next_skill_prompt_start = all_next_skill_prompt_start[: -len(starter)]
    # replace with new text
    all_next_skill_prompt_start += "\nPredict the next skill correctly by choosing from the following next skills: "

    all_next_skill_aggregate_skills = (
        lambda self, text: f"{text.replace(text[-1], ';')}"
    )

    def __init__(self, config):
        assert (
            "opt" in config.llm_model
            or "gpt" in config.llm_model
            or "alpaca" in config.llm_model
            or "llama" in config.llm_model
            or "gemma" in config.llm_model
            or "None" in config.llm_model
        ), "No tokenizer support for non-gpt/opt models"
        super().__init__()
        self.config = config
        self.llm_gpus = config.llm_gpus
        self.llm_max_new_tokens = config.llm_max_new_tokens
        self.llm_batch_size = config.llm_batch_size
        if config.llm_model != "None":
            tokenizer_cls = AutoTokenizer
            model_cls = AutoModelForCausalLM
            self.tokenizer = tokenizer_cls.from_pretrained(
                config.llm_model,
                model_max_length=2048,
                use_fast=False,  # use fast is false to avoid threading issue
                use_auth_token=True,
            )

            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = model_cls.from_pretrained(
                config.llm_model,
                pad_token_id=self.tokenizer.pad_token_id,
                # device_map="auto",
                # load_in_8bit=True,
                # quantization_config=quantization_config,
                # torch_dtype=torch.float16,
                torch_dtype="auto",
                use_auth_token=True,
                # )
            ).to(self.llm_gpus[0])
            # ).to(self.llm_gpus[0])
        self.lock = Lock()
        self.next_skill_top_p = self.config.llm_next_skill_top_p
        self.next_skill_temp = self.config.llm_next_skill_temp
        self.summary_temp = self.config.llm_summary_temp

    def _get_logprobs_hf(
        self,
        output_dict,
    ):
        # get token logprobs from a hf output_dict
        scores_float32 = [score.float() for score in output_dict.logits]
        scores = self.model.compute_transition_scores(
            output_dict.sequences, scores_float32, normalize_logits=True
        )
        return torch.mean(scores, dim=-1).cpu()

    def _get_non_generated_logprobs_hf(
        self,
        input_prompt_input_ids: torch.Tensor,
        input_prompt_attn_mask: torch.Tensor,
        second_skill_attn_mask: torch.Tensor,
    ):
        # get token logprobs for tokens that are not generated
        second_skill_start_pos = second_skill_attn_mask.sum(-1)
        with torch.no_grad():
            with self.lock:
                logits = (
                    self.model(
                        input_prompt_input_ids.to(self.llm_gpus[0]),
                        attention_mask=input_prompt_attn_mask.to(self.llm_gpus[0]),
                        return_dict=True,
                    )
                    .logits.cpu()
                    .float()
                )
        input_ids = input_prompt_input_ids
        if self.tokenizer.bos_token_id is not None:
            # the start token is attended to
            second_skill_start_pos -= 1
            # every logit but the last one because the logits correspond to distributions over the NEXT token given the token at the position
            logits = logits[:, :-1]
            # shifted_input_ids to disregard start token
            input_ids = input_prompt_input_ids[:, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_specific_logprobs = logprobs.gather(2, input_ids.unsqueeze(2)).squeeze(2)
        token_logprobs = []
        for i in range(len(second_skill_start_pos)):
            token_logprobs.append(
                torch.mean(token_specific_logprobs[i, -second_skill_start_pos[i] :])
                # torch.sum(token_specific_logprobs[i, -second_skill_start_pos[i] :])
            )
        return torch.tensor(token_logprobs)

    def preprocess_llm_inputs_for_summarization(self, all_annotations: list[list]):
        # preprocesses annotations for summarization to generate skill labels
        modified_prompts = []
        for primitive_annotations in all_annotations:
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(primitive_annotations)
            ]
            modified_prompts.append(
                self.summary_prompt_start
                + "".join(with_mid_annotations)
                + self.summary_prompt_end
            )
        all_tokenized_prompts = self.tokenizer(
            modified_prompts, padding=True, truncation=True, return_tensors="pt"
        )
        return all_tokenized_prompts

    def preprocess_llm_inputs_for_logprob_summary_prompt(
        self,
        first_annotations: list[list[str]],
        second_annotations: list[list[str]],
    ):
        # process annotations for summarization to generate skill labels and also processess only part two annotations to get their logprobs
        modified_prompts_without_end = []
        only_part_twos = []
        for prompt_part_one, prompt_part_two in zip(
            first_annotations, second_annotations
        ):
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)
            # second_with_mid = self.summary_prompt_mid(next_i, prompt_part_two)

            second_with_mid_annotations = [
                self.summary_prompt_mid(next_i + i, annotation)
                for i, annotation in enumerate(prompt_part_two)
            ]
            modified_prompts_without_end.append(
                self.summary_prompt_start
                + "".join(with_mid_annotations)
                + "".join(second_with_mid_annotations)
                # + self.summary_prompt_end
            )
            only_part_two = (
                " "
                + prompt_part_two[0]
                + "\n"
                + "".join(second_with_mid_annotations[1:])
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

    def preprocess_llm_inputs_for_logprob(
        self,
        first_annotations: list[str],
        second_annotations: list[str],
    ):
        # proprocesses llm inputs just to calculate logprobs
        modified_prompts_without_end = []
        only_part_twos = []
        for prompt_part_one, prompt_part_two in zip(
            first_annotations, second_annotations
        ):
            prompt_part_one = prompt_part_one.lower()
            prompt_part_two = prompt_part_two.lower()
            modified_prompts_without_end.append(
                self.logprob_prompt_start
                + prompt_part_one
                + self.logprob_prompt_mid
                + prompt_part_two
            )
            if len(self.logprob_prompt_mid) > 0 and self.logprob_prompt_mid[-1] == " ":
                only_part_twos.append(" " + prompt_part_two)
            else:
                only_part_twos.append(prompt_part_two)

        all_tokenized_prompts_no_end = None
        tokenized_second_parts = None
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

    def query_logprobs(self, first_annotation, sample_second_annotations):
        # get logprobs for huggingface models with the sample_second_annotations given the prompt and first_annotations
        all_logprobs = []
        all_first_annotations = [first_annotation] * len(sample_second_annotations)
        for i in range(0, len(sample_second_annotations), self.llm_batch_size):
            (
                tokenized_prompt_no_end,
                tokenized_second_part,
            ) = self.preprocess_llm_inputs_for_logprob_summary_prompt(
                all_first_annotations[i : i + self.llm_batch_size],
                sample_second_annotations[i : i + self.llm_batch_size],
            )
            batch_prompt_annotation_ids = tokenized_prompt_no_end.input_ids
            batch_prompt_annotation_attn_mask = tokenized_prompt_no_end.attention_mask
            batch_second_annotation_attn_mask = tokenized_second_part.attention_mask
            batch_logprobs = self._get_non_generated_logprobs_hf(
                batch_prompt_annotation_ids,
                batch_prompt_annotation_attn_mask,
                batch_second_annotation_attn_mask,
            )
            all_logprobs.append(batch_logprobs)
        if len(all_logprobs) > 1:
            all_logprobs = torch.cat(all_logprobs, dim=0)
        else:
            all_logprobs = all_logprobs[0]
        return all_logprobs.cpu()

    def process_hf_generation(self, choice):
        # postprocesses huggingface output dictionaries and spits out the generated text in strings
        generated_tokens = choice["sequences"][:, -self.llm_max_new_tokens :].cpu()
        model_texts = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        generated_texts = []
        special_eos = "."
        for i in range(len(model_texts)):
            model_text = model_texts[i]
            if special_eos in model_text:
                model_text = model_text[: model_text.index(special_eos)]
            # reject bad responses
            if len(model_text) <= 5:
                return False
            generated_texts.append(model_text.strip())
        return process_skill_strings(generated_texts)

    def preprocess_llm_inputs_for_logprob_generation(
        self, first_annotations: list[list[str]]
    ):
        modified_prompts_without_end = []
        for prompt_part_one in first_annotations:
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)
            modified_prompts_without_end.append(
                self.summary_prompt_start
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

    def preprocess_llm_inputs_for_choosing_next_skill_generation(
        self, first_annotations: list[list[str]], all_possible_skills: list[list[str]]
    ):
        # preprocesses llm inputs for choosing next skills given a list of possible skills
        modified_prompts_without_end = []
        for prompt_part_one, available_next_skills in zip(
            first_annotations, all_possible_skills
        ):
            # randomize the order of available next skills
            random.shuffle(available_next_skills.copy())
            all_next_skill_sentence = (
                " ".join(
                    self.all_next_skill_aggregate_skills(skill)
                    for skill in process_skill_strings(available_next_skills)
                )
                + "\n"
            )
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)
            modified_prompts_without_end.append(
                self.all_next_skill_prompt_start
                + all_next_skill_sentence
                + self.starter
                + "".join(with_mid_annotations)
                + str(next_i + 1)
                + "."
            )
            # print(modified_prompts_without_end)
        all_tokenized_prompts_no_end = self.tokenizer(
            modified_prompts_without_end,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return all_tokenized_prompts_no_end

    def preprocess_llm_inputs_for_logprob_summary_prompt_all_skills(
        self,
        first_annotations: list[list[str]],
        all_possible_skills: list[list[str]],
        second_annotations: list[list[str]],
    ):
        # preprocesses llm inputs to generate a summary given all skills, and also preprocesses the second annotations to later get their logprobs
        modified_prompts_without_end = []
        only_part_twos = []
        for (
            prompt_part_one,
            prompt_part_two,
            available_next_skills,
        ) in zip(first_annotations, second_annotations, all_possible_skills):
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)
            # randomize the order of available next skills
            random.shuffle(available_next_skills.copy())
            all_next_skill_sentence = (
                " ".join(
                    self.all_next_skill_aggregate_skills(skill)
                    for skill in process_skill_strings(available_next_skills)
                )
                + "\n"
            )

            second_with_mid_annotations = [
                self.summary_prompt_mid(next_i + i, annotation)
                for i, annotation in enumerate(prompt_part_two)
            ]
            modified_prompts_without_end.append(
                self.all_next_skill_prompt_start
                + all_next_skill_sentence
                + self.starter
                + "".join(with_mid_annotations)
                + "".join(second_with_mid_annotations)
            )
            only_part_two = (
                " "
                + prompt_part_two[0]
                + "\n"
                + "".join(second_with_mid_annotations[1:])
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

    def query_logprobs_with_other_skills(
        self,
        first_annotation: list[str],
        sample_second_annotations: list[list[str]],
        all_possible_skills: list[str],
    ):
        # get logprobs considering a list of other skills
        all_first_annotations = [first_annotation] * len(sample_second_annotations)
        all_possible_skills_repeated = [all_possible_skills] * len(
            sample_second_annotations
        )
        all_logprobs = []
        for i in range(0, len(sample_second_annotations), self.llm_batch_size):
            (
                tokenized_prompt_no_end,
                tokenized_second_part,
            ) = self.preprocess_llm_inputs_for_logprob_summary_prompt_all_skills(
                all_first_annotations[i : i + self.llm_batch_size],
                all_possible_skills_repeated,
                sample_second_annotations[i : i + self.llm_batch_size],
            )
            batch_prompt_annotation_ids = tokenized_prompt_no_end.input_ids
            batch_prompt_annotation_attn_mask = tokenized_prompt_no_end.attention_mask
            batch_second_annotation_attn_mask = tokenized_second_part.attention_mask
            batch_logprobs = self._get_non_generated_logprobs_hf(
                batch_prompt_annotation_ids,
                batch_prompt_annotation_attn_mask,
                batch_second_annotation_attn_mask,
            )
            all_logprobs.append(batch_logprobs)
        if len(all_logprobs) > 1:
            all_logprobs = torch.cat(all_logprobs, dim=0)
        else:
            all_logprobs = all_logprobs[0]
        return all_logprobs.cpu()

    def generate_next_skill_with_other_skills_codex(
        self,
        first_annotations: list[list[str]],
        all_possible_skills: list[list[str]],
        num_generations,
    ):
        # unused, old code for using OpenAI codex (and other openai models) when considering all skill libary skills
        modified_prompts_without_end = []
        for prompt_part_one, available_next_skills in zip(
            first_annotations, all_possible_skills
        ):
            # randomize the order of available next skills
            random.shuffle(available_next_skills.copy())
            all_next_skill_sentence = (
                " ".join(
                    self.all_next_skill_aggregate_skills(skill)
                    for skill in process_skill_strings(available_next_skills)
                )
                + "\n"
            )
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)
            modified_prompts_without_end.append(
                self.all_next_skill_prompt_start
                + all_next_skill_sentence
                + self.starter
                + "".join(with_mid_annotations)
                + str(next_i + 1)
                + "."
            )
        import os
        import openai
        import backoff

        openai.api_key = os.getenv("OPENAI_API_KEY")

        @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
        def generate_completion(prompts):
            try:
                return openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompts,
                    max_tokens=25,
                    stop="\n",
                    temperature=self.next_skill_temp,
                    top_p=self.next_skill_top_p,
                    logprobs=1,
                    n=num_generations,
                )
            except:
                raise openai.error.RateLimitError
            # except openai.error.ServiceUnavailableError:
            #    raise openai.error.RateLimitError
            # except openai.error.APIError:
            #    raise openai.error.RateLimitError
            # except openai.error.APIConnectionError:
            #    raise openai.error.RateLimitError
            # except requests.exceptions.ReadTimeoutError:
            #    raise openai.error.RateLimitError
            # except openai.error.Timeout:
            #    raise openai.error.RateLimitError

        batch_size = 20
        completions = []
        next_skill_preds = []
        logprobs = []
        for i in range(0, len(modified_prompts_without_end), batch_size):
            completions = generate_completion(
                modified_prompts_without_end[i : i + batch_size]
            )
            for completion in completions.choices:
                next_skill_preds.append(completion.text)
                logprobs.append(
                    torch.tensor(completion.logprobs.token_logprobs).mean(
                        -1, keepdim=True
                    )
                )
        logprobs = torch.cat(logprobs, dim=0)
        return process_skill_strings(next_skill_preds), logprobs

    def generate_next_skill_codex(
        self, first_annotations: list[list[str]], num_generations
    ):
        # unused, old code for using OpenAI codex (and other openai models)
        modified_prompts_without_end = []
        for prompt_part_one in first_annotations:
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)
            modified_prompts_without_end.append(
                self.summary_prompt_start
                + "".join(with_mid_annotations)
                + str(next_i + 1)
                + "."
            )
        import os
        import openai
        import backoff

        openai.api_key = os.getenv("OPENAI_API_KEY")

        @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
        def generate_completion(prompts):
            try:
                return openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompts,
                    max_tokens=25,
                    stop="\n",
                    temperature=self.next_skill_temp,
                    top_p=self.next_skill_top_p,
                    logprobs=1,
                    n=num_generations,
                )
            except openai.error.ServiceUnavailableError:
                raise openai.error.RateLimitError
            except openai.error.APIError:
                raise openai.error.RateLimitError
            except openai.error.APIConnectionError:
                raise openai.error.RateLimitError
            except openai.error.Timeout:
                raise openai.error.RateLimitError
            except requests.exceptions.ReadTimeout:
                raise openai.error.RateLimitError

        batch_size = 20
        completions = []
        next_skill_preds = []
        logprobs = []
        for i in range(0, len(modified_prompts_without_end), batch_size):
            completions = generate_completion(
                modified_prompts_without_end[i : i + batch_size]
            )
            for completion in completions.choices:
                next_skill_preds.append(completion.text)
                logprobs.append(
                    torch.tensor(completion.logprobs.token_logprobs).mean(
                        -1, keepdim=True
                    )
                )
        logprobs = torch.cat(logprobs, dim=0)
        return process_skill_strings(next_skill_preds), logprobs

    def _generate_hf_text(
        self, all_tokenized_prompts, num_generations, ret_logprobs=True
    ):
        # actually generate text using huggingface models
        composite_skill_annotations = []
        all_responses = []
        annotation_ids = all_tokenized_prompts.input_ids[0:1].to(self.llm_gpus[0])
        annotation_attn_mask = all_tokenized_prompts.attention_mask[0:1].to(
            self.llm_gpus[0]
        )
        for i in range(0, num_generations, self.llm_batch_size):
            # use a thread-safe lock here to avoid calling the LLM with a batch size greater than the supported batch size
            with self.lock:
                bad_response = True
                while bad_response:
                    responses = self.model.generate(
                        annotation_ids,
                        attention_mask=annotation_attn_mask,
                        return_dict_in_generate=True,
                        max_new_tokens=self.llm_max_new_tokens,
                        do_sample=True,
                        top_p=self.next_skill_top_p,
                        temperature=self.next_skill_temp,
                        num_return_sequences=min(
                            self.llm_batch_size, num_generations - i
                        ),
                        output_logits=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_labels = self.process_hf_generation(responses)
                    bad_response = new_labels is False
            composite_skill_annotations.extend(new_labels)
            all_responses.append(responses)
        logprobs = []
        if ret_logprobs:
            for responses in all_responses:
                if "logits" in responses:
                    with self.lock:
                        logprobs.append(self._get_logprobs_hf(responses))
        if len(logprobs) > 0:
            logprobs = torch.cat(logprobs, dim=0)
        else:
            logprobs = None
        return composite_skill_annotations, logprobs

    def generate_next_skill_with_other_skills(
        self,
        all_annotation_list: list[list[str]],
        next_skill_candidates: list[list[str]],
        num_generations,
    ):
        # generates the next skill and prompts the LLM with other skills
        all_tokenized_prompts = (
            self.preprocess_llm_inputs_for_choosing_next_skill_generation(
                all_annotation_list, next_skill_candidates
            )
        )
        return self._generate_hf_text(all_tokenized_prompts, num_generations)

    def generate_next_skill(
        self, all_annotation_list: list[list[str]], num_generations
    ):
        # unused; generates the next skill without considering other skills
        all_tokenized_prompts = self.preprocess_llm_inputs_for_logprob_generation(
            all_annotation_list
        )
        return self._generate_hf_text(all_tokenized_prompts, num_generations)

    def generate_skill_labels(self, all_annotation_list: list[list[str]]):
        # labels skills by summarizing primitive annotations
        composite_skill_annotations = []
        for i in range(0, len(all_annotation_list), self.llm_batch_size):
            tokenized_prompts = self.preprocess_llm_inputs_for_summarization(
                all_annotation_list[i : i + self.llm_batch_size]
            )
            batch_annotation_ids = tokenized_prompts.input_ids

            batch_annotation_attn_mask = tokenized_prompts.attention_mask

            with self.lock:
                bad_response = True
                while bad_response:
                    responses = self.model.generate(
                        batch_annotation_ids.to(self.llm_gpus[0]),
                        attention_mask=batch_annotation_attn_mask.to(self.llm_gpus[0]),
                        return_dict_in_generate=True,
                        max_new_tokens=self.llm_max_new_tokens,
                        do_sample=True,
                        top_p=self.next_skill_top_p,
                        temperature=self.summary_temp,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_labels = self.process_hf_generation(responses)
                    bad_response = new_labels is False
            composite_skill_annotations.extend(new_labels)
        return composite_skill_annotations
