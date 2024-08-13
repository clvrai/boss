# This is a class for Skills.
from boss.utils.utils import process_skill_strings, AttrDict
import torch


class Skill:
    def __init__(self, skills: list, embedding_model=None, embeddings=None) -> None:
        assert embedding_model is not None or embeddings is not None
        self.primitive_skills = []
        self.primitive_language_embeddings = []
        self.primitive_instructions_to_compose = []
        for i, skill in enumerate(skills):
            if isinstance(skill, dict):
                self.primitive_skills.append(skill)
                processed_skill_string = process_skill_strings([skill["annotations"]])
                if embedding_model is not None:
                    self.primitive_language_embeddings.append(
                        embedding_model.encode(
                            processed_skill_string, convert_to_tensor=True
                        ).cpu()
                    )
                else:
                    if embeddings[i].dtype is torch.long:  # transformer
                        first_index_of_zero_pad = (
                            torch.where(embeddings[i] == 0)[0][0]
                            if 0 in embeddings[i]
                            else len(embeddings[i])
                        )
                        self.primitive_language_embeddings.append(
                            embeddings[i][:first_index_of_zero_pad].unsqueeze(0).cpu()
                        )
                    else:
                        self.primitive_language_embeddings.append(
                            embeddings[i].unsqueeze(0).cpu()
                        )
                self.primitive_instructions_to_compose.append(processed_skill_string[0])

            elif isinstance(skill, Skill):
                self.primitive_skills.extend(skill.primitive_skills)
                self.primitive_language_embeddings.extend(
                    skill.primitive_language_embeddings
                )
                self.primitive_instructions_to_compose.extend(
                    skill.get_precomposed_language_instructions()
                )
                # this code originally used composite instructions to compose new ones, but now we use the primitive instructions
                # self.primitive_instructions_to_compose.append(
                #    skill.composite_language_instruction
                # )
        # this is a primitive skill if this is true
        if len(self.primitive_skills) == 1:
            self.composite_language_instruction = (
                self.primitive_instructions_to_compose[0]
            )
            self.composite_language_embedding = self.primitive_language_embeddings[0]
        else:
            self.composite_language_instruction = (
                None  # natural language instruction string
            )
            self.composite_language_embedding = None
        self.llm_prob = None  # likelihood assigned by the LLM
        self.primitive_skill_keys = tuple(
            [
                (skill["discrete_action"]["action"], *skill["discrete_action"]["args"])
                for skill in self.primitive_skills
            ]
        )

    @property
    def num_skills(self):
        return len(self.primitive_skills)

    def get_precomposed_language_instructions(self):
        return self.primitive_instructions_to_compose

    def set_label(self, language_instruction, language_embedding, llm_prob):
        self.composite_language_instruction = language_instruction
        self.composite_language_embedding = language_embedding
        self.llm_prob = llm_prob

    def get_skill_at_index(self, i):
        return AttrDict(
            skill_info=self.primitive_skills[i],
            language_embedding=self.primitive_language_embeddings[i],
        )

    @property
    def is_composite(self):
        return self.num_skills > 1

    def __hash__(self):
        return hash(self.__key())

    def __key(self):
        return self.primitive_skill_keys

    def __eq__(self, other):
        if isinstance(other, Skill):
            return self.__key() == other.__key()
        return NotImplemented
