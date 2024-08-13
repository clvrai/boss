import os
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import pyxis as px
from boss.dataloaders.custom_lmdb_reader import CustomLMDBReader


class CustomDataset(Dataset):
    def __init__(
        self,
        path,
        data_type,
        max_skill_length,
    ):
        self.path = path
        self.data = self.load_pyxis()
        self.vocab_obj = torch.load(f"{os.environ['BOSS']}/boss/models/obj_cls.vocab")
        self.vocab_ann = torch.load(f"{os.environ['BOSS']}/boss/models/human.vocab")
        self.max_skill_length = max_skill_length
        self.include_list_dict = [
            "lang_low_action",
            "lang_object_ids",
            "lang_valid_interact",
            "lang_subgoals",
            "lang_combinations",
            "lang_ridx",
        ]

        self.include_all_dict = [
            "traj_resnet_feature",
            "skill_switch_point",
        ]

        pkl_name = "ET/ET_composite_skill_set_" + data_type

        pkl_name += ".pkl"
        if os.path.exists(pkl_name):
            with open(pkl_name, "rb") as f:
                (self.single_sample_trajectory_dict,) = pickle.load(f)
        else:
            self.create_single_sample_trajectory_dict()
            # in case another process takes over and saves it first
            if not os.path.exists(pkl_name):
                with open(pkl_name, "wb") as f:
                    pickle.dump(
                        (self.single_sample_trajectory_dict,),
                        f,
                    )

    def create_single_sample_trajectory_dict(self):
        print("generating pickle file")
        single_sample_trajectory_dict = {}
        total_samples = 0

        for i in tqdm(range(len(self.data))):
            num_skill = self.data[i]["lang_ridx"].shape[0]

            for j in range(num_skill):
                single_sample_trajectory_dict[total_samples] = (i, j)
                total_samples += 1

        self.single_sample_trajectory_dict = single_sample_trajectory_dict

    def load_pyxis(self):
        # df = px.Reader(self.path, lock=False)
        return CustomLMDBReader(self.path, lock=False)

    def __len__(self):
        return len(self.single_sample_trajectory_dict)

    def __getitem__(self, idx):
        i, j = self.single_sample_trajectory_dict[idx]
        return self.get_data_from_pyxis(i, j)

    def get_data_from_pyxis(self, i, j):
        # this will be overloaded by the subclass
        raise NotImplementedError
