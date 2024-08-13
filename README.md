# Code for "Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance"
## Accepted to CoRL 2023 as an Oral Presentation (top 6.6%).


[[Project Website]](https://clvrai.github.io/boss/) [[Paper]](https://arxiv.org/abs/2310.10021) [[OpenReview]](https://openreview.net/forum?id=a0mFRgadGO)

[Jesse Zhang](https://jesbu1.github.io/)<sup>1</sup>, [Jiahui Zhang](https://jiahui-3205.github.io/)<sup>1</sup>, [Karl Pertsch](https://kpertsch.github.io/)<sup>1</sup>, [Ziyi Liu](https://taichi-pink.github.io/Ziyi-Liu/)<sup>1</sup>, [Xiang Ren](https://shanzhenren.github.io/)<sup>1</sup>, [Minsuk Chang](https://minsukchang.com/)<sup>2</sup>, [Shao-Hua Sun](https://shaohua0116.github.io/)<sup>3</sup>, [Joseph J. Lim](https://clvrai.com/web_lim/)<sup>4</sup>

<sup>1</sup>University of Southern California 
<sup>2</sup>Google AI
<sup>3</sup>National Taiwan University
<sup>4</sup>KAIST


<a href="docs/static/images/boss_overview.png">
<p align="center">
<img src="docs/static/images/boss_overview.png" width="800">
</p>
</img></a>

This is the official PyTorch implementation of CoRL 2023 paper "**Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance**"

# Running the code

This is the code for running simulated experiments on our ALFRED benchmarks.

## 1. Setting up the environment

### 1.1 Installing Dependencies
The environment can be installed either through pip or conda.

Pip install:
```
pip3 install -r requirements.txt
```
OR

Conda install:
```
conda env create -f environnment.yml
```

Then, you must pip install the boss package and the `spacy en_core_web_md` for language-related things:
```
conda activate boss
pip install -e .
python -m spacy download en_core_web_md
```

The [ALFRED](askforalfred.com) environment requires some additional dependencies -- installing may require sudo access. Read the [ALFRED README](alfred/README.md) for more details if this doesn't work:
```
conda activate boss
cd boss/alfred/scripts
sh install_deps.sh
```

### 1.2 WandB setup
All results will be written to [WandB](https://www.wandb.com/). Before running any of the commands below, create an account and then change the WandB entity and project name at the top of [`boss/utils/wandb_info.py`](boss/utils/wandb_info.py) to match your account and name for the project holding the runs for this repo.

### 1.3 BOSS Environment Variable
Add the location in which you git cloned BOSS to your `~/.bashrc`:
```
export BOSS=[BOSS_DOWNLOAD_LOCATION]
```

## 2. Downloading the data
### 2.1 Model Training Data
You need to pre-train models to run zero-shot or finetuning experiments. 
If you don't want to pre-train a model yourself, you can skip to step 3 as you don't need the pre-training dataset file. 

Download the ALFRED dataset here: [Google Drive Link](https://drive.google.com/file/d/1FDzyZb3TTyGRfYUnsRHjjvMBT6hIm90_).

You can use [Gdown](https://github.com/wkentaro/gdown) to directly download the dataset to your server/computer at the desired location (19GB download):
```
cd [BOSS_REPO_LOCATION]
mkdir data
cd data
pip3 install gdown
gdown 1ZgKDgG9Fv491GVb9rxIVNJpViPNKFWMF
```

Once the dataset is downloaded (`boss_offline_dataset.tar.gz`) simply untar it (40GB after extraction): 

``` 
mkdir data
cd data
tar -xvzf boss_offline_dataset.tar.gz
cd ..
```
### 2.2 ALFRED Evaluation Data
To run evals and fine-tuning/skill bootstrapping experiments, you must extract ALFRED evaluation data we have processed ([Google Drive Link](https://drive.google.com/file/d/1MHDrKSRmyag-DwipyLj-i-BbKU_dxbne)):

```
cd [BOSS_REPO_LOCATION]
cd boss/alfred/data
gdown 1MHDrKSRmyag-DwipyLj-i-BbKU_dxbne
tar -xvzf json_2.1.0_merge_goto.tar.gz
```

## 3. Setting up WandB
We log using WandB. First create a wandb account if you don't already have one [here](https://wandb.ai).
Then, run `wandb login` to login to your account on the machine.

Finally, fill in `WANDB_ENTITY_NAME, WANDB_PROJECT_NAME` in the file `utils/wandb_info.py` where `WANDB_ENTITY_NAME` refers to your wandb account name and `WANDB_PROJECT_NAME` is the name of the wandb project you want to log results to.


## 4. Pre-training a Model
You can either pre-train a model yourself or download a pre-trained checkpoint. Pre-trained model checkpoints can be found here: [Google Drive Link](https://drive.google.com/file/d/11wBTa_uiNDFmLvXz8MpvUz-u_LeiV-ED/view).

Otherwise, run the following command from the base BOSS repo location to train our model, BOSS:

```
python boss/pretrain.py --experiment_name [WANDB_EXP_NAME] --run_group [WANDB_RUN_GROUP] 
```

`--experiment_name` and `--run_group` are used to name the experiment and group of runs in WandB. Experiments in the same `run_group` will appear grouped together on wandB for easier comparison, but this command is completely optional.

All models are saved to `saved_models/` by default. You can add a `--save_dir` command to specify a different location.

## 5. Skill bootstrapping and fine-tuning

### 5.1 Downloading LLaMA:
We used facebook's LLAMA 13B open-source model for the paper. This repo now supports LLaMA-3-8B as LLAMA 13B is completely deprecated (no Meta download links) and also unsupported by latest huggingface versions.
To install, follow these shortened instructions:
- First, to the [llama website](https://llama.meta.com/llama-downloads/) to request access to meta llama.
- Then, request the permission on [huggingface](https://huggingface.co/meta-llama/Meta-Llama-3-8B).
- Finally, `huggingface-cli login` and use your [huggingface API key](https://huggingface.co/settings/tokens).

### 5.2 Skill Bootstrappping:

To run BOSS' skill bootstrapping, run the following code.
Make sure you have enough GPU memory to use LLAMA-8B.

If not, you can play with model sizes, which GPUs to expose, LLM batch size options, or adding 8-bit inference to `boss/models/large_language_model.py`, etc.

There are 4 total floorplans, with 10 evaluation tasks each for a total of 40 tasks. Choose which one to train on with `--which_floorplan [0-3]`:
```
python boss/run_skill_bootstrapping.py --which_floorplan [0-3] --experiment_name [WANDB_EXP_NAME] --run_group [WANDB_RUN_GROUP] --load_model_path [PRETRAINED_MODEL_LOCATION]/alfred_action_model_149.pth --llm_gpus [LLM_GPU]
```

We train on all 4 floorplans and aggregate results across all.

NOTE: Currently, due to possible LLM differences or other code changes that occurred in the massive refactoring effort during cleanup, the results don't fully reproduce the performance in the original paper. The code works and the agent learns new skills, but the performance isn't as good.
I am investigating this but have released the code so others can build upon BOSS in the meantime.

### 5.3 Oracle Fine-tuning:

To fine-tune an oracle model (given ground truth evaluation primitive skill guidance and rewards) on a specific floorplan, run:
```
python boss/run_skill_bootstrapping.py --gpus 2 --which_floorplan [0-3] --experiment_name [WANDB_EXP_NAME] --run_group [WANDB_RUN_GROUP] --load_model_path [PRETRAINED_MODEL_LOCATION]/alfred_action_model_149.pth --no_bootstrap True
```

### 5.4 SayCan
Run Saycan+P (+P = with our skill proposal mechanism, which works better than normal SayCan). `load_model_path` should be set to an offline pre-trained checkpoint (i.e., for regular SayCan pre-trained on the same offline data, point it to the pre-trained models linked earlier--same as the ones used for bootstrapping).

```
python boss/saycan_eval.py --which_floorplan [0-3] --experiment_name [WANDB_EXP_NAME] --run_group [WANDB_RUN_GROUP] --load_model_path [MODEL_PATH] --llm_gpus [GPU]
```

To run regular SayCan without our skill proposal mechanism, just add the flag `--skill_match_with_dataset False`

### 5.5 Interpreting Results:
In the paper, our results list IQMs of **oracle normalized returns** and **oracle normalized success rates**, averaged over all 4 floorplans.

For reference, our numbers we used for the paper: oracle return was: 1.21, oracle success rate was 11.7%. You should get approximately the same numbers if you re-run the oracle.

# Cite our work!

```
@inproceedings{
    zhang2023bootstrap,
    title={Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance},
    author={Jesse Zhang and Jiahui Zhang and Karl Pertsch and Ziyi Liu and Xiang Ren and Minsuk Chang and Shao-Hua Sun and Joseph J Lim},
    booktitle={7th Annual Conference on Robot Learning},
    year={2023},
    url={https://openreview.net/forum?id=a0mFRgadGO}
}
```
