# Model based via State Entropy Maximization

MBRL has a problem for distribution shift problem caused by the bias of the learned model.  
Our Idea is to alleviate this distribution mismatch using a pretrained policy which is trained for maximizing state entropy.  
To run the code follow the instructions bellow.

---

## Requirement

```bash
conda env create -f conda_env.yml


## Training
--------

### 1. Pretrain

```bash
python pretrain.py agent=diayn domain=walker

### 2. Finetuning

```bash
mb_finetune.py agent=apt task=walker_run snapshot_ts=1000000 obs_type=states reward_free=false k=15

