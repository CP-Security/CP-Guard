# CP-Guard: Malicious Agent Detection and Defense in Collaborative Bird's Eye View Perception

[![AAAI'25 Oral](https://img.shields.io/badge/AAAI'25-Oral-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/34486)
[![Python 3.7](https://img.shields.io/badge/Python-3.7-green.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org/)

## 👥 Authors

| Name | Profile |
|------|---------|
| Senkang Hu* | [Google Scholar](https://scholar.google.com/citations?user=rtPVwT8AAAAJ&hl=zh-CN&oi=ao) |
| Yihang Tao* | [Google Scholar](https://scholar.google.com/citations?user=YopoapwAAAAJ&hl=zh-CN&oi=ao) |
| Guowen Xu | [Google Scholar](https://scholar.google.com/citations?user=MDKdG80AAAAJ&hl=zh-CN&oi=ao) |
| Yiqing Deng | [Google Scholar](https://scholar.google.com/citations?user=EdRi_MEAAAAJ&hl=zh-CN) |
| Xianhao Chen | [Google Scholar](https://scholar.google.com/citations?user=TnjiGooAAAAJ&hl=zh-CN&oi=sra) |
| Yuguang Fang | [Google Scholar](https://scholar.google.com/citations?user=dJgRKmwAAAAJ&hl=zh-CN&oi=ao) |
| Sam Kwong | [Google Scholar](https://scholar.google.com/citations?user=_PVI6EAAAAAJ&hl=zh-CN&oi=sra) |

## 📝 Abstract

Collaborative Perception (CP) has shown a promising technique for autonomous driving, where multiple connected and autonomous vehicles (CAVs) share their perception information to enhance the overall perception performance and expand the perception range. However, in CP, ego CAV needs to receive messages from the collaborators, which makes it easy to be attacked by malicious agents. For example, a malicious agent can send harmful information to the ego CAV to mislead it. 
To address this critical issue, we propose a novel method, CP-Guard, a tailored defense mechanism for CP that can be deployed by each agent to accurately detect and eliminate malicious agents in its collaboration network. Our key idea is that CP will lead to a consensus rather than a conflict against the ego CAV's perception results. Based on this idea:

- [x] We develop a probability-agnostic sample consensus (PASAC) method that can effectively sample a subset of the collaborators and verify the consensus without prior probabilities of malicious agents
- [x] We design a collaborative consistency loss (CCLoss) to calculate the discrepancy between the ego CAV and the collaborators
- [x] We conduct extensive experiments in collaborative bird's eye view (BEV) tasks

## 🛠️ Installation

### System Requirements

| Requirement | Version/Specification |
|-------------|---------------------|
| OS | Linux (tested on Ubuntu 22.04) |
| Python | 3.7 |
| Package Manager | Miniconda |
| Deep Learning Framework | PyTorch |
| CUDA | 11.7 |

### Setup Steps

1. **Create and Activate Conda Environment**
```bash
cd coperception
conda env create -f env.yml
conda activate coperception
```

2. **Install CUDA Dependencies**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. **Install CoPerception Library**
```bash
pip install -e .
```

## 📊 Dataset Preparation

1. Download and unzip the [parsed segmentation dataset](https://drive.google.com/file/d/1b5yTc9ujy1pEUxO0RuwVgw9Hy6iyMLDN/view?usp=sharing) of V2X-Sim 2.0

2. Link the test split of V2X-Sim dataset in the default value of argument "**data**":
```bash
/{Your_location}/V2X-Sim-seg/test
```

### Dataset Structure
```
test
├──agent_0
├──agent_1
├──agent_2
├──agent_3
├──agent_4
├──agent_5
      ├──5_0
      ├──0.npy
      ...
```

### Model Checkpoint
1. Download [pre-trained weights](https://drive.google.com/drive/folders/17wOB5ihebyRf263lf6-B5IRGy2I_Cyp-)
2. Save them in `CP-Guard/coperception/ckpt/meanfusion` folder

## 🚀 Running CP-Guard

### Basic Usage

```bash
cd coperception/tools/seg/
python cp_guard.py [options]
```

> **Note**: Due to data syncing issues, CP-Guard cannot be performed under multi-GPU environment.
> For multiple GPUs, use: `CUDA_VISIBLE_DEVICES=0 python cp_guard.py [options]`

### Parameters

#### General Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `-d, --data` | Path to preprocessed sparse BEV training data | `./datasets/V2X-Sim-seg/test` |
| `--batch` | Number of scene | 1 |
| `--th` | CCLoss threshold | 0.08 |
| `--com` | Communication method (disco/when2com/v2v/sum/mean/max/cat/agent) | `v2v` |
| `--num_agent` | Total number of agents | 6 |

#### Adversarial Attack Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--pert_alpha` | Scale of the perturbation | 0.1 |
| `--adv_method` | Attack method (pgd/bim/cw-l2/fgsm) | `pgd` |
| `--eps` | Epsilon of adv attack | 0.5 |
| `--adv_iter` | Adv iterations of computing perturbation | 15 |

#### Scene Settings
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--scene_id` | Target evaluation scene (Scene 8, 96, 97 has 6 agents) | [8] |
| `--sample_id` | Target evaluation sample | None |

#### CP-Guard Specific Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--cpguard` | Mode (upperbound/lowerbound/no_defense/pasac/robosac) | `no_defense` |
| `--ego_agent` | ID of ego agent | 1 |
| `--ego_loss_only` | Only use ego loss to compute adv perturbation | False |
| `--number_of_attackers` | Number of malicious attackers | 1 |
| `--fix_attackers` | Fix attackers across frames | False |

#### ROBOSAC Specific Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--robosac_k` | Consensus set size | None |
| `--step_budget` | Sampling budget per frame | 5 |
| `--use_history_frame` | Use history frame for consensus | False |

## 🙏 Acknowledgment

*CP-Guard* is modified from:
- [coperception](https://github.com/coperception/coperception) library
- [robosac](https://github.com/coperception/ROBOSAC) library

## 📚 Citation

If you find this project useful in your research, please cite:
```bibtex
@article{CP_Guard_2025, 
    title={CP-Guard: Malicious Agent Detection and Defense in Collaborative Bird's Eye View Perception}, 
    volume={39},
    number={22},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Hu, Senkang and Tao, Yihang and Xu, Guowen and Deng, Yiqin and Chen, Xianhao and Fang, Yuguang and Kwong, Sam}, 
    year={2025},
    month={Apr.},
    pages={23203-23211} 
}
```
