
<div id="top" align="center">

<p align="center">
  <img src="https://ik.imagekit.io/StarBurger/SimScale/title_1080p.gif">
</p>

# **Learning to Drive via Real-World Simulation at Scale**

[![Paper](https://img.shields.io/badge/ArXiv-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.23369)
[![Home](https://img.shields.io/badge/project_page-5F259F?style=for-the-badge&logo=homepage&logoColor=white)](https://opendrivelab.com/SimScale/) 
[![Hugging Face](https://img.shields.io/badge/hugging_face-ffc107?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/OpenDriveLab/SimScale) 
[![License](https://img.shields.io/badge/Apache--2.0-2380C1?style=for-the-badge&)](https://github.com/OpenDriveLab/SimScale/blob/main/LICENSE) 


</div>


<div id="top" align="center">
<p align="center">
<img src="assets/teaser.png" >
</p>
</div>



> [Haochen Tian](https://github.com/hctian713), 
> [Tianyu Li](https://github.com/sephyli), 
> [Haochen Liu](https://georgeliu233.github.io/), 
> [Jiazhi Yang](https://github.com/YTEP-ZHI), 
> [Yihang Qiu](https://github.com/gihharwtw),
> [Guang Li](https://scholar.google.com/citations?user=McEfO8UAAAAJ&hl=en),
> [Junli Wang](https://openreview.net/profile?id=%7EJunli_Wang4),
> [Yinfeng Gao](https://scholar.google.com/citations?user=VTn0hqIAAAAJ&hl=en),
> [Zhang Zhang](https://scholar.google.com/citations?user=rnRNwEMAAAAJ&hl=en),
> [Liang Wang](https://scholar.google.com/citations?user=8kzzUboAAAAJ&hl=en),
> [Hangjun Ye](https://scholar.google.com/citations?user=68tXhe8AAAAJ&hl=en),
> [Tieniu Tan](https://scholar.google.com/citations?user=W-FGd_UAAAAJ&hl=en), 
> [Long Chen](https://long.ooo/), 
> [Hongyang Li](https://lihongyang.info/)
> 
>
> - 📧 Primary Contact: Haochen Tian (tianhaochen2023@ia.ac.cn)
> - 📜 Materials: 🌐 [𝕏](https://x.com/OpenDriveLab/status/1999507869633527845) | 📰 [Media](https://mp.weixin.qq.com/s/OGV3Xlb0bHSSSloG11qFJA) | 🗂️ [Slides TODO]()

---

## 🔥 Highlights 

- 🏗️ A scalable simulation pipepline that synthesizes diverse and high-fidelity reactive driving scenarios with pseudo-expert demonstrations. 
- 🚀 An effective sim-real co-training strategy that improves robustness and generalization synergistically across various end-to-end planners. 
- 🔬 A comprehensive recipe that reveals crucial insights into the underlying scaling properties of sim-real learning systems for end-to-end autonomy.


## 📢 News

- **`[2025/12/31]`** We released the data, and models **v1.0**. Happy New Year ! 🎄
- **`[2025/12/1]`** We released our [paper](https://arxiv.org/abs/2511.23369) on arXiv. 


## 📋 TODO List

- [ ] Sim-Real Co-training Code release (Jan. 2026).
- [x] Simulation Data release (Dec. 2025).
- [x] Checkpoints release (Dec. 2025).

---

## 📌 Table of Contents

<!-- - [📢 News](#news)
- [📋 TODO List](#todo-list) -->
- 🤗 [Model Zoo](#-model-zoo)
<!-- - 🎯 [Getting Started](#-getting-started) -->
- 📦 [Data Preparation](#-data-preparation)
  - [Download Dataset](#1-download-dataset)
  <!-- - [Set Up Configuration](#2-set-up-configuration) -->
<!-- - ⚙️ [Sim-Real Co-Training](#-sim-real-co-training-recipe)
  - [Co-Training with Pseudo-Expert](#co-training-with-pseudo-expert)
  - [Co-Training with Rewards Only](#co-training-with-rewards-only)
- 🔍 [Inference](#-inference)
  - [NAVSIM v2 navhard](#navsim-v2-navhard)
  - [NAVSIM v2 navtest](#navsim-v2-navtest) -->
- ⭐ [License and Citation](#-license-and-citation) 

## 🤗 Model Zoo

<table>
  <tr style="text-align: center;">
    <th rowspan="2">Model</th>
    <th rowspan="2">Backbone</th>
    <th rowspan="2">Sim-Real Config</th>
    <th colspan="2">NAVSIM v2 navhard</th>
    <th colspan="2">NAVSIM v2 navtest</t>

  </tr>

  <tr style="text-align: center;">
    <th>EPDMS</th>
    <th>CKPT</th>
    <th>EPDMS</th>
    <th>CKPT</th>
  </tr>

  <!-- LTF -->
  <tr>
    <td><a href="#">LTF</a></td>
    <td>ResNet34</td>
    <td>w/ pseudo-expert</td>
    <td>30.3 | +6.9</td>
    <td><a href="https://huggingface.co/datasets/OpenDriveLab/SimScale/blob/main/SimScale_ckpts/LTF/ltf_sim_navhard.ckpt">Link</a></td>
    <td>84.4 | +2.9</td>
    <td><a href="https://huggingface.co/datasets/OpenDriveLab/SimScale/blob/main/SimScale_ckpts/LTF/ltf_sim_navtest.ckpt">Link</a></td>
  </tr>

  <!-- DiffusionDrive -->
  <tr>
    <td><a href="#">DiffusionDrive</a></td>
    <td>ResNet34</td>
    <td>w/ pseudo-expert</td>
    <td>32.6 | +5.1</td>
    <td><a href="https://huggingface.co/datasets/OpenDriveLab/SimScale/blob/main/SimScale_ckpts/DiffusionDrive/diffusiondrive_sim_navhard.ckpt">Link</a></td>
    <td>85.9 | +1.7</td>
    <td><a href="https://huggingface.co/datasets/OpenDriveLab/SimScale/blob/main/SimScale_ckpts/DiffusionDrive/diffusiondrive_sim_navtest.ckpt">Link</a></td>
  </tr>

  <!-- GTRS-Dense block -->
  <tr>
    <td rowspan="4"><a href="#">GTRS-Dense</a></td>
    <td rowspan="2">ResNet34</td>
    <td>w/ pseudo-expert</td>
    <td>46.1 | +7.8</td>
    <td><a href="https://huggingface.co/datasets/OpenDriveLab/SimScale/blob/main/SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_expert_navhard.ckpt">Link</a></td>
    <td>84.0 | +1.7</td>
    <td><a href="https://huggingface.co/datasets/OpenDriveLab/SimScale/blob/main/SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_expert_navtest.ckpt">Link</a></td>
  </tr>

  <tr>
    <td>rewards only</td>
    <td>46.9 | +8.6</td>
    <td><a href="https://huggingface.co/datasets/OpenDriveLab/SimScale/blob/main/SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_reward_navhard.ckpt">Link</a></td>
    <td>84.6 | +2.3</td>
    <td><a href="https://huggingface.co/datasets/OpenDriveLab/SimScale/blob/main/SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_reward_navtest.ckpt">Link</a></td>
  </tr>

  <tr>
    <td rowspan="2">V2-99</td>
    <td>w/ pseudo-expert</td>
    <td>47.7 | +5.8</td>
    <td><a href="https://huggingface.co/datasets/OpenDriveLab/SimScale/blob/main/SimScale_ckpts/GTRS_Dense/gtrs_dense_vov_sim_expert_navhard.ckpt">Link</a></td>
    <td>84.5 | +0.5</td>
    <td><a href="https://huggingface.co/datasets/OpenDriveLab/SimScale/blob/main/SimScale_ckpts/GTRS_Dense/gtrs_dense_vov_sim_expert_navtest.ckpt">Link</a></td>
  </tr>

  <tr>
    <td>rewards only</td>
    <td>48.0 | +6.1</td>
    <td><a href="https://huggingface.co/datasets/OpenDriveLab/SimScale/blob/main/SimScale_ckpts/GTRS_Dense/gtrs_dense_vov_sim_reward_navhard.ckpt">Link</a></td>
    <td>84.8 | +0.8</td>
    <td><a href="https://huggingface.co/datasets/OpenDriveLab/SimScale/blob/main/SimScale_ckpts/GTRS_Dense/gtrs_dense_vov_sim_reward_navtest.ckpt">Link</a></td>
  </tr>
</table>

> [!NOTE]
> We fixed a minor error in the simulation process without changing the method, resulting in better performance than the numbers reported in the arXiv version. We will update the arXiv paper soon.



## 📦 Data Preparation

Our released simulation data is based on [nuPlan](https://www.nuscenes.org/nuplan) and [NAVSIM](https://github.com/autonomousvision/navsim). We recommend first preparing the real-world data by following the instructions in [Download NAVSIM](https://github.com/autonomousvision/navsim/blob/main/docs/install.md#2-download-the-dataset).

### 1. Download Dataset

Our simulation data format follows that of [OpenScene](https://github.com/OpenDriveLab/OpenScene/blob/main/docs/getting_started.md#download-data), with each clip/log has a fixed temporal horizon of 6 seconds (2s history + 4s future).

We provide [scripts](./docs/download.sh) for downloading the simulation data.



## ❤️ Acknowledgements

We acknowledge all the open-source contributors for the following projects to make this work possible:

- [NAVSIM](https://github.com/autonomousvision/navsim) | [MTGS](https://github.com/OpenDriveLab/MTGS) | [GTRS](https://github.com/NVlabs/GTRS) | [DiffusionDrive](https://github.com/hustvl/DiffusionDrive)

## ⭐ License and Citation

All content in this repository is under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
The released data is based on [nuPlan](https://www.nuscenes.org/nuplan) and is under the [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```bibtex
@article{tian2025simscale,
  title={SimScale: Learning to Drive via Real-World Simulation at Scale},
  author={Haochen Tian and Tianyu Li and Haochen Liu and Jiazhi Yang and Yihang Qiu and Guang Li and Junli Wang and Yinfeng Gao and Zhang Zhang and Liang Wang and Hangjun Ye and Tieniu Tan and Long Chen and Hongyang Li},
  journal={arXiv preprint arXiv:2511.23369},
  year={2025}
}
```


