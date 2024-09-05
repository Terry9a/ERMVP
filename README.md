# ERMVP

The official implementation of CVPR2024 paper "ERMVP: Communication-Efficient and Collaboration-Robust Multi-Vehicle Perception in Challenging Environments".
![ERMVP_Overview](https://github.com/Terry9a/ERMVP/blob/main/image.png)
> [**ERMVP: Communication-Efficient and Collaboration-Robust Multi-Vehicle Perception in Challenging Environments**](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_ERMVP_Communication-Efficient_and_Collaboration-Robust_Multi-Vehicle_Perception_in_Challenging_Environments_CVPR_2024_paper.html),            
>  Jingyu Zhang, Kun Yang, Yilei Wang, Hanqi Wang, Peng Sun\*, Liang Song\*<br>
>  Accepted by CVPR2024 


## Abstract

Collaborative perception enhances perception performance by enabling autonomous vehicles to exchange complementary information. Despite its potential to revolutionize the mobile industry, challenges in various environments, such as communication bandwidth limitations, localization errors and information aggregation inefficiencies, hinder its implementation in practical applications. In this work, we propose ERMVP, a communication-Efficient and collaboration-Robust Multi-Vehicle Perception method in challenging environments. Specifically, ERMVP has three distinct strengths: i) It utilizes the hierarchical feature sampling strategy to abstract a representative set of feature vectors, using less communication overhead for efficient communication; ii) It employs the sparse consensus features to execute precise spatial location calibrations, effectively mitigating the implications of vehicle localization errors; iii) A pioneering feature fusion and interaction paradigm is introduced to integrate holistic spatial semantics among different vehicles and data sources. To thoroughly validate our method, we conduct extensive experiments on real-world and simulated datasets. The results demonstrate that the proposed ERMVP is significantly superior to the state-of-the-art collaborative perception methods.

## Installation

```bash

# Create a conda environment
conda create -n ermvp python=3.7 -y

conda activate ermvp 

# install pytorch
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# install spconv 
pip install spconv-cu113

# install requirements
pip install -r requirements.txt

# Install bbx nms calculation cuda version
python opencood/utils/setup.py build_ext --inplace

# install opencood into the environment
python setup.py develop

```

## Available module codes
  - [x] FMS
  - [ ] FSC
  - [x] AFF
  - [ ] AEI

## Note
You can refer to our previous work [Feaco](https://github.com/jmgu0212/FeaCo) on the basic implementation of the  feature spatial calibration (FSC) module.

## Test with pretrained model
We provide the  pretrained model of ERMVP (all communication), first download the model file from [google url](https://drive.google.com/drive/folders/1cO20HIDAET0oAIvzq2Wc3AkyB5_x1fag?usp=sharing) and
then put it under opencood/logs/ermvp. Change the `validate_path` in `opencood/logs/ermvp/config.yaml` as `/data/v2v4real/test`.

Eventually, run the following command to perform test:
```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER}
```
Arguments Explanation:
- `model_dir`: the path of the checkpoints, e.g. 'opencood/logs/ermvp' for ERMVP testing.

## Train your model
Ermvp uses yaml file to configure all the parameters for training. You can change the parameters topk_ratio and cluster_sample_ratio in the yaml file to achieve sparse communication. To train your own model
from scratch or a continued checkpoint, run the following commands:

```python
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER} --half]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/point_pillar_ermvp.yaml` for ERMVP training.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.
- `half`(optional): if specified, hybrid-precision training will be used to save memory occupation.

## Citation
 If you are using our ERMVP for your research, please cite the following paper:
  ```bibtex
@InProceedings{Zhang_2024_CVPR,
    author    = {Zhang, Jingyu and Yang, Kun and Wang, Yilei and Wang, Hanqi and Sun, Peng and Song, Liang},
    title     = {ERMVP: Communication-Efficient and Collaboration-Robust Multi-Vehicle Perception in Challenging Environments},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {12575-12584}
}
```
## Acknowledgment
Many thanks to the high-quality dataset and codebase, including [V2XSet](https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6), [OPV2V](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu), [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and  [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm.git).
