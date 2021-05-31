# HDRUNet [[Paper Link]](http://arxiv.org/abs/2105.13084)

### HDRUNet: Single Image HDR Reconstruction with Denoising and Dequantization
By Xiangyu Chen, Yihao Liu, Zhengwen Zhang, [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=zh-CN) and [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN)

<img src="https://raw.githubusercontent.com/chxy95/HDRUNet/master/images/introduction.jpg"/>

#### BibTeX

    @inproceedings{chen2021hdrunet,
      title={HDRUNet: Single Image HDR Reconstruction with Denoising and Dequantization},
      author={Chen, Xiangyu and Liu, Yihao and Zhang, Zhengwen and Qiao, Yu and Dong, Chao}, 
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
      year={2021}
    }

## Overview

<img src="https://raw.githubusercontent.com/chxy95/HDRUNet/master/images/Network_Structure.png" width="600"/>

## Getting Started

1. [Dataset](#dataset)
2. [Configuration](#configuration)
3. [How to test](#how-to-test)
4. [How to train](#how-to-train)
5. [Visualization](#visualization)

### Dataset
Register a codalab account and log in, then find the download link on this page:
```
https://competitions.codalab.org/competitions/28161#participate-get-data
```

### Configuration
```
pip install -r requirements.txt
```

### How to test

- Modify the `dataroot_LQ` and `pretrain_model_G` (you can also use the pretrained model which is provided in the `./pretrained_model`) in the `./codes/options/test/test_HDRUNet.yml`, then run
```
cd codes
python test.py -opt options/test/test_HDRUNet.yml
```
- The test results will be saved to `./results/testset_name`.

### How to train

### Visualization

Updating...
