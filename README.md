# AMDANet_ICCV 2025
The code of MRFS : Mutually Reinforcing Image Fusion and Segmentation.

## 💥 Motivation
In this work, we found that existing traditional scene perception methods are ineffective for garment perception in low-light environments.
Our DarkSeg model learns illumination-invariant structural representations from infrared images, enabling accurate detection and classification of garment and facilitating robotic grasping in low-light environments.
<p align="center">
<img src="./docs/pipline.jpg" width=85% height=85% class="center">
</p>


# Training:<br>
* Prepare training data & set the training parameters:<br>
  * Dataset fomula:
  * Dataset name<br>
    ------RGB folder<br>
    ------THE folder<br>
    ------Label folder<br>
    ------train.txt<br>
    ------val.txt<br>
    ------test.txt<br>
* Run ```CUDA_VISIBLE_DEVICES="GPU IDs" python -m torch.distributed.launch --nproc_per_node="GPU numbers you want to use" train.py```<br>
* Example ```CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py```<br>


# Test:<br>
* pretrained weights can be found at here: [FMB](https://drive.google.com/drive/folders/17LroKnuEWttvtcbuv-G-DXaXYdua52cN?usp=sharing), [MFNet](https://drive.google.com/drive/folders/1txDn-U04KEKA6gbSUjSsn-QqaN4nFw0Y?usp=sharing), [PST900](https://drive.google.com/drive/folders/1iEu3QZSV-q18u28X4cB7GK8UpLoRhXTX?usp=sharing), place the checkpoints under the corrspanding floder.
* Set the testing parameters:<br>
* Run ```CUDA_VISIBLE_DEVICES="GPU IDs" python eval.py -e="epoch"```<br>
* Example ```CUDA_VISIBLE_DEVICES=0 python eval.py -e=MRFS```<br>


## Recommended Requirements

```
 - [ ] python = 3.8
 - [ ] torch = 1.8.1+cu111
 - [ ] timm = 0.9.8
 - [ ] numpy = 1.24.4
 - [ ] scipy = 1.10.1
 - [ ] pillow = 10.1.0
 - [ ] tqdm = 4.66.1
 - [ ] tensorboardX = 2.6.2.2
 - [ ] opencv-python = 4.8.1.78
```

````
@inproceedings{zhang2024mrfs,
  title={MRFS: Mutually Reinforcing Image Fusion and Segmentation},
  author={Zhang, Hao and Zuo, Xuhui and Jiang, Jie and Guo, Chunchao and Ma, Jiayi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26974--26983},
  year={2024}
}
````

## Acknowlegement
_**DarkSeg**_ is built upon [SegFormer](https://github.com/NVlabs/SegFormer). We thank their authors for making the source code publicly available.
