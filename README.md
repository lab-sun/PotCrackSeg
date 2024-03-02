# PotCrackSeg
The official pytorch implementation of **Segmentation of Road Negative Obstacles Based on Dual Semantic-feature Complementary Fusion for Autonomous Driving**. ([TIV](https://ieeexplore.ieee.org/document/10114585))


We test our code in Python 3.8, CUDA 11.3, cuDNN 8, and PyTorch 1.12.1. We provide `Dockerfile` to build the docker image we used. You can modify the `Dockerfile` as you want.  
<div align=center>
<img src="docs/overall.png" width="900px"/>
</div>

# Demo

The accompanied video can be found at: 
<div align=center>
<a href="https://www.youtube.com/watch?v=yoW52JeTDR8&t=7s"><img src="docs/qualitativeresultsgray5.png" width="70%" height="70%" />
</div>

# Introduction

PotCrackSeg with an RGB-Depth fusion network with a dual semantic-feature complementary fusion module for the segmentation of potholes and cracks in traffic scenes.

# Dataset

The **NPO++** dataset is upgraded from the existing [**NPO**](https://pan.baidu.com/s/1-LuHyKXEuJ0oLMe1PHtq0Q?pwd=drno) dataset by re-labeling potholes and cracks. You can downloaded **NPO++** dataset from [here](https://pan.baidu.com/s/1608EIKo-be63XE3-7UYcIQ?pwd=uxks)

# Pretrained weights
The pretrained weight of PotCrackSeg can be downloaded from [here](https://pan.baidu.com/s/18xGs1Jp1xbSekBjJVEh9Pg?pwd=ynva).

# Usage
* Clone this repo
```
$ git clone https://github.com/lab-sun/PotCrackSeg.git
```
* Build docker image
```
$ cd ~/PotCrackSeg
$ docker build -t docker_image_PotCrackSeg .
```
* Download the dataset
```
$ (You should be in the PotCrackSeg folder)
$ mkdir ./NPO++
$ cd ./NPO++
$ (download our preprocessed NPO++.zip in this folder)
$ unzip -d . NPO++.zip
```
* To reproduce our results, you need to download our pretrained weights. 
```
$ (You should be in the PotCrackSeg folder)
$ mkdir ./weights_backup
$ cd ./weights_backup
$ (download our preprocessed weights_backup.zip in this folder)
$ unzip -d . weights_backup.zip
$ docker run -it --shm-size 8G -p 1234:6006 --name docker_container_potcrackseg --gpus all -v ~/PotCrackSeg:/workspace docker_image_potcrackseg
$ (currently, you should be in the docker)
$ cd /workspace
$ (To reproduce the results)
$ python3 run_demo.py   
```
The results will be saved in the `./runs` folder. The default results are PotCrackSeg-4B. If you want to reproduce the results of PotCrackSeg-2B, you can modify the *PotCrackSeg-4B* to *PotCrackSeg-2B* in run_demo.py

* To train PotCrackSeg. 
```
$ (You should be in the PotCrackSeg folder)
$ docker run -it --shm-size 8G -p 1234:6006 --name docker_container_potcrackseg --gpus all -v ~/PotCrackSeg:/workspace docker_image_potcrackseg
$ (currently, you should be in the docker)
$ cd /workspace
$ python3 train.py
```

* To see the training process
```
$ (fire up another terminal)
$ docker exec -it docker_container_potcrackseg /bin/bash
$ cd /workspace
$ tensorboard --bind_all --logdir=./runs/tensorboard_log/
$ (fire up your favorite browser with http://localhost:1234, you will see the tensorboard)
```
The results will be saved in the `./runs` folder.
Note: Please change the smoothing factor in the Tensorboard webpage to `0.999`, otherwise, you may not find the patterns from the noisy plots. If you have the error `docker: Error response from daemon: could not select device driver`, please first install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your computer!

# Citation
If you use PotCrackSeg in your academic work, please cite:
```

```

# Acknowledgement
Some of the codes are borrowed from [IGFNet](https://github.com/lab-sun/IGFNet).
