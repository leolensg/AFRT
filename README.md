# AFRT
Official PyTorch implementation of the paper **Adaptive Feature Recalibration Transformer for Enhancing Few-Shot Image Classification**.


**AFRT** is a novel approach for few-shot classification. During the pre-training phase, the feature encoder learns semantic information beyond image labels and contextual relationships of local regions from  masked image modeling (MIM). During the meta-finetuning phase, our method comprises a task-driven salient region refinement module (TSRR) and a bidirectional interactive feature calibration module (BIFC). TSRR establishes local semantic relationships within the support set and filters out regions that contribute more to inference, weakening the expression of irrelevant regions. BIFC facilitates bidirectional interaction between local regions of support class features and query instance features, further focusing on more subtle and discriminative shared features. 

## Prerequisites
Please install [PyTorch](https://pytorch.org/)  as appropriate for your system. This codebase has been developed with 
python 3.8.12, PyTorch 1.11.0, CUDA 11.3 and torchvision 0.12.0 with the use of an 
[anaconda](https://docs.conda.io/en/latest/miniconda.html) environment.

To create an appropriate conda environment (after you have successfully installed conda), run the following command:
```
conda create --name afrt --file requirements.txt
```
Activate your environment via
```
conda activate afrt
```

## Datasets
### <i>mini</i> ImageNet
To download the miniImageNet dataset, you can use the script [download_miniimagenet.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_miniimagenet.sh) in the `datasets` folder.

The miniImageNet dataset (Vinyals et al., 2016; Ravi & Larochelle, 2017) consists of a specific 100 class subset of Imagenet (Russakovsky et al., 2015) with 600 images
for each class. The data is split into 64 training, 16 validation and 20 test classes.

### <i>tiered</i> ImageNet
To download the tieredImageNet dataset, you can use the script [download_tieredimagenet.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_tieredimagenet.sh) in the `datasets` folder.

Similar to the previous dataset, the tieredImageNet (Ren et al., 2018) is a subset of classes
selected form the bigger ImageNet dataset (Russakovsky et al., 2015), however with a substantially larger set of classes and
different structure in mind. It comprises a selection of 34 super-classes with a total of 608 categories,
totalling in 779,165 images that are split into 20,6 and 8 super-classes to achieve better separation
between training, validation and testing, respectively.

### CIFAR-FS
To download the CIFAR-FS dataset (Bertinetto et al., 2018), you can use the script [download_cifar_fs.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_cifar_fs.sh) in the `datasets` folder.

The CIFAR-FS dataset contains the 100 categories with 600 images per category
from the CIFAR100 dataset (Krizhevsky et al., 2009) which are split into 64 training, 16 validation and 20 test classes.

### FC100
To download the FC-100 dataset, you can use the script [download_fc100.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_fc100.sh) in the `datasets` folder.

The FC-100 dataset (Oreshkin et al., 2018) is also derived from CIFAR100 (Krizhevsky et al., 2009) but follows a splitting strategy
similar to tieredImageNet to increase difficulty through higher separation, resulting in 60 training, 20
validation and 20 test classes.

## Training AFRT
For a glimpse at the documentation of all arguments available for training, please check for **self-supervised training**:
```
python train_selfsup_pretrain.py --help
```
and for **meta fine-tuning**:
```
python train_metatrain.py --help
```


### Self-Supervised Pretraining via Masked Image Modelling
To start the self-supervised pre-training procedure using a **ViT-small** architecture on one node with 4 GPUs using a total **batch size** of **512** for **1600 epochs** on **miniImageNet**, run:
```
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345
export WORLD_SIZE=4
echo "Using Master_Addr $MASTER_ADDR:$MASTER_PORT to synchronise, and a world size of $WORLD_SIZE."

echo "Start training..."
torchrun --nproc_per_node 4 train_selfsup_pretrain.py --use_fp16 True --arch vit_small --epochs 1600 --batch_size_per_gpu 128 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --dataset miniimagenet --image_size 224 --data_path <path-to-dataset> --saveckp_freq 50 --shared_head true --out_dim 8192  --local_crops_number 10 --global_crops_scale 0.25 1 --local_crops_scale 0.05 0.25 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2
echo "Finished training!"
```

### Meta Fine-tuning 
To start the meta fine-tuning procedure using a previously pretrained **ViT-small** architecture using 
one GPU for **100 epochs** on the **miniImageNet** training dataset using **20 steps** to adapt the **region scores** at inference time, run:
- For a 5-way 5-shot scenario:
```
python train_metatrain.py --num_epochs 100 --data_path D:/project_file/python/dataset --arch vit_small --n_way 5 --k_shot 5 --optim_steps_online 20  --chkpt_epoch 1600 --mdl_checkpoint_path initialization/miniimagenet
```
- For a 5-way 1-shot scenario:
```
python train_metatrain.py --num_epochs 100 --data_path D:/project_file/python/dataset/miniimagenet --arch vit_small --n_way 5 --k_shot 1 --optim_steps_online 20  --chkpt_epoch 1600 --mdl_checkpoint_path initialization/miniimagenet
```

## Evaluating
To evaluate a meta-trained **ViT-small** architecture on the **miniImageNet** test dataset, run:
- For a 5-way 5-shot scenario:
```
python eval_.py --data_path <path-to-dataset> --arch vit_small --n_way 5 --k_shot 5 --trained_model_type metaft --optim_steps_online 20  --mdl_checkpoint_path <path-to-checkpoint-of-metaft-model>
```
- For a 5-way 1-shot scenario:
```
python eval_.py --data_path <path-to-dataset> --arch vit_small --n_way 5 --k_shot 1 --trained_model_type metaft --optim_steps_online 20  --mdl_checkpoint_path <path-to-checkpoint-of-metaft-model>
```
