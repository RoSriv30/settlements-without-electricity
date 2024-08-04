# Identifying Settlements Without Electricity

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://www.pytorchlightning.ai/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Weights & Biases](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg)](https://wandb.ai/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)


## Overview

### Challenge

- The need for this research stems from a critical development challenge: over 600 million people in Africa lack access to electricity, hindering economic development and contributing to persistent poverty and inequality. Traditional methods of identifying unelectrified regions are often resource-intensive, requiring extensive ground surveys and infrastructure assessments. Satellite imagery, with its ability to cover vast areas efficiently, presents an innovative solution to this problem. However, the complexity of satellite data, including variations in resolution, spectral bands, and temporal coverage across different satellite systems, poses significant analytical challenges. There is a pressing need for robust methodologies that can accurately process and interpret this data to identify non-electrified settlements. By addressing this need, our project aims to support electrification planning and implementation, contributing to the broader goal of universal access to electricity and the United Nations Sustainable Development Goals.
  
- The source of the original satellite dataset used in this project is the 2021 IEEE GRSS Data Fusion Contest: Track DSE. The dataset comprises 60 folders, each representing a distinct 64 km^2 geographical region. Within each folder, there exists a collection of 98 images, corresponding to various bands captured by four different satellites: Sentinel-1, Sentinel-2, Landsat 8, and the Suomi VIIRS night-time dataset. These images are standardized to a resolution of 800×800 pixels. Each band captures different aspects of the Earth's surface, ranging from intensity values for VV and VH polarization (Sentinel-1) to reflectance data across various spectral ranges (Sentinel-2, Landsat 8).
[Link To the Dataset](https://drive.google.com/file/d/1mVDV9NkmyfZbkSiD5lkskv_MwOuYxiog/view?usp=sharing).


### Preprocessing

- Functions were written to extract channels from images, including Sentinel-1, Sentinel-2, Landsat 8, and the Suomi VIIRS. Metadata regarding the image files were extracted and stored in data structures needed for the later stages.
  
- Techniques like Gaussian filtering, quantile clipping, minmax scaling, brightening, and gamma correction were used. Gaussian filtering was employed to reduce noise and smooth out irregularities in each of the bands/channels corresponding to each tile. Quantile clipping was used to remove outliers in images that were outside of a particular range. Min-max scaling was used to keep image data in the range of 0 to 1. Brightening was used to adjust image brightness using gain and bias parameters. Gamma correction was used to adjust image luminance.

### Dataset Creation

- Preprocessed and subtiled the satellite images into smaller images and associated subtile metadata to save.

- Wrote a custom PyTorch Dataset class using the subtiles generated in the previous step.

- Wrote custom PyTorch Transforms to augment the subtiled dataset.

- Built a PyTorch Lighting DataModule to encapsulate training and validation dataloaders utilizing the custom PyTorch Dataset created from the previous step.

- Explored dimensionality reduction techniques to identify patterns band combinations of the ESD subtiles, using: Principle Component Analysis (PCA), T-distributed stochastic neighbor embedding (TSNE), and Uniform Manifold Approximation and Projection (UMAP)

### Baseline Models Creation

- Segmentation CNN: This model represents a single feedforward CNN which works as a funnel to lower the resolution down to our segmentation's resolution. Each encoder does 2 convolutions, followed by a `MaxPool2d` operation which reduces the resolution of the image, at the same time, each encoder will increase the number of channels by 2, starting from `embedding_size`. This encoder is used 3 times, the first one lowers the resolution from WxH to W/5xH/5, the second one to W/25xH/25 and the third one to W/50xH/50. For example, if you are using a 200x200 subtile size, this will result in a prediction mask of size 4x4, which should correspond to the size of the image. A final decoder then takes the resulting channels and uses a 1x1 convolution to lower them down to the number of classes, creating logits for each class.

![SegmentationCNN](assets/segcnn.png)

- Transfer Resnet101: This model tries to leverage a pretrained model used for another task. While the task is completely different, many layers in the neural network are similar between related tasks, meaning that the gradient descent process is easier since most layers are pre-trained with similar solutions. Both tasks, however, use different kinds of images, requiring you to replace some layers to make them compatible. Namely, the first layer in an FCNResnet101 is meant to be used for RGB images, meaning that you will need to change this layer to allow for the number of satellite channels. This can easily be done in PyTorch though, by simply assigning the layer with a new layer, for example, `model.backbone.conv1 = nn.Conv2d(...)` will replace the layer with a new one of your choice. The other layer that needs to be changed is the last one, `model.classifier`, which needs to be changed so that the number of channels matches the number of prediction classes. Finally, an `AvgPool2d` layer must be added at the end, this is because FCNResnet is meant to be used in segmentation problems where the resolution of the images is the same as the resolution of the masks, however, as we don't have that luxury, the next best thing is to average a 50x50 patch in order to get the average prediction of all the pixels.

![FCNResnet101](assets/fcn.png)

- U-Net: This model uses what is called a "skip connection", these are inspired by the nonlinear nature of brains, and are generally good at helping models "remember" informatiion that might have been lost as the network gets longer. These are done by saving the partial outputs of the networks, known as residuals, and appending them later to later partial outputs of the network. In our case, we have the output of the `inc` layer, as the first residual, and each layer but the last one as the rest of the residuals. Each residual and current partial output are then fed to the `Decoder` layer, which performs a reverse convolution (`ConvTranspose2d`) on the partial output, concatenates it to the residual and then performs another convolution. At the end, we end up with an output of the same resolution as the input, so we must `MaxPool2d` in order to make it the same resolution as our target mask.

![UNet](assets/unet.png)

### Final Model Creation

- UNet++ was the experimental model used in this project. Unlike UNet, UNet++ incorporated different encoders based on the efficientnet-b7 architecture, diverging from the traditional approach of employing two convolutional layers. These encoders were equipped with pretrained weights derived from ImageNet, enabling the model to leverage pre-existing knowledge for feature extraction. By integrating efficientnet-b7 encoders, UNet++ aimed to enhance the feature representation process.

## Pipeline


### Data Download

- The dataset used in this project can be downloaded from [here](https://drive.google.com/file/d/1mVDV9NkmyfZbkSiD5lkskv_MwOuYxiog/view?usp=sharing).

### Data Preprocessing

- Functions were written to extract channels from images, including Sentinel-1, Sentinel-2, Landsat 8, and the Suomi VIIRS. Metadata regarding the image files were extracted and stored in data structures needed for the later stages.

- Techniques like Gaussian filtering, quantile clipping, minmax scaling, brightening, and gamma correction were used.

### Tile Subdivision

- The satellite images were preprocessed and subtiled into smaller images. Specifically, images were divided into 4 200x200 subtiles while the ground truth image was divided into 4 4x4 subtiles. Subtile metadata was also generated and stored for further use.

### Augmentation

- Custom PyTorch Transforms were implemented to augment the subtiled dataset.

### Custom Dataloaders

- A PyTorch Dataset class was created using the subtiles generated in the previous step. Additionally, a PyTorch Lightning DataModule was built to encapsulate training and validation dataloaders utilizing the custom PyTorch Dataset.

### Model Training

- Over a certain number of epochs consisting of a select batch size and other hyperparameters, all of the baseline models and the UNet++ model were trained.

### Results Analysis/Visualization

- Monitored model training using Weights and Biases. Jaccard Index, Intersection over Union, Accuracy, AUC: Area under the Receiver Operator Curve (ROC), and F1-score were analyzed for each model.

- Predictions were plotted for the validation dataset on each of the models.

## Getting Started

### Clone

```sh
  git clone https://github.com/cs175cv-w2024/final-project-kr-2
```

### Create Environment

- Linux/macOS
  
```sh
  python3 -m venv .my_venv
```
```sh
  source .my_venv/bin/activate
```

- Windows
  
```sh
  py -m venv .my_venv
```
```sh
  .my_venv\Scripts\activate
```


### Install Requirements

- Linux/macOS
  
```sh
  python3 -m pip install -r requirements.txt
```

- Windows
  
```sh
  py -m pip install -r requirements.txt
```

### Download Dataset

- Install dataset linked above and save as Raw under the Data folder


### Download Baseline Model Weights

- [Model Weights](https://drive.google.com/drive/u/1/folders/1aqXrdFTnVT8RRrqTp4hEwU6CBkSzHgQ6)

### Train (If weights not downloaded)

- Linux Command (Similar command in Windows)
```sh
  python3 -m scripts.train
```
- With Hyperparameter Sweeps after modifying sweeps.yml

 ```sh
  python3 -m train_sweeps.py --sweep_file=sweeps.yml
```


### Evaluate

- Linux Command (Similar command in Windows)
```sh
  python3 -m scripts.evaluate
```

