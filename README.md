# Dataset of cross-view image pairs & image-based geolocalization methods.

## Introduction

To evaluate the effect of cross-view image-based geo-localization method under real scenes, we build up our cross-view image dataset. Different with CVUSA or CVACT, our cross-view image dataset is **the first cross-view image dataset contains heading data** and are collected in real scenarios. It also including different settings such as spatial resolution and north alignment.

The download link of cross-view image pais:https://pan.baidu.com/s/1ftxWnywyZzPdLbcTFSMqXw. Extraction code: cvip

The siamese network model used for image retrieval will be uploaded soon.

## Data acquisition hardware
The ground-level panoramic images are collected by Ladybug5, Satellite images are collected from google earth and geo-locations of image are collected by GPS receiver of surveying level. The image collection are synchronized with GPS receiver at 1Hz update rate. All the equipment are deployed on Buick GL8.
 <div align="center">
 
 ![0001](https://user-images.githubusercontent.com/35421034/125156753-b18d3780-e199-11eb-8f60-021145e4d3c7.jpg)
Data acquisition hardware </div>

## Some Instance of cross-view image pairs

### The trajectory of image collection car on satellite image.
Our cross-view image pairs are collected in city, campus and rural areas. There are totally 377 pairs for each cross-view image setting.

 <div align="center">
 
 ![Trajectory](https://user-images.githubusercontent.com/35421034/125153396-ee4f3380-e185-11eb-9144-34bc10936254.jpg)
 One trajectory of our vehicle </div>

### The satellite image block
 Satellite images with different spatial resolution but same image width. Spatial resolution from 18 to 20.
 <div align="center">

 ![000180](https://user-images.githubusercontent.com/35421034/125153440-4423db80-e186-11eb-961f-3c2abd66a7bf.jpg)
 18: 0.46 meter\pixel </div>

<div align="center">

 ![000660](https://user-images.githubusercontent.com/35421034/125153456-63bb0400-e186-11eb-82ee-ddf9b7dee168.jpg)
 19: 0.23 meter\pixel </div>

<div align="center">

 ![000660](https://user-images.githubusercontent.com/35421034/125153468-78979780-e186-11eb-9f4c-ad4f320ee735.jpg)
 20: 0.11 meter\pixel </div>

### The ground-level panoramic images
