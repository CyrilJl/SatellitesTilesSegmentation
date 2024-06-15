# Satellite Tiles - Segmentation of Buildings

## Project Overview

This project focuses on the segmentation of buildings from satellite images. It comprises two main steps: building a dataset of satellite images with corresponding building segmentation masks, and developing a neural network to perform the segmentation using the U-Net architecture.

## Notebooks

### 1. Building the Dataset

The first notebook is dedicated to creating the dataset. It includes the following steps:

1. **Sampling Coordinates**: Random longitude and latitude coordinates are sampled across France, with sampling weighted by population density.
2. **Fetching Data**: For each coordinate, the corresponding Google Maps satellite tile is fetched. Building data for the area is retrieved from the OpenStreetMap API.
3. **Rasterizing Building Presence**: The presence of buildings is rasterized to match the satellite tile mesh.
4. **Generating Batches**: Data is generated in batches for different zoom levels. Each batch contains a specified number of samples, with each sample consisting of a 256x256 RGB satellite tile and a corresponding 256x256 target segmentation image. Each pixel in the target image is labeled 1 if it covers a building and 0 otherwise.

This notebook is fully launchable without any additional data, as it queries all necessary information dynamically.

### 2. Building the Segmentation Neural Network

The second notebook focuses on developing a neural network to perform building segmentation:

1. **Dataset Loading**: The dataset generated in the first notebook is loaded.
2. **Model Architecture**: A U-Net architecture is built from scratch. U-Net is a convolutional neural network architecture designed for biomedical image segmentation but is also effective for other segmentation tasks.
3. **Training**: The model is trained using the dataset, with the goal of accurately segmenting buildings in the satellite images.
4. **Evaluation**: The performance of the model is evaluated on a test set.