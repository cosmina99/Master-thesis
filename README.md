# Master-thesis

## Instance-level recognition for artworks: A comparative study between CNNs and Transformers used as backbones

### Abstract
Instance-level recognition (ILR) for artworks is the task of recognising the specific instance of an art object. In order to perform this task, the approach we have taken is to use pre-trained models to extract the image features and then feed these to a classifier. This research project focuses on comparing Transformer-based architectures, which have recently shown promising results in the computer vision field, with Convolutional Neural Network (CNN) ones as backbone (feature extracting network) for the task of artwork recognition. The different models will be evaluated in terms of accuracy using the accuracy score (ACC) and the Global Average Precision (GAP) measure, which are standard ILR metrics. In terms of data, The Met dataset is used, which is the newly-developed large-scale dataset for instance-level recognition in the artwork domain, containing artworks from the Metropolitan Museum of Art in New York. The results showed that using the approach that we have chosen the Vision Transformer models do not perform better than the CNN-based ones, obtaining lower scores across all evaluation metrics.

### How to use this repository?

This repository contains the code used for the thesis.

File "EDA.ipynb" contains the exploratory data analysis performed prior to conducting the experiments.

In order to run the code you will need to download:
 - the code from this repo
 
   git clone https://github.com/cosmina99/Master-thesis.git
   
 - the required packages from requirements.txt

   pip install -r requirements.txt
   
 - the Met Dataset which can be found at http://cmp.felk.cvut.cz/met - create a folder "images" in the "data" folder and download the dataset there

To replicate the results:

 1. Extracting the features
    - run the scripts in jobs/extract_descriptors for each model


 2. Evaluating the models
    - descriptors will be downloaded in data/descriptors
    - run the scripts in jobs/knn_evaluation for each model
