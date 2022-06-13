# Master-thesis

## Instance-level recognition for artworks: A comparative study between CNNs and Transformers used as backbones

### Abstract
Instance-level recognition (ILR) for artworks is the task of recognising the specific instance of an art object. In order to perform this task, the approach we have taken is to use pre-trained models to extract the image features and then feed these to a classifier. This research project focuses on comparing Transformer-based architectures, which have recently shown promising results in the computer vision field, with Convolutional Neural Network (CNN) ones as backbone (feature extracting network) for the task of artwork recognition. The different models will be evaluated in terms of accuracy using the accuracy score (ACC) and the Global Average Precision (GAP) measure, which are standard ILR metrics. In terms of data, The Met dataset is used, which is the newly-developed large-scale dataset for instance-level recognition in the artwork domain, containing artworks from the Metropolitan Museum of Art in New York. The results showed that using the approach that we have chosen the Vision Transformer models do not perform better than the CNN-based ones, obtaining lower scores across all evaluation metrics.

### How to use this repository?

This repository contains the code used for the thesis.

In order to run the code you will need:
 - the required packages from requirements.txt,
 - the Met Dataset which can be found at http://cmp.felk.cvut.cz/met. 

