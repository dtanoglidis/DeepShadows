# DeepShadows
Welcome to the code repository for the paper: "*DeepShadows*: Separating Low Surface Brightness Galaxies from Artifacts using Deep Learning"!\
Astronomy & Computing, *submitted*\
ArXiv: 


**Authors:**\
Dimitrios Tanoglidis\
Aleksandra Ćipricanović\
Alex Drlica-Wagner

### Abstract 

Searches for low-surface-brightness galaxies (LSBGs) in galaxy surveys are plagued by the presence of a large number of artifacts (objects blended in the diffuse light from stars and galaxies, Galactic cirrus, star-forming regions in the arms of spiral galaxies etc.) that have to be rejected through time consuming visual inspection. In future surveys, which are expected to collect petabytes of data and detect billions of objects, such an approach will not be feasible. We investigate the use of convolutional neural networks (CNNs) for the problem of separating LSBGs from artifacts in survey images. We take advantage of the fact that, for the first time, we have available a large number of labeled LSBGs (20k) and artifacts (20k) from the Dark Energy Survey, that we use to train, validate, and test a CNN model. That model, which we call *DeepShadows*, achieves a test accuracy of 92.0%, a significant improvement relative to feature-based machine learning models. We also study the ability to use transfer learning to adapt this model to classify objects from the deeper Hyper-Suprime-Cam survey, and we show that after the model is retrained on a very small sample from the new survey, it can reach an accuracy of 87.6%. These results demonstrate that CNNs can offer a very promising path in the quest for automating the LSBG classification.

---
### Table of contents

- [Architecture](#DeepShadows-Architecture)
- [Datasets](#Datasets)
- [Notebook descriptions](#Notebook_Descriptions)
---


### *DeepShadows* Architecture

![Architecture of DeepShadows](/Images/DeepShadows.png)

The *DeepShadows* architecture consists of 3 convolutional layers, each one followed by a max pooling layer, and then 2 fully connected layers (after flattening). 
Dropout with rate 0.4 is applied after each pooling layer. Weight regularization is applied in the convolutional and dense layers. 

--- 

### Datasets

In our paper we have considered two types of data: raw images and features derived from the imaging data using `SourceExtractor`.

The image cutouts, for LSBGs and artifacts alike, were generated using the [DESI Legacy Sky Viewer](https://www.legacysurvey.org/viewer). Because of their large size
we do not provide the image datasets; however, in [File_Creation.ipynb](/File_Creation.ipynb) notebook we provide the code we used to generate the cutouts and store the data into numpy arrays.

To run it you will nee the coordinates of the objects. These are provided in files within the [Datasets](/Datasets) folder.

<p float="center">
  <img src="/Images/Training.png" width="600" />
  <img src="/Images/Validation.png" width="600" /> 
  <img src="/Images/Test.png" width="600" />
</p>

---
### Notebook descriptions
