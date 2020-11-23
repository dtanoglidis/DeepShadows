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
- [Notebook descriptions](#Notebook-descriptions)
- [Requirements](#Requirements)
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

To run it you will nee the coordinates of the objects. These are provided in files within the [Datasets](/Datasets) folder and correspond to the columns `ra` and `dec` of the following:

- For the LSBGs, use: `random_LSBGs_all.csv`
- For the Artifacts of type 1, use: `random_negative_all_1.csv`
- For the Artifacts of type 2, use: `random_negative_all_2.csv`

For the diffrence between type 1 and type 2 artifacts, see our paper. The artifacts we used in the main body of the paper, for a binary classification, is of type 2. We added those of type 1 in Appendix A, where we considered a three-class problem. The word `random` in the names of the files means that we have randomly selected 20000 objects from each 
category. The detection of LSBGs (and artifacts, along the way) from the [Dark Energy Survey](https://www.darkenergysurvey.org/) is presented in [Tanoglidis et al. (2020)](https://arxiv.org/abs/2006.04294).

The above files provide also the `SourceExtractor`-derived features (half-light radii, magnitudes, surf. brightnesses etc.) we used when training SVM and random forest models.

In the main body of the paper we used a particular random split of the dataset into training, validation and test sets (refered to as the "Baseline" one).
Below we present the spatial distribution of the objects in these three sets in DES footprint (30000: training, 5000: validation, 5000: test).

<p float="center">
  <img src="/Images/Training.png" width="600" />
  <img src="/Images/Validation.png" width="600" /> 
  <img src="/Images/Test.png" width="600" />
</p>

We can see that the selection is indeed random and all three datasets follow a similar spatial distribution, with the same peaks and troughs.
If you are interested in reproducing this particular dataset split, you can find the coordinates of the objects in the files `Baseline_training.csv`, 
`Baseline_validation.csv` and `Baseline_test.csv`, again inside the [Datasets](Datasets) folder.

For the transfer learning task, cutouts of objects from the HSC survey have to be generated (for more info about LSBGs in HSC, and the source of our data, see [Greco et al. (2018)](https://arxiv.org/abs/1709.04474). 

To do so, you have to:

- Comment out the line that starts with `url_name = "http: ...` in [File_Creation.ipynb](/File_Creation.ipynb) and uncommment the line that is immediately below that.
- Use the coordinates you can find in `hsc_LSBGs.dat` and `hsc_artifacts` within the [Datasets](Datasets) folder (we gratefully thank Johnny Greco for providing us the list of artifacts).

---
### Notebook descriptions

In the above we described the notebook we used to generate the datasets. Let us give brief descriptions of the contents of the other notebooks.

- The main notebook for the present work is the [DeepShadows.ipynb](DeepShadows.ipynb). For the baseline split described above it runs the two machine learning (SVM and random
forests) and the *DeepShadows* CNN model. Produces results, calculates uncertainties using the Bootstrap method. It also performs the transfer learning task (with and without fine tuning). 

The image below summarizes the performance (ROC curves and AUC metrics) of the three models considered:

<img src="/Images/ROC_curves.png" width="400" />


- In [Uncertainty_quantification.ipynb](Uncertainty_quantification.ipynb) we consider the uncertainties arising from different random splits of the dataset into training-validation-test sets and also the impact of label noise (errors the labels).

<img src="/Images/Uncertainties.png" width="700" />

- In [GradCam_Visualizations.ipynb](GradCam_Visualizations.ipynb) we use the Gradient-weighted Class Activation Maps [Grad-CAM](https://arxiv.org/abs/1610.02391) technique 
to visualize the regions of the image that were the most important for the classification process. This allows us to better understand how *DeepShadows* decided how to classify an image (for example if there are very bright off-centered sources in an image, it is probable that it is going to be classified as depicting an artifact).

We generate examples of high-confidence (probabilty) correct and misclassifications, as shown below:

<img src="/Images/Grad_CAM.png" width="700" />

- Finally in [Three_class_classification.ipynb](Three_class_classification.ipynb), a three-class classification problem is considered with two classes of artifacts and one of LSBGs (this analysis is presented in the Appendix A of our paper).

---
### Requirements

We ran all of our experiments in Google Colab Pro, using GPUs and High-RAM mode.

We use `Python==3.6.9`. 
The other Python libraries our notebooks depend on are: 

`keras==2.4.0`\
`matplotlib==3.2.2`\
`numpy==1.18.5`\
`pandas==1.14`\
`scikit-learn==0.22.2`\
`scipy==1.4.1`\
`seaborn==0.11.0`\
`tensorflow==2.3.0`

For [GradCam_Visualizations.ipynb](GradCam_Visualizations.ipynb), specifically, we need the following:

`scipy==1.1.0`\
`tensorflow==1.15.2`\
`keras-vis==0.4.1`

---

