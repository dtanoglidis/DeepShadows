# DeepShadows
Code repository for the paper "DeepShadows: Separating Low Surface Brightness Galaxies from Artifacts using Deep Learning"

## Abstract 

Searches for low-surface-brightness galaxies (LSBGs) in galaxy surveys are plagued by the presence of a large number of artifacts (objects blended in the diffuse light from stars and galaxies, Galactic cirrus, star-forming regions in the arms of spiral galaxies etc.) that have to be rejected through time consuming visual inspection. In future surveys, which are expected to collect petabytes of data and detect billions of objects, such an approach will not be feasible. We investigate the use of convolutional neural networks (CNNs) for the problem of separating LSBGs from artifacts in survey images. We take advantage of the fact that, for the first time, we have available a large number of labeled LSBGs (20k) and artifacts (20k) from the Dark Energy Survey, that we use to train, validate, and test a CNN model. That model, which we call \textit{DeepShadows}, achieves a test accuracy of $91.8 \%$, a significant improvement relative to feature-based machine learning models. We also study the ability to use transfer learning to adapt this model to classify objects from the deeper Hyper-Suprime-Cam survey, and we show that after the model is retrained on a very small sample from the new survey, it can reach an accuracy of $86.4\%$. These results demonstrate that CNNs can offer a very promising path in the quest for automating the LSBG classification.


![Architecture of DeepShadows](DeepShadows.png)
