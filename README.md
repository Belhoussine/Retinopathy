# Medical Image Processing: Retinal Fundus Disease Detection
![Retina Blood Vessel Segmentation](images/vessel.png)
## Goal:
### Identify potential diseases, such as diabetes, based on retinal fundus images.

## Approach:
### The disease identification process will be divided into 2 major parts:
### 1. Blood vessel extraction:
#### 1.1. Pre-processing: 
- [x] **Contrast Limited Histogram Equalization (CLAHE)**
    * *Increase the contrast of the image by equalizing the intensity histogram of fixed-sized windows (small matrices) within the image.*
- [x] **Median Filter**
    * *Non-linear digital filtering technique that removes "salt and pepper" noise from an image by replacing the grey level of a pixel by the median of the grey levels of surrounding pixels.*
#### 1.2. Segmentation: 
- [x] **Mean-C thresholding**
    * *Convolve fixed-sized windows with a N x N mean-filter kernel, take the difference between the result image and the original image, binarize it using a constant threshold C, then calculate the complement of the binarized image.*

#### 1.3. Post-processing: 
- [x] **Morphological cleaning**
    * *Apply morphological operations (erosion and dilation) on the given image*
- [x] **Pixel-island removal**
    * *Cleaning method that removes isolated pixels*

### 2. Classification of the extracted blood vessels:
#### 2.1. Convolutional Neural Network: 
- [x] **Define the Model's Architecture**
    * *4 Convolutional Layers with Max-pooling*
    * *1 Flatten layer*
    * *4 Dense Neuron layers*
- [x] **Train the model on the dataset**
    * *20 epochs of training produce a testing accuracy of ~80%*
#### 2.2. K-nearest Neighbours: 
- [x] **Compare different K values**
    * *Start from K = 1, calculate the accuracy of the model, then increase K and repeat the process.*
- [x] **Pick the best K value**
    * *K = 3 and K = 5 were revealed as giving the best results.*

## Challenges:
- Slow time for image processing using python and big images.
- The lack of publicly available datasets for retinal fundus images.
- The different shapes, sizes, and lightings of images of retinal fundus images.

## Future Work:
- Improve the image processing phase by conducting more trials in order to identify the best parameters for thresholding, filtering and cleaning.
- Improve the processing time of thessing phase by condue images.
- Identify more eye diseases using the classifier.
- Create a web platform to allow upload of retinal fundus images and process it.

## References
<a id="1">[1]</a> 
Dash, J., & Bhoi, N. (2017). A thresholding based technique to extract retinal blood vessels from fundus images. Future Computing and Informatics Journal, 2(2), 103-109. doi:10.1016/j.fcij.2017.10.001


**NOTES:** 
- All Python files should be run from within their parent folder in order to avoid path errors. 
- The VesselExtraction/main.py script extracts blood vessels from retinal fundus images and displays the different intermediate steps as well as the end result as images at the end.
- The "Dataset" folder is composed of two folders: "DR" and "Healthy", which contain Healthy and diseased retinal fundus images labelled as \[i\].png, such that \[i\] is a number (i.e: 37.png).
- The "ProcessedImages" folder follows the same structure as the "Dataset" folder, and it contains the processed images from the "Dataset" folder. The processing of those images is done by the "VesselExtraction/main.py" script.
