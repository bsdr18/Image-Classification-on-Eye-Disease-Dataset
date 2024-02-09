# Image-Classification-on-Eye-Disease-Dataset

## Project Title: Exploratory Analysis and Clustering Techniques for Customer Segmentation in Banking
### PROJECT OVERVIEW
The task is Eye Diseases Classification using a Convolutional Neural Network (CNN). The goal is to develop a model capable of classifying retinal images into different categories of eye diseases. 

### DATASET OVERVIEW
The [dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) used for this classification task comprises Normal, Diabetic Retinopathy, Cataract, and Glaucoma retinal images, with approximately 1000 images per class. These images are sourced from various datasets, including IDRiD, Oculur recognition, HRF, etc.

### APPROACH
#### Loading and Preprocessing
The dataset is loaded using the ImageFolder dataset from PyTorch, with images resized to (256, 256) pixels.

#### Convolutional Neural Network (CNN) Architecture
- The CNN model is designed with three convolutional layers, each followed by batch normalization, leaky ReLU activation, and max-pooling, to capture hierarchical features.
- Dense layers include dropout for regularization and two fully connected layers.
- The model is trained using the Adam optimizer and cross-entropy loss.

#### Training
- The training loop consists of iterating through batches of data, performing forward and backward passes, and optimizing the model's parameters.
- The model is trained for 7 epochs, and training/test losses are recorded for analysis.

#### Evaluation
- Model predictions on the test set are used to generate a confusion matrix, providing insights into classification performance.
- Additional metrics such as accuracy, precision, recall, and F1-score are calculated for comprehensive evaluation.

### MODEL VISUALIZATIONS
#### Model Architecture
- The CNN consists of three convolutional layers, progressively reducing spatial dimensions.
- Max-pooling layers help retain important features, and dense layers perform the final classification.

![image](https://github.com/bsdr18/Image-Classification-on-Eye-Disease-Dataset/assets/76464269/87c17f5f-6ecf-4e7c-8900-e7ce6b436f25)

#### Explanation:
1. **Convolutional Layers:** Extract hierarchical features.
2. **Batch Normalization:** Normalizes activations, aiding convergence.
3. **Leaky ReLU Activation:** Introduces non-linearity.
4. **Max-pooling:** Down-samples spatial dimensions.
5. **Flatten and Dropout:** Reduces overfitting.
6. **Fully Connected Layers:** Perform classification.
 
#### Sample Predictions
Displaying predictions on a few test set images provides a qualitative assessment of the model's performance.

![image](https://github.com/bsdr18/Image-Classification-on-Eye-Disease-Dataset/assets/76464269/241ccafc-da01-4e56-b94b-94d75707c6a2)

![image](https://github.com/bsdr18/Image-Classification-on-Eye-Disease-Dataset/assets/76464269/887e1ac0-d373-4f92-b7dd-6d7a503a91ae)

![image](https://github.com/bsdr18/Image-Classification-on-Eye-Disease-Dataset/assets/76464269/a8c3ce12-7383-48b1-8559-31caf83eb9d5)

### MODEL EVALUATION METRICS
Utilizing metrics like accuracy, precision, recall, and F1-score provides a quantitative measure of the model's effectiveness.

![image](https://github.com/bsdr18/Image-Classification-on-Eye-Disease-Dataset/assets/76464269/ba8134f0-8107-4c60-82ec-5db52c63c928)

