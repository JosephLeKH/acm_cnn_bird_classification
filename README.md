# Bird Species Classification Project

## Overview
This project is part of the MLab team at Stanford ACM. It focuses on building a deep learning model to classify bird species using convolutional neural networks (CNNs). The model predicts the species of a bird from an image, addressing a real-world problem that benefits biodiversity monitoring, ecological conservation, and other fields. Classifying these 20 bird species is valuable for tasks such as automating ecological surveys, improving drone-based monitoring, and enhancing birdwatching applications with real-time identification capabilities.

Due to the size of the training dataset, the actual data cannot be included in this repository. Instead, the code and directions for processing and utilizing the data are provided here.

## Bird Species List
The 20 bird species classified in this project include:

- Baltimore Oriole
- Northern Cardinal
- Myna
- Barn Owl
- Ostrich
- Purple Finch
- Peacock
- Albatross
- Ruby-throated Hummingbird
- Hawfinch
- Robin
- Spoonbill
- Sora
- Scarlet Ibis
- Oyster Catcher
- Magpie Goose
- House Finch
- Green Jay
- Golden Eagle
- Frigate

## Technologies and Methods

### Framework
This project is implemented using PyTorch, a versatile deep learning framework.

### Model Architecture
The classification model, `BirdModel`, is built using the following components:
- **Three Convolutional Blocks**: Each block consists of convolutional layers with ReLU activation and max-pooling for feature extraction and spatial reduction.
- **Global Average Pooling**: Reduces the dimensionality of the feature maps while retaining their spatial information.
- **Fully Connected Layers**: Maps the extracted features to the output space of 20 bird classes.

### Techniques and Optimizations
1. **Dropout Regularization**: Dropout layers are incorporated to prevent overfitting by randomly deactivating neurons during training.
2. **Data Augmentation**: Training images are augmented with transformations such as random resizing, cropping, flipping, color jittering, rotation, and affine transformations to improve generalization.
3. **Optimizer**: The Adam optimizer is used with a learning rate of 0.0005 for efficient convergence.
4. **Learning Rate Scheduling**: Adjusts the learning rate dynamically based on validation performance to fine-tune the training process.

### Dataset
- **Compressed Format**: The dataset is stored in `bird_data.hdf5` for efficient loading (not included as mentioned above).
- **Mini Dataset**: A smaller subset (`small_data`) is provided for quick experimentation and testing.

### Training Process
1. Data preprocessing and augmentation.
2. Training the `BirdModel` with a batch size of 32 over 50 epochs.
3. Monitoring accuracy and loss to evaluate model performance.

### Evaluation
The `evaluate` function computes accuracy on validation data to ensure the model performs well on unseen images. By the end of training, the model achieves a validation accuracy of ~78.67%.

## Project Structure
### Files Provided
- `bird_data.csv`: Metadata for the dataset.
- `bird_data.hdf5`: Compressed dataset containing images and labels.
- `bird_data`: Directory with the full dataset.
- `small_data`: Mini dataset for initial testing.
- `utils.py`: Helper functions for data loading.
- `MLab_Onboarding_Project_Fall_2024.ipynb`: Starter code for the project.

## Instructions
1. Clone this repository.
2. Access the data files from the shared Google Drive folder (posted in the `#mlab-general` channel).
3. Use the starter code in `MLab_Onboarding_Project_Fall_2024.ipynb` to:
   - Implement a PyTorch dataset for `small_data`.
   - Train and evaluate the `BirdModel` on the dataset.
4. Experiment with different hyperparameters, data augmentation techniques, and optimization strategies.

## Credits
- MLab Team, Stanford ACM
- Joseph Le
- Jadelyn Tran

## Notes
This project demonstrates the practical application of deep learning techniques to image classification tasks. For further exploration, consider trying different architectures or datasets to enhance your understanding of CNNs.
