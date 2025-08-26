# Geographical Landmark Recognition 🗺️

A deep learning project to classify famous global landmarks from images using transfer learning with TensorFlow and Keras.

## Overview

This project implements a computer vision model capable of identifying geographical landmarks from around the world. The model was built using a deep convolutional neural network (CNN) and leverages transfer learning from the `EfficientNetB0` architecture to achieve high accuracy on a large-scale image dataset from Kaggle.

## 🛠️ Tech Stack

* **Python 3**
* **TensorFlow & Keras:** For building and training the deep learning model.
* **Ultralytics YOLOv8:** Used in the object detection phase of the project for real-time mask detection.
* **Pandas & NumPy:** For data manipulation and preprocessing.
* **Matplotlib & Seaborn:** For data visualization and plotting results.
* **Scikit-learn:** For data splitting and label encoding.
* **Google Colab:** For training the model with free GPU acceleration.
* **Git & GitHub:** For version control and project hosting.


## Dataset

The model was trained on a custom subset of the **Google Landmarks Dataset v2** from Kaggle. The subset used for this project contains thousands of images across 20+ distinct landmark classes.

You can find the original dataset [here](https://www.kaggle.com/c/landmark-recognition-2021/data).

## 🧠 Model Architecture

The core of this project is a transfer learning approach.
1.  **Base Model:** We used the **`EfficientNetB0`** architecture, pre-trained on the ImageNet dataset.
2.  **Freezing Layers:** The layers of the base model were initially frozen to leverage its powerful, pre-trained features.
3.  **Custom Head:** A new classification head was added on top, consisting of a `GlobalAveragePooling2D` layer, a `Dropout` layer for regularization, and a final `Dense` layer with `softmax` activation for multi-class classification.
4.  **Fine-Tuning:** The model was further optimized through fine-tuning, where the top layers of the base model were unfrozen and trained with a very low learning rate.

## 🚀 Setup and Usage

To run this project yourself, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    ```
2.  **Download the data:**
    * Download the [Landmark Recognition 2021 dataset](https://www.kaggle.com/c/landmark-recognition-2021/data) from Kaggle.
    * Use the data preparation steps outlined in the notebook to create the `landmark_subset.zip` file.
3.  **Open in Google Colab:**
    * Upload the `landmark_recognition.ipynb` notebook, `train.csv`, and `landmark_subset.zip` to your Colab session.
4.  **Run the notebook:**
    * Ensure the runtime is set to **GPU**.
    * Execute all cells from top to bottom.

