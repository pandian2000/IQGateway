Project Name: Automated Diagnosis of Malaria Using Convolutional Neural Networks

Description:
This project involves the development of a machine learning model for the automated classification of malaria-infected and uninfected cells using cell images. 
The trained model is deployed as a Streamlit web application for interactive use.

Files:
1. dataset/: Contains the dataset of cell images, organized into infected and uninfected directories.
2. Malaria_cell_diagnosis.ipynb: Jupyter Notebook containing the code for  EDA,data preprocessing, model training and evaluation of trained model on test data by generating performance metrics.
3. [malaria_detector_architecture.json, malaria_detector_weights.weights.h5]
   - malaria_detector_architecture.json contains the architecture of the trained malaria detection model in JSON format.It likely describes the layers, configurations, and connections of the neural network model. This file is useful for reconstructing the model architecture without needing to retrain it, which can be helpful for model sharing, deployment, or transfer learning.
   - malaria_detector_weights.weights.h5 contains the weights of the trained malaria detection model. After training a neural network, the weights represent the learned parameters that define the model's behavior. Saving the weights separately allows for efficient storage and transfer of the model parameters, especially when combined with the model architecture stored in the JSON file.
   - These files collectively represent a trained machine learning model for malaria detection, including its architecture (JSON), weights (HDF5). They are essential for deploying, sharing, or further refining the trained model.
4. p2.py: Streamlit web application for interactive model deployment and visualization.
5. requirements.txt: List of Python packages required to run the project.

Instructions:
1. Dataset Preparation:
   - Ensure that the dataset is downloaded and stored in the "dataset/" directory.
   - The dataset should be organized into subdirectories for infected and uninfected cells.

2. Model Building:
   - Open and run the "Malaria_cell_diagnosis.ipynb" notebook in a Jupyter environment to build the  model.
   - This notebook contains code for EDA,data preprocessing, model training and evaluate the trained model on test data by generating  performance metrics such as accuracy, precision, recall, and F1-score. .
   - This notebook also contains code for saving both architecture and the learned parameters of the model separately,allowing for easy reconstruction and deployment of the trained model at a later time.
3. Streamlit Web Application:
   - To deploy the model and visualize results interactively, run the "p2.py" script.
   - Install the required Python packages listed in "requirements.txt" using pip if not already installed.
   - Execute the following command in the terminal to run the Streamlit app:
     ```
     streamlit run p2.py
     ```
   - Once the app is running, upload cell images to classify them as infected or uninfected and visualize the results.



