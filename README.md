Steel Industry Energy Consumption Prediction App

This project is a Streamlit application that predicts the energy consumption load type (Light, Medium, Maximum) for the steel industry. The app allows users to upload a dataset, preprocess the data, train a neural network model, and make predictions.



1.Upload a CSV dataset for training.

2.Data preprocessing including label encoding and standard scaling.

3.Neural network model training with TensorFlow and Keras.

4.Predict energy consumption load types based on user input.

5.Interactive UI using Streamlit.

Project Structure

Steel-Industry-Energy-Prediction

│── app.py                  
│── requirements.txt       
│── README.md             
│── sample_data.csv        

Requirements

1) Python 3.x

2) Streamlit

3) pandas

4) numpy

5) scikit-learn

TensorFlow

Model Training
The neural network model has 3 hidden layers with dropout regularization.
The output layer uses softmax activation for multi-class classification.
