Steel Industry Energy Consumption Prediction App

This project is a Streamlit application that predicts the energy consumption load type (Light, Medium, Maximum) for the steel industry. The app allows users to upload a dataset, preprocess the data, train a neural network model, and make predictions.

🚀 Features

1.Upload a CSV dataset for training.
2.Data preprocessing including label encoding and standard scaling.
3.Neural network model training with TensorFlow and Keras.
4.Predict energy consumption load types based on user input.
5.Interactive UI using Streamlit.

📁 Project Structure
📦 Steel-Industry-Energy-Prediction
│── app.py                 # Main Streamlit app
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
│── sample_data.csv        # Example dataset

🛠️ Requirements
Python 3.x
Streamlit
pandas
numpy
scikit-learn
TensorFlow

🧠 Model Training
The neural network model has 3 hidden layers with dropout regularization.
The output layer uses softmax activation for multi-class classification.
