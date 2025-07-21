Steel Industry Energy Consumption Prediction App
Predict energy load types (Light, Medium, Maximum) for steel plants using AI

This Streamlit app helps steel industry professionals forecast energy consumption patterns by training a neural network on historical data. Just upload your dataset, preprocess it, and get predictions in minutes!

Features
1) Simple CSV Upload – Drop your dataset (like sample_data.csv) and get started
2) Auto Preprocessing – Handles encoding & scaling so you don’t have to
3) Neural Network Model – 3 hidden layers with dropout to prevent overfitting
4) Interactive UI – Clean, user-friendly interface powered by Streamlit

How It Works
Upload Data – Provide a CSV with energy consumption metrics

Preprocess – The app encodes labels and scales features automatically

Train Model – A TensorFlow/Keras neural network learns patterns in your data

Predict – Input new data points and classify load type (Light/Medium/Max)

Project Structure
bash
Steel-Industry-Energy-Prediction/  
├── app.py                
├── requirements.txt     
├── README.md             
└── sample_data.csv      

Setup
Install dependencies:

bash
pip install -r requirements.txt
Run the app:

bash
streamlit run app.py



