import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Custom CSS to style the app
st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: #ffffff;
            font-family: 'Poppins', sans-serif;
        }
        h1 {
            color: #00bfff; /* Bright blue */
            text-align: center;
        }
        .stButton button {
            background-color: #8a2be2; /* BlueViolet */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #7b68ee; /* MediumSlateBlue */
        }
        .stTextInput input, .stNumberInput input, .stSelectbox select {
            background-color: #1e1e1e; /* Dark gray */
            color: #ffffff;
            padding: 10px;
            border-radius: 8px;
            font-size: 14px;
            border: 1px solid #4b0082; /* Indigo */
        }
        .stMetric {
            color: #00ff00; /* Bright green */
        }
        .stText {
            color: #d3d3d3; /* Light gray */
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a section:", ["Home", "Upload Dataset"])

# Layout for displaying content
col1, col2 = st.columns([1, 3])  # Adjust the ratio as needed

if app_mode == "Home":
    with col2:
        st.title('Welcome to the Steel Industry Energy Consumption Prediction App')
        st.write("Use the navigation bar to upload your dataset and make predictions.")
        st.write("This app allows you to upload a dataset, train a neural network model, and make predictions.")

elif app_mode == "Upload Dataset":
    with col2:
        st.title('📁 Upload Your Dataset')
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.header("🔍 Data Preview:")
            st.write(df.head())

            # Data Preprocessing
            df['Load_Type'].replace(['Light_Load', 'Medium_Load', 'Maximum_Load'], [0, 1, 2], inplace=True)
            df['WeekStatus'] = df['WeekStatus'].replace(['Weekend', 'Weekday'], [0, 1], inplace=False)

            label_encoders = {}
            for column in ['WeekStatus', 'Day_of_week', 'Load_Type']:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le

            scaler = StandardScaler()
            continuous_features = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 
                                   'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)', 
                                   'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 
                                   'NSM']

            df[continuous_features] = scaler.fit_transform(df[continuous_features])

            X = df.drop(columns=['date', 'Load_Type'])
            y = df['Load_Type']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = Sequential()
            model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(3, activation='softmax'))

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            st.header("⚡ Energy Consumption Prediction")
            
            # Train the model
            if st.button("🚀 Train Model"):
                with st.spinner('Training...'):
                    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
                st.success('🎉 Training Completed!')

              
                loss, accuracy = model.evaluate(X_test, y_test)
                st.metric(label="📊 Test Accuracy", value=f"{accuracy:.2f}")

       
            st.header("🧠 Make Predictions with the Trained Model")
            
            with st.form("prediction_form"):
                cols = st.columns(3)

                
                continuous_input = []
                for i, feature in enumerate(continuous_features):
                    value = cols[i % 3].number_input(f"Input {feature}", value=0.00, format="%.2f")
                    continuous_input.append(value)

             
                categorical_input = []
                for i, feature in enumerate(['WeekStatus', 'Day_of_week']):
                    value = cols[i % 2 + 1].selectbox(f"Select {feature}", options=label_encoders[feature].classes_)
                    encoded_value = label_encoders[feature].transform([value])[0]
                    categorical_input.append(encoded_value)

                predict_button = st.form_submit_button(label="🔮 Predict")
                
                if predict_button:
                    # Scale the continuous input using the previously fitted scaler
                    continuous_input = scaler.transform([continuous_input])[0]
                    
                    # Combine continuous and categorical inputs
                    sample_input = list(continuous_input) + categorical_input
                    
                    # Convert sample_input to a NumPy array and reshape it to be 2D
                    sample_input = np.array(sample_input).reshape(1, -1)
                    
                    # Predict the class using the trained model
                    prediction = model.predict(sample_input)
                    predicted_class = prediction.argmax(axis=1)[0]
                    
                    # Convert the predicted class back to its original label
                    load_type = label_encoders['Load_Type'].inverse_transform([predicted_class])[0]
                    st.success(f"💡 Predicted Load Type: {load_type}")