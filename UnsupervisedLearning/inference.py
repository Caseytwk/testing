import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Anomaly Data
anomaly_data = {
    "network_packet_size": 108.0,
    "login_attempts": 7.0,
    "session_duration": 3325.308689,
    "ip_reputation_score": 0.001509,
    "failed_logins": 2.0,
    "unusual_time_access": 0.0,
    "protocol_type": "TCP",
    "encryption_used": "DES",
    "browser_type": "Chrome"
}

# Categorical Features and Numerical Features
cat_features = ["protocol_type", "encryption_used", "browser_type"]
num_features = ["network_packet_size", "login_attempts", "session_duration", "ip_reputation_score", "failed_logins", "unusual_time_access"]

def load_configurations():
    # Load Configurations
    scaler = joblib.load("./model/scaler.pkl") # MinMaxScaler
    encoder = joblib.load("./model/encoder.pkl") # OneHotEncoder
    threshold = joblib.load("./model/threshold.pkl") # 95th percentile
    model = tf.keras.models.load_model("./model/autoencoder.h5") # Load Model
    return scaler, encoder, threshold, model

def preprocess_data(data, scaler, encoder):
    # Convert to DataFrame
    new_df = pd.DataFrame([data])

    # Normalize numerical features
    new_df[num_features] = scaler.transform(new_df[num_features])  # Use the same scaler!

    # One-hot encode categorical features
    new_encoded = encoder.transform(new_df[cat_features])
    new_encoded_df = pd.DataFrame(new_encoded, columns=encoder.get_feature_names_out(cat_features))

    # Merge preprocessed features
    new_df = new_df.drop(columns=cat_features).reset_index(drop=True)
    new_df = pd.concat([new_df, new_encoded_df], axis=1)

    # Convert to NumPy array for TensorFlow
    new_input = new_df.to_numpy()
    return new_input


def inferencing(new_input, model):
    # Pass the input through the autoencoder
    reconstructed = model.predict(new_input)

    # Compute Reconstruction Error (MAE)
    error = np.mean(np.abs(new_input - reconstructed))
    return error

scaler, encoder, threshold, model = load_configurations()
new_input = preprocess_data(anomaly_data, scaler, encoder)
error = inferencing(new_input, model)

# Print the Reconstruction Error
# print(f"Reconstruction Error: {error:.6f}")

if error > threshold:
    print("🚨 Anomalous Data Point Detected!")
else:
    print("✅ Normal Data Point")