import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

# Define the LSTM model architecture 
class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1, dropout=0.3):
        super(SimpleLSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.linear(output)
        return output

# Function to load the ARIMA model
def load_arima_model(model_path='arima_model.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ARIMA model file not found at {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to load the LSTM model and scaler
def load_lstm_model(model_path='final_lstm_model_state_dict.pth', scaler_path='scaler.pkl', input_size=1, hidden_size=16, output_size=1, dropout=0.3):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model file not found at {model_path}")
    if not os.path.exists(scaler_path):
         raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Initialize and load LSTM model state dictionary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleLSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    return model, scaler, device

# Function to make predictions
def predict(model_name, data, forecast_horizon=1, sequence_length=60):
    """
    Makes predictions using either the ARIMA or LSTM model.

    Args:
        model_name (str): 'ARIMA' or 'LSTM'.
        data (pd.Series or np.ndarray): The input time series data.
        forecast_horizon (int): The number of steps to forecast ahead (for ARIMA).
        sequence_length (int): The length of input sequences for LSTM.

    Returns:
        np.ndarray: The predicted values.
    """
    if model_name.lower() == 'arima':
        arima_model = load_arima_model()
        
        try:
             
             if isinstance(data, np.ndarray):
                 data_series = pd.Series(data)
             else:
                 data_series = data

             
             arima_model = load_arima_model()
             forecast = arima_model.forecast(steps=forecast_horizon)
             return forecast.values # Return numpy array

        except Exception as e:
             print(f"Error during ARIMA prediction: {e}")
             # Fallback or error handling
             return np.array([np.nan] * forecast_horizon) # Return NaN if prediction fails


    elif model_name.lower() == 'lstm':
        lstm_model, scaler, device = load_lstm_model(sequence_length=sequence_length) # Pass sequence_length
        lstm_model.eval() 

        
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Reshape data for scaler (needs 2D input)
        data_reshaped = data.reshape(-1, 1)

      
        if len(data) < sequence_length:
            raise ValueError(f"Input data length ({len(data)}) is less than required sequence length ({sequence_length}) for LSTM.")

        # Take the last 'sequence_length' points and scale them
        input_sequence = data[-sequence_length:].reshape(-1, 1)
        scaled_input_sequence = scaler.transform(input_sequence)


        # Convert to tensor and add batch dimension
        scaled_input_tensor = torch.FloatTensor(scaled_input_sequence).unsqueeze(0).to(device) # Add batch and feature dimensions

        with torch.no_grad():
            prediction_scaled = lstm_model(scaled_input_tensor).cpu().numpy()

        # Inverse transform the prediction
        prediction_original = scaler.inverse_transform(prediction_scaled)

        return prediction_original.flatten() # Return numpy array


    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'ARIMA' or 'LSTM'.")

if __name__ == '__main__':
    
    try:
        
        print("Running example prediction...")
        dummy_recent_data = np.linspace(170, 180, 100) 

        # For LSTM, we need a sequence of length sequence_length
        lstm_input_data = dummy_recent_data[-sequence_length:] if len(dummy_recent_data) >= sequence_length else dummy_recent_data
        print("\nARIMA Prediction:")
        arima_prediction = predict('ARIMA') # Forecasts from the end of the saved model's training data
        print(f"Next 1-step ARIMA forecast: {arima_prediction[0]:.2f}") #  forecast_horizon=1

        print("\nLSTM Prediction:")
        if len(lstm_input_data) >= sequence_length:
            lstm_prediction = predict('LSTM', data=lstm_input_data, sequence_length=sequence_length)
            print(f"Next 1-step LSTM forecast: {lstm_prediction[0]:.2f}") #forecast_horizon=1 for comparison
        else:
            print(f"Not enough data ({len(lstm_input_data)}) for LSTM sequence length ({sequence_length}). Skipping LSTM example prediction.")

    except Exception as e:
        print(f"An error occurred during example usage: {e}")
