import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def preprocess_data(data, column_name):
    """ Preprocess data for LSTM input """
    # Extract the desired column
    series = data[column_name]
    
    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    return scaled_data, scaler

def create_lstm_model(input_shape):
    """ Create and compile an LSTM model """
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_lstm_data(scaled_data, time_step=60):
    """ Prepare data for LSTM (X, y split) """
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X to be compatible with LSTM (samples, time steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y

def plot_bms_data(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return
    
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Convert 'Time' column to datetime
    data['Time'] = pd.to_datetime(data['Time'], errors='coerce')

    # Set 'Time' as the index
    data.set_index('Time', inplace=True)

    # Plotting the data for visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)

    # Plot Pack Amperage (Current)
    axes[0].plot(data.index, data['Pack Amperage (Current)'], label='Pack Amperage (Current)', color='blue')
    axes[0].set_ylabel('Amperage (A)')
    axes[0].set_title('Pack Amperage Over Time')
    axes[0].legend()

    # Plot Pack Voltage
    axes[1].plot(data.index, data['Pack Voltage'], label='Pack Voltage', color='orange')
    axes[1].set_ylabel('Voltage (V)')
    axes[1].set_title('Pack Voltage Over Time')
    axes[1].legend()

    # Plot Pack State of Charge (SOC)
    axes[2].plot(data.index, data['Pack State of Charge (SOC)'], label='Pack SOC', color='green')
    axes[2].set_ylabel('State of Charge (%)')
    axes[2].set_title('Pack State of Charge Over Time')
    axes[2].legend()

    # Plot individual cell voltages
    for col in [col for col in data.columns if 'Inst. Cell' in col]:
        axes[3].plot(data.index, data[col], label=col, alpha=0.7)
    axes[3].set_ylabel('Voltage (V)')
    axes[3].set_title('Individual Cell Voltages Over Time')
    axes[3].legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout and save the plots
    plt.tight_layout()
    plt.show()

    # Predict the next data points for 'Pack Voltage' using LSTM
    column_name = 'Pack Voltage'
    
    # Preprocess data
    scaled_data, scaler = preprocess_data(data, column_name)
    
    # Prepare data for LSTM
    time_step = 60  # Use 60 previous time steps to predict the next value
    X, y = prepare_lstm_data(scaled_data, time_step)
    
    # Create and compile the LSTM model
    model = create_lstm_model((X.shape[1], 1))
    
    # Train the model with early stopping
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, callbacks=[early_stop])
    
    # Predict future values
    predictions = model.predict(X)
    
    # Inverse transform the predictions to get actual values
    predictions = scaler.inverse_transform(predictions)
    
    # Plot the actual vs predicted values for 'Pack Voltage'
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[time_step:], data['Pack Voltage'][time_step:], label='Actual Pack Voltage', color='blue')
    plt.plot(data.index[time_step:], predictions, label='Predicted Pack Voltage', color='red')
    plt.xlabel('Time')
    plt.ylabel('Pack Voltage (V)')
    plt.title('Actual vs Predicted Pack Voltage')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot BMS Data from a CSV file and predict future values")
    parser.add_argument("file", help="Path to the BMS data CSV file")
    args = parser.parse_args()

    # Call the function with the provided file path
    plot_bms_data(args.file)