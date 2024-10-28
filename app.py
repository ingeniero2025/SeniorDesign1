# app.py

from flask import Flask, render_template, send_file, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from threading import Thread
import os

# Create Flask app
app = Flask(__name__)

# Step 1: Load data
data = pd.read_excel('Battery_Test_Data.xlsx', sheet_name='Sheet2')

# Step 2: Feature selection and normalization
features = data[['Load_Current', 'Cell_1_Current', 'Cell_2_Current', 'Cell_3_Current', 'Cell_4_Current', 'Pack_Voltage']].values
targets = data[['Cell_1_SOC', 'Cell_2_SOC', 'Cell_3_SOC', 'Cell_4_SOC']].values

# Normalize the features and targets
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)
targets_normalized = scaler.fit_transform(targets)

# Step 3: Prepare data for LSTM
X = features_normalized.reshape((features_normalized.shape[0], 1, features_normalized.shape[1]))
y = targets_normalized

# Step 4: Build and train the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(4))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=32, verbose=1)

# Step 5: Make predictions and rescale
predictions = model.predict(X)
predictions_rescaled = scaler.inverse_transform(predictions)
actual_rescaled = scaler.inverse_transform(targets_normalized)

# Create directory for plots
if not os.path.exists('soc_plots'):
    os.makedirs('soc_plots')

# Step 6: Plot each cell's actual vs. predicted SOC
time_steps = np.arange(len(predictions_rescaled))
for i in range(4):
    plt.figure(figsize=(14, 5))
    plt.plot(time_steps, actual_rescaled[:, i], label='Actual SOC', color='blue', alpha=0.7)
    plt.plot(time_steps, predictions_rescaled[:, i], label='Predicted SOC', color='orange', linestyle='--', alpha=0.7)
    plt.title(f'SOC for Cell {i + 1}')
    plt.xlabel('Time Steps')
    plt.ylabel('State of Charge (SOC)')
    plt.legend()
    plt.savefig(f'soc_plots/cell_{i + 1}_soc_plot.png')
    plt.close()

# Home route to render index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve each cell plot
@app.route('/get_plot/<int:cell_number>', methods=['GET'])
def get_plot(cell_number):
    plot_file = f'soc_plots/cell_{cell_number}_soc_plot.png'
    if os.path.exists(plot_file):
        return send_file(plot_file, mimetype='image/png')
    else:
        return jsonify({"error": "Plot not found"}), 404

# Run Flask app
def run_app():
    app.run(port=5000)
    print("Flask app is running at http://127.0.0.1:5000/")

# Start the app
if __name__ == '__main__':
    thread = Thread(target=run_app)
    thread.start()
