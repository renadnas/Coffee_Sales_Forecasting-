# Coffee_Sales_Forecasting-
This project implements and compares three different neural network architectures to forecast daily coffee sales using a dataset from Kaggle. The objective is to predict future sales amounts based on historical patterns through sequence modeling and reservoir computing.

Models Implemented
1. Echo State Network (ESN)
A type of reservoir computing that uses a fixed, sparsely connected recurrent structure.
Architecture: 600 neurons in the reservoir with a spectral radius of 0.8.
Input: 30-day sliding window of lagged features.
Key Feature: Includes a "warm-up" period to stabilize the reservoir state before prediction.
2. Long Short-Term Memory (LSTM)
A standard recurrent neural network (RNN) designed to capture long-term dependencies in time series data.
Architecture: 2 LSTM layers with 64 hidden units and a 0.3 dropout rate.
Training: Optimized using Adam over 50 epochs with Mean Squared Error (MSE) loss.
3. Bidirectional LSTM (BiLSTM)
An extension of the LSTM that processes data in both forward and backward directions.
Architecture: Dual-directional layers with a fully connected layer sized at $hidden\_size \times 2$ to handle the concatenated outputs.
Benefit: Captures information from both past and future contexts within the sequence length.
 Dataset & Preprocessing
Source: via KaggleHub.
Aggregation: Sales are summed by date to create a daily revenue time series.
Smoothing: A 7-day rolling mean is applied to reduce noise and highlight underlying trends.
Scaling: Data is normalized using MinMaxScaler to a range of $[0, 1]$ to improve neural network convergence.

Project Structure
ESN_time_series.py: Implementation using the pyESN library with a 30-day window.
LSTM_time_series.py: PyTorch-based implementation of a standard LSTM.
BiLSTM.py: PyTorch-based implementation of a Bidirectional LSTM.

Evaluation Metrics
All models are evaluated on a 20% test split using the following metrics:
MSE: Mean Squared Error
RMSE: Root Mean Squared Error
MAE: Mean Absolute Error
R² Score: Coefficient of Determination

Requirements
How to Run
Clone this repository.
Ensure you have an internet connection to download the dataset automatically via kagglehub.
Run any of the model scripts:


