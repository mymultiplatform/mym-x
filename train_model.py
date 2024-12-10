import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import joblib
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
from config import MODEL_PATH, SCALER_PATH, SYMBOL, TIMEFRAME

# Setup logging
logging.basicConfig(filename='model_training.log', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

########################
# MODEL DEFINITION
########################
class DipModel(nn.Module):
    def __init__(self, input_size):
        super(DipModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

########################
# FEATURE ENGINEERING
########################
def compute_rsi(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(data_history):
    data_history['ma_20'] = data_history['close'].rolling(window=20).mean()
    data_history['rsi'] = compute_rsi(data_history['close'])
    data_history = data_history.dropna().reset_index(drop=True)
    return data_history

########################
# TRAINING ROUTINE
########################
def train_model():
    # TODO: Replace with your static training dataset path
    # For better reproducibility, you may store a static dataset in CSV form locally.
    # E.g., PATH_TO_TRAINING_DATA = 'historical_data.csv'
    # data_history = pd.read_csv(PATH_TO_TRAINING_DATA)
    
    # For demonstration, we connect to MT5 and fetch historical data:
    if not mt5.initialize():
        logging.error(f"MT5 Initialization failed, error: {mt5.last_error()}")
        return
    rates = mt5.copy_rates_from_pos(SYMBOL, getattr(mt5, 'TIMEFRAME_'+TIMEFRAME), 0, 2000)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        logging.error("Failed to fetch historical data.")
        return
    data_history = pd.DataFrame(rates)
    data_history['time'] = pd.to_datetime(data_history['time'], unit='s')

    # Feature Engineering
    data_history = add_features(data_history)
    data_history['returns'] = data_history['close'].pct_change()
    data_history['dip'] = (data_history['returns'] < -0.03).astype(int)
    data_history = data_history.dropna()
    
    features = ['close', 'ma_20', 'rsi']
    X = data_history[features].values
    y = data_history['dip'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    model = DipModel(input_size=len(features))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    # Save the model and scaler
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logging.info("Model trained and saved successfully.")


if __name__ == "__main__":
    train_model()
