import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import time
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from config import MODEL_PATH, SCALER_PATH, TIMEFRAME, SYMBOL, DIP_THRESHOLD, TIMEFRAME_SECONDS, LOG_FILE, LOG_LEVEL

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=getattr(logging, LOG_LEVEL),
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

# To avoid recalculating entire rolling features each time:
# We assume we have a data_history and we just append new candles.
def update_features(data_history):
    # Only compute on the new segment to save computation
    # But for simplicity, we recompute on the entire dataset. 
    # TODO: Optimize by only updating the last rolling calculations.
    data_history['ma_20'] = data_history['close'].rolling(window=20).mean()
    data_history['rsi'] = compute_rsi(data_history['close'])
    return data_history

########################
# MT5 CONNECTION & DATA FETCHING
########################
def connect_mt5(account, password, server):
    if not mt5.initialize():
        logging.error(f"MT5 Initialization failed, error: {mt5.last_error()}")
        return False
    authorized = mt5.login(account, password=password, server=server)
    if not authorized:
        logging.error(f"Login failed, error: {mt5.last_error()}")
    else:
        logging.info("MT5 connected successfully.")
    return authorized

def get_latest_candle(symbol, timeframe):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
    if rates is None or len(rates) == 0:
        logging.warning("No data received for the latest candle.")
        return None
    candle = pd.DataFrame(rates)
    candle['time'] = pd.to_datetime(candle['time'], unit='s')
    return candle

########################
# PREDICTION
########################
def predict_dip(model, scaler, latest_features):
    model.eval()
    with torch.no_grad():
        scaled_features = scaler.transform([latest_features])
        X_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        prediction = model(X_tensor).item()
    return prediction  # Probability of dip

########################
# TIME MANAGEMENT
########################
def get_time_until_next_candle():
    # Calculate time until next candle based on current time and timeframe
    timeframe_sec = TIMEFRAME_SECONDS.get(TIMEFRAME, 3600)
    now = datetime.utcnow()
    # Next candle time = now rounded up to the nearest timeframe interval
    epoch_now = int(now.timestamp())
    next_candle_epoch = ((epoch_now // timeframe_sec) + 1) * timeframe_sec
    wait_time = next_candle_epoch - epoch_now
    return wait_time if wait_time > 0 else timeframe_sec

########################
# MAIN LIVE DETECTION
########################
def main():
    # Replace these credentials with your MT5 details
    account = 123456
    password = "your_password"
    server = "your_server"

    if not connect_mt5(account, password, server):
        return

    # Load model and scaler
    if not (MODEL_PATH and SCALER_PATH):
        logging.error("Model or Scaler paths are not set properly.")
        return

    if not mt5.symbol_select(SYMBOL, True):
        logging.error(f"Could not select symbol {SYMBOL}")
        return

    if not torch.cuda.is_available():
        logging.info("CUDA not available. Using CPU.")

    # Loading model and scaler
    model = DipModel(input_size=3)  # we know we have 3 features: close, ma_20, rsi
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    scaler = joblib.load(SCALER_PATH)

    # Initialize data history
    # Fetch some history to start with
    candles = mt5.copy_rates_from_pos(SYMBOL, getattr(mt5, 'TIMEFRAME_'+TIMEFRAME), 0, 1000)
    if candles is None or len(candles) == 0:
        logging.error("Failed to fetch initial data.")
        mt5.shutdown()
        return
    
    data_history = pd.DataFrame(candles)
    data_history['time'] = pd.to_datetime(data_history['time'], unit='s')
    data_history = update_features(data_history)
    data_history = data_history.dropna().reset_index(drop=True)

    # Start live dip detection
    logging.info("Starting live dip detection...")
    
    features = ['close', 'ma_20', 'rsi']
    
    try:
        while True:
            latest_candle = get_latest_candle(SYMBOL, getattr(mt5, 'TIMEFRAME_'+TIMEFRAME))
            if latest_candle is not None:
                last_time = data_history['time'].iloc[-1]
                new_time = latest_candle['time'].iloc[0]
                if new_time > last_time:
                    logging.info(f"New candle received at {new_time}")
                    # Append new candle and update features
                    data_history = pd.concat([data_history, latest_candle], ignore_index=True)
                    data_history = update_features(data_history)
                    latest_row = data_history.iloc[-1]
                    
                    if latest_row[features].isnull().any():
                        logging.info("Not enough data to form features yet.")
                    else:
                        latest_features = latest_row[features].values
                        probability = predict_dip(model, scaler, latest_features)
                        is_dip = probability > DIP_THRESHOLD
                        logging.info(f"Time: {new_time}, Dip Probability: {probability:.4f}, Dip Detected: {is_dip}")
                        if is_dip:
                            # Implement your dip handling logic here
                            logging.info("Dip detected! Consider taking action.")
                else:
                    logging.debug("No new candle.")
            else:
                logging.debug("No candle data fetched.")

            # Wait until the next candle is due
            wait_time = get_time_until_next_candle()
            time.sleep(wait_time)
    
    except KeyboardInterrupt:
        logging.info("Live dip detection stopped by user.")
    
    finally:
        mt5.shutdown()
        logging.info("MT5 connection closed.")


if __name__ == "__main__":
    main()
