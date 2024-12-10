# config.py
DIP_THRESHOLD = 0.5  # Probability threshold for dip detection
MODEL_PATH = 'dip_model.pth'
SCALER_PATH = 'scaler.pkl'
TIMEFRAME = 'H1'
SYMBOL = 'BTCUSD'

# Next candle time calculation: (For H1 timeframe)
# You can adjust this mapping for other timeframes if needed
TIMEFRAME_SECONDS = {
    'M1': 60,
    'M5': 300,
    'M15': 900,
    'M30': 1800,
    'H1': 3600,
    'H4': 14400,
    'D1': 86400
}

# Logging configuration
LOG_FILE = 'dip_detection.log'
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
