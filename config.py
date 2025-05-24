"""
Configuration module for the Binance Futures trading bot.

This module handles loading API keys from environment variables,
strategy parameters, Telegram credentials, and runtime settings.
"""

import os
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

# API Configuration
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY", "")

# Testnet flag - Always true for safety
USE_TESTNET = True

# Telegram Configuration
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Runtime Configuration
RUNTIME_HOURS = 6  # Default runtime window in hours
START_TIME = datetime.now()
END_TIME = START_TIME + timedelta(hours=RUNTIME_HOURS)

# Trading Parameters
MAX_OPEN_POSITIONS = 3  # Maximum number of open positions
POSITION_SIZE_PERCENT = 0.05  # Position size as percentage of account balance
MAX_DRAWDOWN_PERCENT = 0.15  # Maximum allowed drawdown before stopping

# Risk Management
USE_STOP_LOSS = True
STOP_LOSS_PERCENT = 0.02  # Stop loss as percentage of entry price
USE_TAKE_PROFIT = True
TAKE_PROFIT_PERCENT = 0.04  # Take profit as percentage of entry price
TRAILING_STOP = False
TRAILING_STOP_PERCENT = 0.01  # Trailing stop percentage

# Data Parameters
TIMEFRAMES = ["15m"]  # Timeframes to fetch data for
LOOKBACK_PERIODS = 100  # Number of candles to fetch for historical data
TOP_SYMBOLS_COUNT = 3  # Number of top symbols to trade

# Feature Engineering Parameters
FEATURE_PARAMS = {
    "ema_short": 9,
    "ema_medium": 21,
    "ema_long": 50,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "volume_ma_period": 20,
}

# ML Model Parameters
MODEL_PARAMS = {
    "model_type": "random_forest",  # Options: random_forest, gradient_boosting, neural_network
    "train_test_split": 0.8,
    "prediction_threshold": 0.6,  # Threshold for buy/sell signals
    "retrain_interval_hours": 24,  # How often to retrain the model
}

# Logging Configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "trading_bot.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# Check if configuration is valid
def validate_config() -> bool:
    """
    Validate the configuration settings.
    
    Returns:
        bool: True if configuration is valid, False otherwise.
    """
    if USE_TESTNET:
        # In testnet mode, we don't strictly need API keys
        return True
    
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        print("Error: Binance API keys not set. Please set BINANCE_API_KEY and BINANCE_SECRET_KEY environment variables.")
        return False
    
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Warning: Telegram credentials not set. Notifications will be disabled.")
    
    return True
