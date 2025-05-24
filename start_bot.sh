#!/bin/bash

# Binance Futures Trading Bot Startup Script
# This script sets environment variables and starts the trading bot

# Set environment variables for API keys and Telegram credentials
# Replace these with your actual keys and credentials
export BINANCE_API_KEY="your_binance_api_key_here"
export BINANCE_SECRET_KEY="your_binance_secret_key_here"
export TELEGRAM_TOKEN="your_telegram_bot_token_here"
export TELEGRAM_CHAT_ID="your_telegram_chat_id_here"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install or update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the trading bot
echo "Starting Binance Futures Trading Bot..."
python main.py

# Deactivate virtual environment on exit
deactivate
