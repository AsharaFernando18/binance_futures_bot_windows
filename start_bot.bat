@echo off
REM Binance Futures Trading Bot Startup Script
REM This script sets environment variables and starts the trading bot

REM Set environment variables for API keys and Telegram credentials
REM Replace these with your actual keys and credentials
set BINANCE_API_KEY=
set BINANCE_SECRET_KEY=
set TELEGRAM_TOKEN=your_telegram_bot_token_here
set TELEGRAM_CHAT_ID=your_telegram_chat_id_here

REM Check if virtual environment exists, create if not
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install or update dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Start the trading bot
echo Starting Binance Futures Trading Bot...
python main.py

REM Deactivate virtual environment on exit
deactivate
