# Binance Futures Trading Bot

A modular, production-quality trading bot for Binance Futures that uses machine learning to generate trading signals for the top 3 most profitable cryptocurrencies.

## Features

- **Fully Automated Trading**: Identifies top cryptocurrencies, generates signals, and executes trades
- **Machine Learning**: Uses ML models to predict price movements and generate trading signals
- **Risk Management**: Implements position sizing, stop-loss, take-profit, and maximum drawdown limits
- **Real-time Monitoring**: Sends trade notifications and account updates via Telegram
- **Modular Architecture**: Well-structured codebase with clear separation of concerns
- **Testnet Support**: Safely test strategies on Binance Futures Testnet

## Architecture

The trading bot is built with a modular architecture, with each component responsible for a specific aspect of the trading process:

- **config.py**: Configuration settings, API keys, and trading parameters
- **logger.py**: Centralized logging to console and rotating log files
- **data_fetcher.py**: Connects to Binance to stream real-time market data
- **feature_engineering.py**: Transforms raw data into features for ML models
- **model.py**: Implements ML pipeline for signal generation
- **strategy.py**: Generates trading signals based on ML predictions
- **executor.py**: Places and manages orders on Binance Futures
- **telegram_bot.py**: Sends real-time alerts and notifications
- **main.py**: Orchestrates the entire trading process

## Installation

### Prerequisites

- Python 3.10 or higher
- Binance account with API keys (Testnet for testing)
- Telegram bot token and chat ID (optional, for notifications)

### Setup

1. Clone the repository or download the source code

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   # On Linux/Mac:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure your API keys and Telegram credentials:

   **For Linux/Mac (start_bot.sh):**
   ```bash
   export BINANCE_API_KEY="your_binance_api_key_here"
   export BINANCE_SECRET_KEY="your_binance_secret_key_here"
   export TELEGRAM_TOKEN="your_telegram_bot_token_here"
   export TELEGRAM_CHAT_ID="your_telegram_chat_id_here"
   # Set to "false" to use the live Binance environment, defaults to "true" (testnet)
   export USE_TESTNET="true" 
   ```
   
   **For Windows (start_bot.bat):**
   ```batch
   set BINANCE_API_KEY=your_binance_api_key_here
   set BINANCE_SECRET_KEY=your_binance_secret_key_here
   set TELEGRAM_TOKEN=your_telegram_bot_token_here
   set TELEGRAM_CHAT_ID=your_telegram_chat_id_here
   REM Set to "false" to use the live Binance environment, defaults to "true" (testnet)
   set USE_TESTNET=true
   ```

## Configuration

The bot's behavior can be customized by editing the parameters in `config.py` or by setting environment variables.

### Environment Variables

- `BINANCE_API_KEY`: Your Binance API key.
- `BINANCE_SECRET_KEY`: Your Binance secret key.
- `TELEGRAM_TOKEN`: Your Telegram bot token (optional).
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID (optional).
- `USE_TESTNET`: Set to `false` to use the live Binance environment. Defaults to `true` (testnet mode for safety).

### Trading Parameters (in `config.py`)

- `MAX_OPEN_POSITIONS`: Maximum number of open positions (default: 3)
- `POSITION_SIZE_PERCENT`: Position size as percentage of account balance (default: 5%)
- `MAX_DRAWDOWN_PERCENT`: Maximum allowed drawdown before stopping (default: 15%)

### Risk Management

- `USE_STOP_LOSS`: Whether to use stop-loss orders (default: True)
- `STOP_LOSS_PERCENT`: Stop loss as percentage of entry price (default: 2%)
- `USE_TAKE_PROFIT`: Whether to use take-profit orders (default: True)
- `TAKE_PROFIT_PERCENT`: Take profit as percentage of entry price (default: 4%)
- `TRAILING_STOP`: Whether to use trailing stops (default: False)
- `TRAILING_STOP_PERCENT`: Trailing stop percentage (default: 1%)

### ML Model Parameters

- `MODEL_PARAMS`: Dictionary with model configuration
  - `model_type`: Type of ML model to use (options: random_forest, gradient_boosting, neural_network)
  - `prediction_threshold`: Threshold for buy/sell signals (default: 0.6)
  - `retrain_interval_hours`: How often to retrain the model (default: 24 hours)

## Usage

### Running the Bot

**For Linux/Mac:**
1. Make the startup script executable:
   ```bash
   chmod +x start_bot.sh
   ```

2. Run the bot:
   ```bash
   ./start_bot.sh
   ```

**For Windows:**
1. Run the batch file by double-clicking it in File Explorer or using Command Prompt:
   ```
   start_bot.bat
   ```

The bot will:
1. Initialize all components
2. Identify the top 3 most profitable cryptocurrencies
3. Train the ML model on historical data
4. Start generating and executing trading signals
5. Send notifications via Telegram (if configured)
6. Run for the configured time window (default: 6 hours)
7. Gracefully shut down and send a summary report

### Monitoring

If Telegram integration is configured, you will receive:
- Startup notification with selected trading symbols
- Trade notifications when positions are opened or closed
- Hourly account balance and P&L updates
- Error notifications for critical issues
- Shutdown summary with performance metrics

You can also monitor the bot's activity through the log file (`trading_bot.log`).

## Extending the Bot

### Adding New Features

The modular architecture makes it easy to extend the bot with new features:

- **New Technical Indicators**: Add new indicators in `feature_engineering.py`
- **Different ML Models**: Implement new model types in `model.py`
- **Custom Trading Strategies**: Modify signal generation logic in `strategy.py`
- **Additional Risk Management**: Enhance position management in `executor.py`

### Retraining the Model

The model is automatically retrained based on the `retrain_interval_hours` parameter in `config.py`. To manually retrain:

1. Modify the `should_retrain` method in `model.py` to return `True`
2. The bot will retrain the model on the next iteration

### Backtesting

The `model.py` module includes a `backtest` method that evaluates the model's performance on historical data. The results include:

- Win rate
- Profit factor
- Sharpe ratio
- Maximum drawdown

## Security Considerations

- **API Keys**: Never hardcode API keys in the source code
- **Testnet**: Always test new strategies on Testnet before using real funds
- **Position Sizing**: Use conservative position sizes to limit risk
- **Stop Loss**: Always use stop-loss orders to protect against large losses

## Troubleshooting

### Common Issues

- **Connection Errors**: Check your internet connection and Binance API status
- **Authentication Errors**: Verify your API keys and permissions
- **Insufficient Balance**: Ensure you have enough funds in your account
- **Order Placement Failures**: Check symbol trading status and account restrictions

### Logging

The bot uses a centralized logging system with different log levels:

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about bot operation
- **WARNING**: Potential issues that don't affect operation
- **ERROR**: Errors that affect specific operations
- **CRITICAL**: Critical errors that require immediate attention

Log files are stored with rotation to prevent excessive disk usage.

## Disclaimer

This trading bot is provided for educational and research purposes only. Trading cryptocurrency futures involves significant risk and may not be suitable for all investors. Past performance is not indicative of future results. Always use proper risk management and never trade with funds you cannot afford to lose.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
