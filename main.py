"""
Main module for the Binance Futures trading bot.

This module orchestrates the startup and operation of the trading bot,
including loading config, identifying top cryptocurrencies, initializing
components, and running the event loop.
"""

import asyncio
import time
import signal
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import config
import logger
from data_fetcher import DataFetcher
from feature_engineering import FeatureEngineering
from model import ModelTrainer
from strategy import Strategy
from executor import Executor
from telegram_bot import TelegramBot

class TradingBot:
    """
    Main class for the Binance Futures trading bot.
    """
    
    def __init__(self):
        """Initialize the TradingBot with all components."""
        # Initialize logger
        self.logger = logger.setup_logger()
        self.logger.info("Initializing trading bot...")
        
        # Initialize components
        self.data_fetcher = None
        self.feature_engineering = None
        self.model_trainer = None
        self.strategy = None
        self.executor = None
        self.telegram_bot = None
        
        # Runtime variables
        self.running = False
        self.start_time = datetime.now()
        self.end_time = config.END_TIME
        self.total_trades = 0
        self.initial_balance = 0.0
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def initialize(self) -> bool:
        """
        Initialize all components of the trading bot.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Validate configuration
            if not config.validate_config():
                self.logger.error("Invalid configuration")
                return False
            
            # Initialize data fetcher
            self.logger.info("Initializing data fetcher...")
            self.data_fetcher = DataFetcher()
            await self.data_fetcher.initialize()
            
            # Initialize feature engineering
            self.logger.info("Initializing feature engineering...")
            self.feature_engineering = FeatureEngineering()
            
            # Initialize model trainer
            self.logger.info("Initializing model trainer...")
            self.model_trainer = ModelTrainer()
            
            # Initialize executor
            self.logger.info("Initializing executor...")
            self.executor = Executor()
            await self.executor.initialize()
            
            # Get initial account balance
            self.initial_balance = await self.executor.get_account_balance()
            self.logger.info("Initial account balance: %.2f USDT", self.initial_balance)
            
            # Initialize strategy
            self.logger.info("Initializing strategy...")
            self.strategy = Strategy(self.model_trainer)
            self.strategy.set_account_balance(self.initial_balance)
            
            # Initialize Telegram bot
            self.logger.info("Initializing Telegram bot...")
            self.telegram_bot = TelegramBot()
            
            # Start data streams
            self.logger.info("Starting data streams...")
            await self.data_fetcher.start_data_streams()
            
            # Send startup notification
            await self.telegram_bot.send_startup_notification(self.data_fetcher.symbols)
            
            self.logger.info("Trading bot initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception("Error initializing trading bot: %s", str(e))
            return False
    
    async def train_model(self) -> bool:
        """
        Train the ML model on historical data.
        
        Returns:
            True if training was successful, False otherwise
        """
        try:
            self.logger.info("Training model...")
            
            # Process historical data for each symbol
            for symbol in self.data_fetcher.symbols:
                for timeframe in config.TIMEFRAMES:
                    # Get historical data
                    df = self.data_fetcher.get_current_data(symbol, timeframe)
                    
                    if df.empty:
                        self.logger.warning("No historical data available for %s (%s)", 
                                          symbol, timeframe)
                        continue
                    
                    # Apply feature engineering
                    df = self.feature_engineering.process_data(df)
                    
                    # Generate target variable
                    df = self.model_trainer.generate_target(df)
                    
                    # Train model
                    success = self.model_trainer.train(df)
                    
                    if success:
                        self.logger.info("Model trained successfully on %s (%s) data", 
                                       symbol, timeframe)
                        
                        # Backtest model
                        metrics = self.model_trainer.backtest(df)
                        
                        if metrics:
                            self.logger.info("Backtest results for %s (%s):", 
                                           symbol, timeframe)
                            self.logger.info("  Win Rate: %.2f%%", metrics['win_rate'] * 100)
                            self.logger.info("  Profit Factor: %.2f", metrics['profit_factor'])
                            self.logger.info("  Sharpe Ratio: %.2f", metrics['sharpe_ratio'])
                        
                        # Only need to train on one symbol/timeframe combination
                        return True
                    
            self.logger.warning("Failed to train model on any symbol/timeframe combination")
            return False
            
        except Exception as e:
            self.logger.exception("Error training model: %s", str(e))
            return False
    
    async def run(self) -> None:
        """Run the trading bot main loop."""
        try:
            self.running = True
            self.logger.info("Starting trading bot main loop...")
            
            # Train model
            await self.train_model()
            
            # Main loop
            while self.running:
                # Check if runtime has expired
                if datetime.now() > self.end_time:
                    self.logger.info("Runtime window expired, shutting down...")
                    break
                
                # Process each symbol
                for symbol in self.data_fetcher.symbols:
                    try:
                        # Get latest data
                        df = self.data_fetcher.get_current_data(symbol, config.TIMEFRAMES[0])
                        
                        if df.empty:
                            self.logger.warning("No data available for %s", symbol)
                            continue
                        
                        # Apply feature engineering
                        df = self.feature_engineering.process_data(df)
                        
                        # Generate trading signal
                        signal = self.strategy.generate_signals(symbol, df)
                        
                        # Execute signal if not hold
                        if signal['action'] != 'hold':
                            result = await self.executor.execute_signal(signal)
                            
                            # Send notification
                            await self.telegram_bot.send_trade_notification(signal, result)
                            
                            # Update total trades count
                            if result['success'] and signal['action'] != 'hold':
                                self.total_trades += 1
                            
                            # Update account balance in strategy
                            balance = await self.executor.get_account_balance()
                            self.strategy.set_account_balance(balance)
                        
                    except Exception as e:
                        self.logger.exception("Error processing symbol %s: %s", symbol, str(e))
                
                # Check order status
                await self.executor.check_order_status()
                
                # Send balance update every hour
                if datetime.now().minute == 0 and datetime.now().second < 10:
                    balance_info = await self.executor.get_position_summary()
                    await self.telegram_bot.send_balance_update(balance_info)
                
                # Check if model needs retraining
                if self.model_trainer.should_retrain():
                    self.logger.info("Retraining model...")
                    await self.train_model()
                
                # Sleep for a short period
                await asyncio.sleep(10)
            
            # Graceful shutdown
            await self.shutdown()
            
        except Exception as e:
            self.logger.exception("Error in trading bot main loop: %s", str(e))
            await self.telegram_bot.send_error_notification(f"Critical error in main loop: {str(e)}")
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Perform graceful shutdown."""
        try:
            self.logger.info("Shutting down trading bot...")
            
            # Stop running
            self.running = False
            
            # Get final account balance
            final_balance = await self.executor.get_account_balance()
            
            # Calculate runtime
            runtime_minutes = (datetime.now() - self.start_time).total_seconds() / 60
            
            # Send shutdown notification
            await self.telegram_bot.send_shutdown_notification(
                runtime_minutes, 
                self.total_trades, 
                final_balance, 
                self.initial_balance
            )
            
            # Stop data fetcher
            if self.data_fetcher:
                await self.data_fetcher.stop()
            
            # Stop executor
            if self.executor:
                await self.executor.stop()
            
            self.logger.info("Trading bot shutdown complete")
            
        except Exception as e:
            self.logger.exception("Error during shutdown: %s", str(e))
    
    def _signal_handler(self, sig, frame) -> None:
        """
        Handle termination signals.
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        self.logger.info("Received termination signal, shutting down...")
        self.running = False
        
        # Create asyncio task for shutdown
        loop = asyncio.get_event_loop()
        loop.create_task(self.shutdown())

async def main() -> None:
    """Main entry point for the trading bot."""
    # Create and initialize trading bot
    bot = TradingBot()
    
    # Initialize bot
    success = await bot.initialize()
    
    if success:
        # Run bot
        await bot.run()
    else:
        logger.error("Failed to initialize trading bot")
        sys.exit(1)

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
