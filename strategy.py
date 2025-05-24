"""
Trading strategy module for the Binance Futures trading bot.

This module uses outputs from the ML model to generate buy/sell/hold signals
and incorporates risk management including position sizing, stop-loss,
take-profit, and max drawdown limits.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import config
import logger
from model import ModelTrainer

class Strategy:
    """
    Class for generating trading signals and managing risk.
    """
    
    def __init__(self, model_trainer: ModelTrainer):
        """
        Initialize the Strategy with a model trainer.
        
        Args:
            model_trainer: Trained ML model for signal generation
        """
        self.model_trainer = model_trainer
        self.positions = {}  # Current open positions
        self.account_balance = 0.0  # Will be updated from executor
        self.initial_balance = 0.0  # Initial account balance
        self.max_drawdown = 0.0  # Maximum drawdown experienced
        
        logger.info("Strategy initialized")
    
    def set_account_balance(self, balance: float) -> None:
        """
        Set the current account balance.
        
        Args:
            balance: Current account balance
        """
        self.account_balance = balance
        
        # Set initial balance if not set
        if self.initial_balance == 0.0:
            self.initial_balance = balance
            
        logger.debug("Account balance set to %.2f USDT", balance)
    
    def generate_signals(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals for a symbol based on ML model predictions.
        
        Args:
            symbol: Trading pair symbol
            df: DataFrame with features
            
        Returns:
            Dictionary with signal information
        """
        try:
            # Get the latest candle
            latest_candle = df.iloc[-1]
            
            # Generate prediction
            predictions = self.model_trainer.predict(df.iloc[[-1]])
            
            if len(predictions) == 0:
                logger.warning("No predictions generated for %s", symbol)
                return {'symbol': symbol, 'action': 'hold', 'confidence': 0.0}
            
            # Get prediction for latest candle
            prediction = predictions[0]
            
            # Get prediction probabilities for confidence
            probabilities = self.model_trainer.predict_proba(df.iloc[[-1]])
            
            if len(probabilities) == 0:
                confidence = 0.5
            else:
                # Get confidence based on model type
                if self.model_trainer.model_type in ['random_forest', 'gradient_boosting']:
                    # For sklearn models, get probability of predicted class
                    # Convert prediction to index (0, 1, 2) for (-1, 0, 1)
                    pred_idx = int(prediction + 1)
                    confidence = probabilities[0][pred_idx]
                else:
                    # For neural network, get probability of predicted class
                    pred_idx = int(prediction + 1)
                    confidence = probabilities[0][pred_idx]
            
            # Map prediction to action
            if prediction == 1:
                action = 'buy'
            elif prediction == -1:
                action = 'sell'
            else:
                action = 'hold'
            
            # Check if confidence meets threshold
            if confidence < config.MODEL_PARAMS['prediction_threshold']:
                action = 'hold'
                logger.debug("Signal confidence %.2f below threshold %.2f, holding",
                           confidence, config.MODEL_PARAMS['prediction_threshold'])
            
            # Check if we already have a position for this symbol
            if symbol in self.positions:
                current_position = self.positions[symbol]
                
                # If we have a long position and signal is sell, close position
                if current_position['side'] == 'long' and action == 'sell':
                    action = 'close'
                    
                # If we have a short position and signal is buy, close position
                elif current_position['side'] == 'short' and action == 'buy':
                    action = 'close'
                    
                # If signal matches current position, hold
                elif (current_position['side'] == 'long' and action == 'buy') or \
                     (current_position['side'] == 'short' and action == 'sell'):
                    action = 'hold'
            
            # Check max open positions
            if action in ['buy', 'sell'] and len(self.positions) >= config.MAX_OPEN_POSITIONS:
                logger.debug("Maximum open positions reached (%d), holding",
                           config.MAX_OPEN_POSITIONS)
                action = 'hold'
            
            # Check max drawdown
            current_drawdown = 1.0 - (self.account_balance / self.initial_balance)
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
                
            if self.max_drawdown > config.MAX_DRAWDOWN_PERCENT:
                logger.warning("Maximum drawdown reached (%.2f%%), stopping trading",
                             self.max_drawdown * 100)
                action = 'hold'
            
            # Calculate position size
            position_size = self._calculate_position_size(symbol, latest_candle['close'])
            
            # Calculate stop loss and take profit levels
            stop_loss, take_profit = self._calculate_exit_levels(
                symbol, action, latest_candle['close']
            )
            
            # Create signal dictionary
            signal = {
                'symbol': symbol,
                'action': action,
                'confidence': float(confidence),
                'price': float(latest_candle['close']),
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Generated signal for %s: %s (confidence: %.2f)",
                      symbol, action, confidence)
            
            return signal
            
        except Exception as e:
            logger.exception("Error generating signals for %s: %s", symbol, str(e))
            return {'symbol': symbol, 'action': 'hold', 'confidence': 0.0}
    
    def update_position(self, symbol: str, position_data: Dict[str, Any]) -> None:
        """
        Update position information.
        
        Args:
            symbol: Trading pair symbol
            position_data: Dictionary with position information
        """
        self.positions[symbol] = position_data
        logger.debug("Updated position for %s: %s", symbol, position_data)
    
    def close_position(self, symbol: str) -> None:
        """
        Close a position.
        
        Args:
            symbol: Trading pair symbol
        """
        if symbol in self.positions:
            del self.positions[symbol]
            logger.debug("Closed position for %s", symbol)
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """
        Calculate position size based on account balance and risk parameters.
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            
        Returns:
            Position size in base currency
        """
        # Calculate position size as percentage of account balance
        position_value = self.account_balance * config.POSITION_SIZE_PERCENT
        
        # Convert to quantity based on price
        quantity = position_value / price
        
        # Round to appropriate precision (this would need to be adjusted per symbol)
        # For simplicity, we'll round to 5 decimal places
        quantity = round(quantity, 5)
        
        return quantity
    
    def _calculate_exit_levels(self, symbol: str, action: str, price: float) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            symbol: Trading pair symbol
            action: Trading action (buy, sell)
            price: Current price
            
        Returns:
            Tuple of (stop_loss, take_profit) prices
        """
        stop_loss = 0.0
        take_profit = 0.0
        
        if action == 'buy':
            # For long positions
            if config.USE_STOP_LOSS:
                stop_loss = price * (1 - config.STOP_LOSS_PERCENT)
            
            if config.USE_TAKE_PROFIT:
                take_profit = price * (1 + config.TAKE_PROFIT_PERCENT)
                
        elif action == 'sell':
            # For short positions
            if config.USE_STOP_LOSS:
                stop_loss = price * (1 + config.STOP_LOSS_PERCENT)
            
            if config.USE_TAKE_PROFIT:
                take_profit = price * (1 - config.TAKE_PROFIT_PERCENT)
        
        # Round to appropriate precision
        stop_loss = round(stop_loss, 2)
        take_profit = round(take_profit, 2)
        
        return stop_loss, take_profit
