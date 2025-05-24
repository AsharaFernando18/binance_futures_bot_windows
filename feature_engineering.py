"""
Feature engineering module for the Binance Futures trading bot.

This module cleans and transforms raw data into features for machine learning,
including technical indicators like EMA crossovers, RSI, MACD, and volume spikes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

import config
import logger

class FeatureEngineering:
    """
    Class for transforming raw OHLCV data into features for ML models.
    """
    
    def __init__(self):
        """Initialize the FeatureEngineering with parameters from config."""
        self.params = config.FEATURE_PARAMS
        logger.info("FeatureEngineering initialized with parameters: %s", self.params)
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw OHLCV data and add technical indicators as features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to feature engineering")
            return df
        
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Add price-based indicators
            result_df = self._add_moving_averages(result_df)
            result_df = self._add_rsi(result_df)
            result_df = self._add_macd(result_df)
            
            # Add volume-based indicators
            result_df = self._add_volume_indicators(result_df)
            
            # Add volatility indicators
            result_df = self._add_bollinger_bands(result_df)
            result_df = self._add_atr(result_df)
            
            # Add momentum indicators
            result_df = self._add_momentum_indicators(result_df)
            
            # Add pattern recognition
            result_df = self._add_candlestick_patterns(result_df)
            
            # Add market structure indicators
            result_df = self._add_support_resistance(result_df)
            
            # Drop NaN values that may have been introduced
            result_df.dropna(inplace=True)
            
            logger.debug("Feature engineering completed, added %d features", 
                       len(result_df.columns) - 5)  # Subtract 5 for OHLCV columns
            
            return result_df
            
        except Exception as e:
            logger.exception("Error in feature engineering: %s", str(e))
            # Return original DataFrame if there's an error
            return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Exponential Moving Averages (EMAs) and their crossovers.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added EMA indicators
        """
        # Add EMAs
        df[f'ema_short'] = df['close'].ewm(span=self.params['ema_short']).mean()
        df[f'ema_medium'] = df['close'].ewm(span=self.params['ema_medium']).mean()
        df[f'ema_long'] = df['close'].ewm(span=self.params['ema_long']).mean()
        
        # Add EMA crossovers
        df['ema_short_over_medium'] = (df['ema_short'] > df['ema_medium']).astype(int)
        df['ema_short_over_long'] = (df['ema_short'] > df['ema_long']).astype(int)
        df['ema_medium_over_long'] = (df['ema_medium'] > df['ema_long']).astype(int)
        
        # Add distance from price to EMAs (normalized)
        df['price_to_ema_short'] = (df['close'] - df['ema_short']) / df['close']
        df['price_to_ema_medium'] = (df['close'] - df['ema_medium']) / df['close']
        df['price_to_ema_long'] = (df['close'] - df['ema_long']) / df['close']
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added RSI indicator
        """
        period = self.params['rsi_period']
        
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Add RSI conditions
        df['rsi_overbought'] = (df['rsi'] > self.params['rsi_overbought']).astype(int)
        df['rsi_oversold'] = (df['rsi'] < self.params['rsi_oversold']).astype(int)
        
        # Add RSI divergence
        df['price_up'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['rsi_up'] = (df['rsi'] > df['rsi'].shift(1)).astype(int)
        df['bullish_divergence'] = ((df['price_up'] == 0) & (df['rsi_up'] == 1)).astype(int)
        df['bearish_divergence'] = ((df['price_up'] == 1) & (df['rsi_up'] == 0)).astype(int)
        
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added MACD indicator
        """
        # Calculate MACD components
        fast_ema = df['close'].ewm(span=self.params['macd_fast']).mean()
        slow_ema = df['close'].ewm(span=self.params['macd_slow']).mean()
        
        # MACD line
        df['macd'] = fast_ema - slow_ema
        
        # Signal line
        df['macd_signal'] = df['macd'].ewm(span=self.params['macd_signal']).mean()
        
        # MACD histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # MACD crossovers
        df['macd_cross_above'] = ((df['macd'] > df['macd_signal']) & 
                                 (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_below'] = ((df['macd'] < df['macd_signal']) & 
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        # MACD histogram direction
        df['macd_hist_up'] = (df['macd_hist'] > df['macd_hist'].shift(1)).astype(int)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added volume indicators
        """
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=self.params['volume_ma_period']).mean()
        
        # Volume ratio (current volume / average volume)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volume spike (volume > 2x average)
        df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
        
        # On-balance volume (OBV)
        df['obv'] = (df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']).cumsum()
        
        # Price-volume trend
        df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Bollinger Bands indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added Bollinger Bands
        """
        # Calculate middle band (20-day SMA)
        period = 20
        std_dev = 2
        
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        df['bb_std'] = df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        
        # Calculate bandwidth
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate %B (position within bands)
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Identify price touching or crossing bands
        df['price_above_upper'] = (df['close'] > df['bb_upper']).astype(int)
        df['price_below_lower'] = (df['close'] < df['bb_lower']).astype(int)
        
        return df
    
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Average True Range (ATR) indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added ATR indicator
        """
        period = 14
        
        # Calculate true range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['true_range'].rolling(window=period).mean()
        
        # Normalize ATR by price
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Clean up temporary columns
        df.drop(['tr1', 'tr2', 'tr3'], axis=1, inplace=True)
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added momentum indicators
        """
        # Rate of Change (ROC)
        df['roc_1'] = df['close'].pct_change(periods=1) * 100
        df['roc_5'] = df['close'].pct_change(periods=5) * 100
        df['roc_10'] = df['close'].pct_change(periods=10) * 100
        
        # Stochastic Oscillator
        period = 14
        k_period = 3
        d_period = 3
        
        # Calculate %K
        df['lowest_low'] = df['low'].rolling(window=period).min()
        df['highest_high'] = df['high'].rolling(window=period).max()
        df['stoch_k'] = 100 * ((df['close'] - df['lowest_low']) / 
                              (df['highest_high'] - df['lowest_low']))
        
        # Calculate %D
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        # Stochastic crossovers
        df['stoch_cross_above'] = ((df['stoch_k'] > df['stoch_d']) & 
                                  (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)
        df['stoch_cross_below'] = ((df['stoch_k'] < df['stoch_d']) & 
                                  (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))).astype(int)
        
        # Clean up temporary columns
        df.drop(['lowest_low', 'highest_high'], axis=1, inplace=True)
        
        return df
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add candlestick pattern recognition.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added candlestick pattern indicators
        """
        # Calculate candle body and shadows
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Doji (small body)
        df['doji'] = (df['body'] < (df['high'] - df['low']) * 0.1).astype(int)
        
        # Hammer (small body, long lower shadow, small upper shadow)
        df['hammer'] = ((df['body'] < (df['high'] - df['low']) * 0.3) & 
                       (df['lower_shadow'] > df['body'] * 2) & 
                       (df['upper_shadow'] < df['body'])).astype(int)
        
        # Shooting Star (small body, long upper shadow, small lower shadow)
        df['shooting_star'] = ((df['body'] < (df['high'] - df['low']) * 0.3) & 
                              (df['upper_shadow'] > df['body'] * 2) & 
                              (df['lower_shadow'] < df['body'])).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = ((df['open'] < df['close']) & 
                                  (df['open'] < df['close'].shift(1)) & 
                                  (df['close'] > df['open'].shift(1)) & 
                                  (df['open'].shift(1) > df['close'].shift(1))).astype(int)
        
        df['bearish_engulfing'] = ((df['open'] > df['close']) & 
                                  (df['open'] > df['close'].shift(1)) & 
                                  (df['close'] < df['open'].shift(1)) & 
                                  (df['open'].shift(1) < df['close'].shift(1))).astype(int)
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add support and resistance level indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added support/resistance indicators
        """
        # Simple pivot-based support/resistance
        window = 5
        
        # Identify local highs and lows
        df['local_high'] = df['high'].rolling(window=window, center=True).apply(
            lambda x: 1 if x[window//2] == max(x) else 0, raw=True
        )
        
        df['local_low'] = df['low'].rolling(window=window, center=True).apply(
            lambda x: 1 if x[window//2] == min(x) else 0, raw=True
        )
        
        # Calculate distance to nearest support/resistance
        # (This is a simplified approach; more sophisticated methods would use actual levels)
        df['near_resistance'] = (df['high'] / df['high'].rolling(window=20).max() > 0.98).astype(int)
        df['near_support'] = (df['low'] / df['low'].rolling(window=20).min() < 1.02).astype(int)
        
        return df
