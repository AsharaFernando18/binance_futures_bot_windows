"""
Data fetcher module for the Binance Futures trading bot.

This module connects to Binance Testnet via WebSocket or REST API
to stream real-time OHLCV and order-book data for the top 3 most
profitable symbols by recent momentum.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import config
import logger

class DataFetcher:
    """
    Class for fetching and streaming market data from Binance Futures.
    """
    
    def __init__(self):
        """Initialize the DataFetcher with Binance Futures exchange connection."""
        self.exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'testnet': config.USE_TESTNET,
                'adjustForTimeDifference': True,
            }
        })
        self.symbols = []
        self.data_streams = {}
        self.ohlcv_data = {}
        self.orderbook_data = {}
        self.running = False
        logger.info("DataFetcher initialized with testnet=%s", config.USE_TESTNET)
    
    async def initialize(self) -> None:
        """Initialize the data fetcher and identify top symbols."""
        try:
            await self.exchange.load_markets()
            logger.info("Markets loaded successfully")
            
            # Identify top symbols by recent momentum
            self.symbols = await self.get_top_symbols(config.TOP_SYMBOLS_COUNT)
            logger.info("Top %d symbols identified: %s", config.TOP_SYMBOLS_COUNT, self.symbols)
            
            # Initialize data structures for each symbol
            for symbol in self.symbols:
                self.ohlcv_data[symbol] = {}
                for timeframe in config.TIMEFRAMES:
                    self.ohlcv_data[symbol][timeframe] = pd.DataFrame()
                self.orderbook_data[symbol] = {}
            
            # Fetch initial historical data
            await self.fetch_historical_data()
            
        except Exception as e:
            logger.exception("Error initializing DataFetcher: %s", str(e))
            raise
    
    async def get_top_symbols(self, count: int = 3) -> List[str]:
        """
        Identify top symbols by recent momentum.
        
        Args:
            count: Number of top symbols to return
            
        Returns:
            List of symbol names
        """
        try:
            # Get all available futures symbols
            markets = self.exchange.markets
            futures_symbols = [symbol for symbol in markets.keys() 
                              if markets[symbol]['future'] and 
                              'USDT' in symbol]
            
            # Fetch recent performance data
            performance_data = []
            
            for symbol in futures_symbols[:20]:  # Limit to first 20 for efficiency
                # Fetch 1-day candles for the past week
                candles = await self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe='1d',
                    limit=7
                )
                
                if len(candles) < 7:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    candles, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Calculate momentum metrics
                price_change_pct = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                volume_change_pct = (df['volume'].iloc[-1] - df['volume'].iloc[0]) / df['volume'].iloc[0]
                volatility = df['close'].pct_change().std()
                
                # Calculate a combined score
                momentum_score = price_change_pct * 0.5 + volume_change_pct * 0.3 + volatility * 0.2
                
                performance_data.append({
                    'symbol': symbol,
                    'price_change_pct': price_change_pct,
                    'volume_change_pct': volume_change_pct,
                    'volatility': volatility,
                    'momentum_score': momentum_score
                })
            
            # Sort by momentum score and get top symbols
            performance_df = pd.DataFrame(performance_data)
            top_symbols = performance_df.sort_values(
                by='momentum_score', 
                ascending=False
            )['symbol'].tolist()[:count]
            
            return top_symbols
            
        except Exception as e:
            logger.exception("Error getting top symbols: %s", str(e))
            # Return some default symbols if there's an error
            return ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    
    async def fetch_historical_data(self) -> None:
        """Fetch historical OHLCV data for all symbols and timeframes."""
        try:
            for symbol in self.symbols:
                for timeframe in config.TIMEFRAMES:
                    logger.info("Fetching historical data for %s (%s)", symbol, timeframe)
                    
                    # Fetch historical data
                    candles = await self.exchange.fetch_ohlcv(
                        symbol, 
                        timeframe=timeframe,
                        limit=config.LOOKBACK_PERIODS
                    )
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(
                        candles, 
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Store in data structure
                    self.ohlcv_data[symbol][timeframe] = df
                    
                    logger.debug("Fetched %d candles for %s (%s)", 
                                len(df), symbol, timeframe)
                    
                # Fetch initial orderbook
                orderbook = await self.exchange.fetch_order_book(symbol)
                self.orderbook_data[symbol] = orderbook
                
                logger.debug("Fetched orderbook for %s with %d bids and %d asks", 
                            symbol, len(orderbook['bids']), len(orderbook['asks']))
                
        except Exception as e:
            logger.exception("Error fetching historical data: %s", str(e))
            raise
    
    async def start_data_streams(self) -> None:
        """Start WebSocket streams for real-time data."""
        self.running = True
        
        # Create tasks for each data stream
        tasks = []
        
        for symbol in self.symbols:
            # OHLCV data stream
            for timeframe in config.TIMEFRAMES:
                task = asyncio.create_task(
                    self._stream_ohlcv(symbol, timeframe)
                )
                tasks.append(task)
            
            # Orderbook data stream
            task = asyncio.create_task(
                self._stream_orderbook(symbol)
            )
            tasks.append(task)
        
        self.data_streams['tasks'] = tasks
        logger.info("Data streams started for %d symbols", len(self.symbols))
    
    async def _stream_ohlcv(self, symbol: str, timeframe: str) -> None:
        """
        Stream OHLCV data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
        """
        try:
            while self.running:
                # Fetch latest candle
                candles = await self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=timeframe,
                    limit=1
                )
                
                if candles and len(candles) > 0:
                    # Convert to DataFrame format
                    candle = candles[0]
                    timestamp = pd.to_datetime(candle[0], unit='ms')
                    
                    # Update the DataFrame
                    if timestamp in self.ohlcv_data[symbol][timeframe].index:
                        # Update existing candle
                        self.ohlcv_data[symbol][timeframe].loc[timestamp] = [
                            candle[1], candle[2], candle[3], candle[4], candle[5]
                        ]
                    else:
                        # Add new candle
                        new_row = pd.DataFrame(
                            [[candle[1], candle[2], candle[3], candle[4], candle[5]]],
                            columns=['open', 'high', 'low', 'close', 'volume'],
                            index=[timestamp]
                        )
                        self.ohlcv_data[symbol][timeframe] = pd.concat(
                            [self.ohlcv_data[symbol][timeframe], new_row]
                        )
                        
                        # Trim old data
                        if len(self.ohlcv_data[symbol][timeframe]) > config.LOOKBACK_PERIODS:
                            self.ohlcv_data[symbol][timeframe] = self.ohlcv_data[symbol][timeframe].iloc[
                                -config.LOOKBACK_PERIODS:
                            ]
                
                # Sleep based on timeframe
                timeframe_seconds = self._timeframe_to_seconds(timeframe)
                await asyncio.sleep(min(timeframe_seconds / 4, 15))  # Update at least every 15 seconds
                
        except Exception as e:
            logger.exception("Error in OHLCV stream for %s (%s): %s", 
                           symbol, timeframe, str(e))
            
            # Try to reconnect
            await asyncio.sleep(5)
            if self.running:
                asyncio.create_task(self._stream_ohlcv(symbol, timeframe))
    
    async def _stream_orderbook(self, symbol: str) -> None:
        """
        Stream orderbook data for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
        """
        try:
            while self.running:
                # Fetch latest orderbook
                orderbook = await self.exchange.fetch_order_book(symbol)
                
                # Update the orderbook data
                self.orderbook_data[symbol] = orderbook
                
                # Sleep before next update
                await asyncio.sleep(1)  # Update orderbook every second
                
        except Exception as e:
            logger.exception("Error in orderbook stream for %s: %s", 
                           symbol, str(e))
            
            # Try to reconnect
            await asyncio.sleep(5)
            if self.running:
                asyncio.create_task(self._stream_orderbook(symbol))
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """
        Convert timeframe string to seconds.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '15m', '1h')
            
        Returns:
            Number of seconds in the timeframe
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 24 * 60 * 60
        else:
            return 60  # Default to 1 minute
    
    def get_current_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Get the current OHLCV data for a symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            
        Returns:
            DataFrame with OHLCV data
        """
        if symbol in self.ohlcv_data and timeframe in self.ohlcv_data[symbol]:
            return self.ohlcv_data[symbol][timeframe].copy()
        return pd.DataFrame()
    
    def get_current_orderbook(self, symbol: str) -> Dict:
        """
        Get the current orderbook for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with orderbook data
        """
        if symbol in self.orderbook_data:
            return self.orderbook_data[symbol].copy()
        return {'bids': [], 'asks': []}
    
    async def stop(self) -> None:
        """Stop all data streams and close connections."""
        self.running = False
        
        # Cancel all tasks
        if 'tasks' in self.data_streams:
            for task in self.data_streams['tasks']:
                task.cancel()
        
        # Close exchange connection
        await self.exchange.close()
        
        logger.info("DataFetcher stopped")
