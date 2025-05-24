"""
Order execution module for the Binance Futures trading bot.

This module places and manages orders on Binance Testnet via CCXT,
monitors fills, handles partial fills, and adjusts stops.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
import ccxt.async_support as ccxt
from datetime import datetime

import config
import logger

class Executor:
    """
    Class for executing trades on Binance Futures.
    """
    
    def __init__(self):
        """Initialize the Executor with Binance Futures exchange connection."""
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
        self.open_orders = {}
        self.positions = {}
        self.account_balance = 0.0
        
        logger.info("Executor initialized with testnet=%s", config.USE_TESTNET)
    
    async def initialize(self) -> None:
        """Initialize the executor and fetch account information."""
        try:
            await self.exchange.load_markets()
            logger.info("Markets loaded successfully")
            
            # Fetch account balance
            await self.update_account_info()
            
        except Exception as e:
            logger.exception("Error initializing Executor: %s", str(e))
            raise
    
    async def update_account_info(self) -> None:
        """Update account balance and positions information."""
        try:
            # Fetch account balance
            balance = await self.exchange.fetch_balance()
            
            # Get USDT balance
            if 'USDT' in balance['total']:
                self.account_balance = balance['total']['USDT']
                logger.debug("Account balance: %.2f USDT", self.account_balance)
            
            # Fetch open positions
            positions = await self.exchange.fetch_positions()
            
            # Update positions dictionary
            self.positions = {}
            for position in positions:
                if float(position['contracts']) > 0:
                    symbol = position['symbol']
                    self.positions[symbol] = {
                        'side': 'long' if position['side'] == 'long' else 'short',
                        'size': float(position['contracts']),
                        'entry_price': float(position['entryPrice']),
                        'unrealized_pnl': float(position['unrealizedPnl']),
                        'liquidation_price': float(position['liquidationPrice']) if 'liquidationPrice' in position else 0.0
                    }
            
            logger.debug("Updated account info: %.2f USDT, %d open positions",
                       self.account_balance, len(self.positions))
            
        except Exception as e:
            logger.exception("Error updating account info: %s", str(e))
    
    async def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading signal.
        
        Args:
            signal: Dictionary with signal information
            
        Returns:
            Dictionary with execution results
        """
        try:
            symbol = signal['symbol']
            action = signal['action']
            
            # Skip if action is hold
            if action == 'hold':
                return {'success': True, 'action': 'hold', 'message': 'No action taken'}
            
            # Update account info before executing
            await self.update_account_info()
            
            # Execute based on action
            if action == 'buy':
                return await self._open_long_position(signal)
                
            elif action == 'sell':
                return await self._open_short_position(signal)
                
            elif action == 'close':
                return await self._close_position(signal)
                
            else:
                logger.warning("Unknown action: %s", action)
                return {'success': False, 'action': action, 'message': f'Unknown action: {action}'}
                
        except Exception as e:
            logger.exception("Error executing signal: %s", str(e))
            return {'success': False, 'action': signal['action'], 'message': str(e)}
    
    async def _open_long_position(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Open a long position.
        
        Args:
            signal: Dictionary with signal information
            
        Returns:
            Dictionary with execution results
        """
        symbol = signal['symbol']
        price = signal['price']
        position_size = signal['position_size']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        
        try:
            # Place market order
            order = await self.exchange.create_market_buy_order(
                symbol=symbol,
                amount=position_size
            )
            
            logger.info("Opened long position for %s: %.5f @ market", 
                      symbol, position_size)
            
            # Set stop loss if enabled
            stop_order_id = None
            if config.USE_STOP_LOSS and stop_loss > 0:
                stop_order = await self.exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side='sell',
                    amount=position_size,
                    price=stop_loss,
                    params={'stopPrice': stop_loss}
                )
                stop_order_id = stop_order['id']
                logger.info("Set stop loss for %s at %.2f", symbol, stop_loss)
            
            # Set take profit if enabled
            take_profit_order_id = None
            if config.USE_TAKE_PROFIT and take_profit > 0:
                take_profit_order = await self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side='sell',
                    amount=position_size,
                    price=take_profit
                )
                take_profit_order_id = take_profit_order['id']
                logger.info("Set take profit for %s at %.2f", symbol, take_profit)
            
            # Update positions
            await self.update_account_info()
            
            # Store order information
            self.open_orders[symbol] = {
                'order_id': order['id'],
                'stop_order_id': stop_order_id,
                'take_profit_order_id': take_profit_order_id,
                'side': 'long',
                'size': position_size,
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'success': True,
                'action': 'buy',
                'message': f'Opened long position for {symbol}: {position_size} @ market',
                'order_id': order['id']
            }
            
        except Exception as e:
            logger.exception("Error opening long position for %s: %s", symbol, str(e))
            return {
                'success': False,
                'action': 'buy',
                'message': f'Error opening long position: {str(e)}'
            }
    
    async def _open_short_position(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Open a short position.
        
        Args:
            signal: Dictionary with signal information
            
        Returns:
            Dictionary with execution results
        """
        symbol = signal['symbol']
        price = signal['price']
        position_size = signal['position_size']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        
        try:
            # Place market order
            order = await self.exchange.create_market_sell_order(
                symbol=symbol,
                amount=position_size
            )
            
            logger.info("Opened short position for %s: %.5f @ market", 
                      symbol, position_size)
            
            # Set stop loss if enabled
            stop_order_id = None
            if config.USE_STOP_LOSS and stop_loss > 0:
                stop_order = await self.exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side='buy',
                    amount=position_size,
                    price=stop_loss,
                    params={'stopPrice': stop_loss}
                )
                stop_order_id = stop_order['id']
                logger.info("Set stop loss for %s at %.2f", symbol, stop_loss)
            
            # Set take profit if enabled
            take_profit_order_id = None
            if config.USE_TAKE_PROFIT and take_profit > 0:
                take_profit_order = await self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side='buy',
                    amount=position_size,
                    price=take_profit
                )
                take_profit_order_id = take_profit_order['id']
                logger.info("Set take profit for %s at %.2f", symbol, take_profit)
            
            # Update positions
            await self.update_account_info()
            
            # Store order information
            self.open_orders[symbol] = {
                'order_id': order['id'],
                'stop_order_id': stop_order_id,
                'take_profit_order_id': take_profit_order_id,
                'side': 'short',
                'size': position_size,
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'success': True,
                'action': 'sell',
                'message': f'Opened short position for {symbol}: {position_size} @ market',
                'order_id': order['id']
            }
            
        except Exception as e:
            logger.exception("Error opening short position for %s: %s", symbol, str(e))
            return {
                'success': False,
                'action': 'sell',
                'message': f'Error opening short position: {str(e)}'
            }
    
    async def _close_position(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Close an open position.
        
        Args:
            signal: Dictionary with signal information
            
        Returns:
            Dictionary with execution results
        """
        symbol = signal['symbol']
        
        try:
            # Check if we have an open position
            if symbol not in self.positions:
                logger.warning("No open position found for %s", symbol)
                return {
                    'success': False,
                    'action': 'close',
                    'message': f'No open position found for {symbol}'
                }
            
            position = self.positions[symbol]
            side = position['side']
            size = position['size']
            
            # Close position based on side
            if side == 'long':
                order = await self.exchange.create_market_sell_order(
                    symbol=symbol,
                    amount=size
                )
                logger.info("Closed long position for %s: %.5f @ market", 
                          symbol, size)
                
            else:  # short
                order = await self.exchange.create_market_buy_order(
                    symbol=symbol,
                    amount=size
                )
                logger.info("Closed short position for %s: %.5f @ market", 
                          symbol, size)
            
            # Cancel any open orders for this symbol
            if symbol in self.open_orders:
                order_info = self.open_orders[symbol]
                
                # Cancel stop loss order
                if order_info['stop_order_id']:
                    try:
                        await self.exchange.cancel_order(
                            id=order_info['stop_order_id'],
                            symbol=symbol
                        )
                        logger.debug("Cancelled stop loss order for %s", symbol)
                    except Exception as e:
                        logger.warning("Error cancelling stop loss order: %s", str(e))
                
                # Cancel take profit order
                if order_info['take_profit_order_id']:
                    try:
                        await self.exchange.cancel_order(
                            id=order_info['take_profit_order_id'],
                            symbol=symbol
                        )
                        logger.debug("Cancelled take profit order for %s", symbol)
                    except Exception as e:
                        logger.warning("Error cancelling take profit order: %s", str(e))
                
                # Remove from open orders
                del self.open_orders[symbol]
            
            # Update positions
            await self.update_account_info()
            
            return {
                'success': True,
                'action': 'close',
                'message': f'Closed {side} position for {symbol}: {size} @ market',
                'order_id': order['id']
            }
            
        except Exception as e:
            logger.exception("Error closing position for %s: %s", symbol, str(e))
            return {
                'success': False,
                'action': 'close',
                'message': f'Error closing position: {str(e)}'
            }
    
    async def check_order_status(self) -> None:
        """Check the status of open orders and update trailing stops if needed."""
        try:
            # Update account info
            await self.update_account_info()
            
            # Check each open order
            for symbol, order_info in list(self.open_orders.items()):
                # Skip if symbol is not in positions (already closed)
                if symbol not in self.positions:
                    logger.debug("Position for %s already closed, removing from open orders", symbol)
                    del self.open_orders[symbol]
                    continue
                
                # Get current position
                position = self.positions[symbol]
                
                # Check if trailing stop is enabled
                if config.TRAILING_STOP:
                    # Fetch current market price
                    ticker = await self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    # Update trailing stop for long positions
                    if position['side'] == 'long':
                        # Calculate new stop loss level
                        new_stop_loss = current_price * (1 - config.TRAILING_STOP_PERCENT)
                        
                        # Only update if new stop loss is higher than current
                        if new_stop_loss > order_info['stop_loss']:
                            # Cancel existing stop loss order
                            if order_info['stop_order_id']:
                                try:
                                    await self.exchange.cancel_order(
                                        id=order_info['stop_order_id'],
                                        symbol=symbol
                                    )
                                    logger.debug("Cancelled old stop loss order for %s", symbol)
                                except Exception as e:
                                    logger.warning("Error cancelling old stop loss order: %s", str(e))
                            
                            # Create new stop loss order
                            stop_order = await self.exchange.create_order(
                                symbol=symbol,
                                type='stop',
                                side='sell',
                                amount=position['size'],
                                price=new_stop_loss,
                                params={'stopPrice': new_stop_loss}
                            )
                            
                            # Update order info
                            order_info['stop_order_id'] = stop_order['id']
                            order_info['stop_loss'] = new_stop_loss
                            
                            logger.info("Updated trailing stop for %s to %.2f", 
                                      symbol, new_stop_loss)
                    
                    # Update trailing stop for short positions
                    elif position['side'] == 'short':
                        # Calculate new stop loss level
                        new_stop_loss = current_price * (1 + config.TRAILING_STOP_PERCENT)
                        
                        # Only update if new stop loss is lower than current
                        if new_stop_loss < order_info['stop_loss']:
                            # Cancel existing stop loss order
                            if order_info['stop_order_id']:
                                try:
                                    await self.exchange.cancel_order(
                                        id=order_info['stop_order_id'],
                                        symbol=symbol
                                    )
                                    logger.debug("Cancelled old stop loss order for %s", symbol)
                                except Exception as e:
                                    logger.warning("Error cancelling old stop loss order: %s", str(e))
                            
                            # Create new stop loss order
                            stop_order = await self.exchange.create_order(
                                symbol=symbol,
                                type='stop',
                                side='buy',
                                amount=position['size'],
                                price=new_stop_loss,
                                params={'stopPrice': new_stop_loss}
                            )
                            
                            # Update order info
                            order_info['stop_order_id'] = stop_order['id']
                            order_info['stop_loss'] = new_stop_loss
                            
                            logger.info("Updated trailing stop for %s to %.2f", 
                                      symbol, new_stop_loss)
            
        except Exception as e:
            logger.exception("Error checking order status: %s", str(e))
    
    async def get_account_balance(self) -> float:
        """
        Get the current account balance.
        
        Returns:
            Current account balance in USDT
        """
        await self.update_account_info()
        return self.account_balance
    
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            Dictionary of open positions
        """
        await self.update_account_info()
        return self.positions
    
    async def get_position_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all open positions.
        
        Returns:
            Dictionary with position summary
        """
        await self.update_account_info()
        
        total_pnl = 0.0
        position_count = len(self.positions)
        
        for symbol, position in self.positions.items():
            total_pnl += position['unrealized_pnl']
        
        return {
            'balance': self.account_balance,
            'position_count': position_count,
            'total_unrealized_pnl': total_pnl,
            'positions': self.positions
        }
    
    async def stop(self) -> None:
        """Close all positions and stop the executor."""
        try:
            # Close all open positions
            for symbol in list(self.positions.keys()):
                signal = {'symbol': symbol, 'action': 'close'}
                await self._close_position(signal)
            
            # Close exchange connection
            await self.exchange.close()
            
            logger.info("Executor stopped, all positions closed")
            
        except Exception as e:
            logger.exception("Error stopping executor: %s", str(e))
