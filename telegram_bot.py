"""
Telegram bot module for the Binance Futures trading bot.

This module sends real-time alerts (orders placed/executed, P&L updates,
critical errors) to a Telegram chat.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

import config
import logger

class TelegramBot:
    """
    Class for sending notifications to Telegram.
    """
    
    def __init__(self):
        """Initialize the TelegramBot with credentials from config."""
        self.token = config.TELEGRAM_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.enabled = bool(self.token and self.chat_id)
        
        if self.enabled:
            logger.info("TelegramBot initialized")
        else:
            logger.warning("TelegramBot disabled: missing token or chat_id")
    
    async def send_message(self, message: str) -> bool:
        """
        Send a message to the Telegram chat.
        
        Args:
            message: Message text to send
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("TelegramBot disabled, message not sent: %s", message)
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        logger.debug("Telegram message sent successfully")
                        return True
                    else:
                        response_text = await response.text()
                        logger.error("Failed to send Telegram message: %s", response_text)
                        return False
                        
        except Exception as e:
            logger.exception("Error sending Telegram message: %s", str(e))
            return False
    
    async def send_trade_notification(self, signal: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """
        Send a notification about a trade execution.
        
        Args:
            signal: Dictionary with signal information
            result: Dictionary with execution results
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        try:
            symbol = signal['symbol']
            action = signal['action']
            success = result['success']
            message = result['message']
            
            # Format message based on action and success
            if success:
                if action == 'buy':
                    notification = (
                        f"🟢 *LONG POSITION OPENED*\n"
                        f"Symbol: `{symbol}`\n"
                        f"Size: `{signal['position_size']}`\n"
                        f"Price: `{signal['price']}`\n"
                        f"Stop Loss: `{signal['stop_loss']}`\n"
                        f"Take Profit: `{signal['take_profit']}`\n"
                        f"Confidence: `{signal['confidence']:.2f}`\n"
                        f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
                    )
                elif action == 'sell':
                    notification = (
                        f"🔴 *SHORT POSITION OPENED*\n"
                        f"Symbol: `{symbol}`\n"
                        f"Size: `{signal['position_size']}`\n"
                        f"Price: `{signal['price']}`\n"
                        f"Stop Loss: `{signal['stop_loss']}`\n"
                        f"Take Profit: `{signal['take_profit']}`\n"
                        f"Confidence: `{signal['confidence']:.2f}`\n"
                        f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
                    )
                elif action == 'close':
                    notification = (
                        f"⚪ *POSITION CLOSED*\n"
                        f"Symbol: `{symbol}`\n"
                        f"Result: `{message}`\n"
                        f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
                    )
                else:
                    notification = (
                        f"ℹ️ *TRADE NOTIFICATION*\n"
                        f"Symbol: `{symbol}`\n"
                        f"Action: `{action}`\n"
                        f"Result: `{message}`\n"
                        f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
                    )
            else:
                notification = (
                    f"⚠️ *TRADE ERROR*\n"
                    f"Symbol: `{symbol}`\n"
                    f"Action: `{action}`\n"
                    f"Error: `{message}`\n"
                    f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
                )
            
            return await self.send_message(notification)
            
        except Exception as e:
            logger.exception("Error sending trade notification: %s", str(e))
            return False
    
    async def send_balance_update(self, balance_info: Dict[str, Any]) -> bool:
        """
        Send a notification with account balance and P&L information.
        
        Args:
            balance_info: Dictionary with balance information
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        try:
            balance = balance_info['balance']
            position_count = balance_info['position_count']
            total_pnl = balance_info['total_unrealized_pnl']
            
            # Format message
            notification = (
                f"💰 *ACCOUNT UPDATE*\n"
                f"Balance: `{balance:.2f} USDT`\n"
                f"Open Positions: `{position_count}`\n"
                f"Unrealized P&L: `{total_pnl:.2f} USDT`\n"
                f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            )
            
            # Add position details if there are any
            if position_count > 0:
                notification += "\n\n*Open Positions:*"
                for symbol, position in balance_info['positions'].items():
                    side = position['side']
                    size = position['size']
                    entry_price = position['entry_price']
                    unrealized_pnl = position['unrealized_pnl']
                    
                    notification += f"\n`{symbol}`: "
                    if side == 'long':
                        notification += "🟢 LONG"
                    else:
                        notification += "🔴 SHORT"
                    
                    notification += f" | Size: `{size}` | Entry: `{entry_price}` | PnL: `{unrealized_pnl:.2f} USDT`"
            
            return await self.send_message(notification)
            
        except Exception as e:
            logger.exception("Error sending balance update: %s", str(e))
            return False
    
    async def send_error_notification(self, error_message: str) -> bool:
        """
        Send a notification about a critical error.
        
        Args:
            error_message: Error message to send
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        try:
            notification = (
                f"🚨 *CRITICAL ERROR*\n"
                f"Error: `{error_message}`\n"
                f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            )
            
            return await self.send_message(notification)
            
        except Exception as e:
            logger.exception("Error sending error notification: %s", str(e))
            return False
    
    async def send_startup_notification(self, symbols: List[str]) -> bool:
        """
        Send a notification when the bot starts.
        
        Args:
            symbols: List of symbols being traded
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        try:
            symbols_str = ", ".join([f"`{s}`" for s in symbols])
            
            notification = (
                f"🚀 *TRADING BOT STARTED*\n"
                f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
                f"Trading Symbols: {symbols_str}\n"
                f"Runtime: `{config.RUNTIME_HOURS} hours`\n"
                f"Expected End: `{config.END_TIME.strftime('%Y-%m-%d %H:%M:%S')}`"
            )
            
            return await self.send_message(notification)
            
        except Exception as e:
            logger.exception("Error sending startup notification: %s", str(e))
            return False
    
    async def send_shutdown_notification(self, runtime_minutes: float, total_trades: int, final_balance: float, initial_balance: float) -> bool:
        """
        Send a notification when the bot shuts down.
        
        Args:
            runtime_minutes: Total runtime in minutes
            total_trades: Total number of trades executed
            final_balance: Final account balance
            initial_balance: Initial account balance
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        try:
            profit = final_balance - initial_balance
            profit_percent = (profit / initial_balance) * 100 if initial_balance > 0 else 0
            
            notification = (
                f"🔚 *TRADING BOT SHUTDOWN*\n"
                f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
                f"Runtime: `{runtime_minutes:.1f} minutes`\n"
                f"Total Trades: `{total_trades}`\n"
                f"Initial Balance: `{initial_balance:.2f} USDT`\n"
                f"Final Balance: `{final_balance:.2f} USDT`\n"
                f"Profit/Loss: `{profit:.2f} USDT ({profit_percent:.2f}%)`"
            )
            
            return await self.send_message(notification)
            
        except Exception as e:
            logger.exception("Error sending shutdown notification: %s", str(e))
            return False
