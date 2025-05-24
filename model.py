"""
Machine learning model module for the Binance Futures trading bot.

This module implements ML pipeline to train or load a pre-trained model
for signal classification or regression.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM

import config
import logger

class ModelTrainer:
    """
    Class for training and using ML models for trading signal generation.
    """
    
    def __init__(self):
        """Initialize the ModelTrainer with parameters from config."""
        self.model_type = config.MODEL_PARAMS['model_type']
        self.model_params = config.MODEL_PARAMS
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.last_train_time = None
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        logger.info("ModelTrainer initialized with model type: %s", self.model_type)
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for model training or prediction.
        
        Args:
            df: DataFrame with features
            target_column: Name of the target column for training (None for prediction)
            
        Returns:
            Tuple of (X, y) where X is feature array and y is target array (or None for prediction)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to model")
            return np.array([]), None
        
        try:
            # Drop non-feature columns
            feature_df = df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
            
            # If target column exists, separate it
            y = None
            if target_column and target_column in feature_df.columns:
                y = feature_df[target_column].values
                feature_df = feature_df.drop(target_column, axis=1)
            
            # Store feature columns for future reference
            self.feature_columns = feature_df.columns.tolist()
            
            # Scale features
            X = self.scaler.fit_transform(feature_df)
            
            return X, y
            
        except Exception as e:
            logger.exception("Error preparing data for model: %s", str(e))
            return np.array([]), None
    
    def generate_target(self, df: pd.DataFrame, lookahead_periods: int = 3, threshold_pct: float = 0.01) -> pd.DataFrame:
        """
        Generate target variable for supervised learning.
        
        Args:
            df: DataFrame with OHLCV data
            lookahead_periods: Number of periods to look ahead for price movement
            threshold_pct: Price movement threshold for signal generation
            
        Returns:
            DataFrame with added target column
        """
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Calculate future returns
            result_df['future_return'] = result_df['close'].shift(-lookahead_periods) / result_df['close'] - 1
            
            # Generate target signals based on threshold
            result_df['target'] = 0  # Default to hold
            result_df.loc[result_df['future_return'] > threshold_pct, 'target'] = 1  # Buy signal
            result_df.loc[result_df['future_return'] < -threshold_pct, 'target'] = -1  # Sell signal
            
            # Drop NaN values created by the shift
            result_df.dropna(inplace=True)
            
            logger.debug("Generated target variable with %d buy, %d sell, and %d hold signals",
                       (result_df['target'] == 1).sum(),
                       (result_df['target'] == -1).sum(),
                       (result_df['target'] == 0).sum())
            
            return result_df
            
        except Exception as e:
            logger.exception("Error generating target variable: %s", str(e))
            return df
    
    def train(self, df: pd.DataFrame, target_column: str = 'target') -> bool:
        """
        Train the ML model on the provided data.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of the target column
            
        Returns:
            True if training was successful, False otherwise
        """
        try:
            logger.info("Training %s model...", self.model_type)
            
            # Prepare data
            X, y = self.prepare_data(df, target_column)
            
            if len(X) == 0 or y is None:
                logger.error("No data available for training")
                return False
            
            # Split data into train and test sets
            test_size = 1.0 - self.model_params['train_test_split']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            # Train model based on type
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.model.fit(X_train, y_train)
                
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                self.model.fit(X_train, y_train)
                
            elif self.model_type == 'neural_network':
                # For simplicity, we'll use a basic neural network
                self.model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(3, activation='softmax')  # 3 classes: buy, hold, sell
                ])
                
                self.model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Convert target to 0, 1, 2 for softmax
                y_train_nn = y_train + 1  # -1 -> 0, 0 -> 1, 1 -> 2
                y_test_nn = y_test + 1
                
                self.model.fit(
                    X_train, y_train_nn,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test_nn),
                    verbose=0
                )
            
            # Evaluate model
            self._evaluate_model(X_test, y_test)
            
            # Save model
            self._save_model()
            
            # Update last train time
            self.last_train_time = datetime.now()
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.exception("Error training model: %s", str(e))
            return False
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of predictions (-1 for sell, 0 for hold, 1 for buy)
        """
        try:
            if self.model is None:
                logger.warning("Model not trained, attempting to load saved model")
                if not self._load_model():
                    logger.error("No trained model available for prediction")
                    return np.array([])
            
            # Prepare data
            X, _ = self.prepare_data(df)
            
            if len(X) == 0:
                logger.warning("No data available for prediction")
                return np.array([])
            
            # Generate predictions based on model type
            if self.model_type in ['random_forest', 'gradient_boosting']:
                predictions = self.model.predict(X)
                
            elif self.model_type == 'neural_network':
                # Neural network outputs probabilities for each class
                probabilities = self.model.predict(X)
                # Convert from 0,1,2 back to -1,0,1
                predictions = np.argmax(probabilities, axis=1) - 1
            
            logger.debug("Generated %d predictions: %d buy, %d sell, %d hold",
                       len(predictions),
                       (predictions == 1).sum(),
                       (predictions == -1).sum(),
                       (predictions == 0).sum())
            
            return predictions
            
        except Exception as e:
            logger.exception("Error generating predictions: %s", str(e))
            return np.array([])
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate prediction probabilities using the trained model.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of prediction probabilities for each class
        """
        try:
            if self.model is None:
                logger.warning("Model not trained, attempting to load saved model")
                if not self._load_model():
                    logger.error("No trained model available for prediction")
                    return np.array([])
            
            # Prepare data
            X, _ = self.prepare_data(df)
            
            if len(X) == 0:
                logger.warning("No data available for prediction")
                return np.array([])
            
            # Generate prediction probabilities based on model type
            if self.model_type in ['random_forest', 'gradient_boosting']:
                # For sklearn models, probabilities are for each class
                probabilities = self.model.predict_proba(X)
                
            elif self.model_type == 'neural_network':
                # Neural network outputs probabilities for each class
                probabilities = self.model.predict(X)
            
            return probabilities
            
        except Exception as e:
            logger.exception("Error generating prediction probabilities: %s", str(e))
            return np.array([])
    
    def should_retrain(self) -> bool:
        """
        Check if the model should be retrained based on the configured interval.
        
        Returns:
            True if model should be retrained, False otherwise
        """
        if self.last_train_time is None:
            return True
        
        retrain_interval = timedelta(hours=self.model_params['retrain_interval_hours'])
        return datetime.now() - self.last_train_time > retrain_interval
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
        """
        try:
            if self.model_type in ['random_forest', 'gradient_boosting']:
                y_pred = self.model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # For multi-class, we need to specify average method
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
            elif self.model_type == 'neural_network':
                # Convert target to 0, 1, 2 for softmax
                y_test_nn = y_test + 1  # -1 -> 0, 0 -> 1, 1 -> 2
                
                # Evaluate model
                loss, accuracy = self.model.evaluate(X_test, y_test_nn, verbose=0)
                
                # For simplicity, we'll just use accuracy for neural networks
                precision = recall = f1 = 0.0
            
            logger.info("Model evaluation metrics:")
            logger.info("  Accuracy: %.4f", accuracy)
            logger.info("  Precision: %.4f", precision)
            logger.info("  Recall: %.4f", recall)
            logger.info("  F1 Score: %.4f", f1)
            
        except Exception as e:
            logger.exception("Error evaluating model: %s", str(e))
    
    def _save_model(self) -> bool:
        """
        Save the trained model to disk.
        
        Returns:
            True if model was saved successfully, False otherwise
        """
        try:
            model_path = f"models/{self.model_type}_model"
            
            if self.model_type in ['random_forest', 'gradient_boosting']:
                # Save sklearn model using pickle
                with open(f"{model_path}.pkl", 'wb') as f:
                    pickle.dump(self.model, f)
                
                # Save scaler
                with open(f"{model_path}_scaler.pkl", 'wb') as f:
                    pickle.dump(self.scaler, f)
                
                # Save feature columns
                with open(f"{model_path}_features.pkl", 'wb') as f:
                    pickle.dump(self.feature_columns, f)
                
            elif self.model_type == 'neural_network':
                # Save Keras model
                self.model.save(f"{model_path}.h5")
                
                # Save scaler
                with open(f"{model_path}_scaler.pkl", 'wb') as f:
                    pickle.dump(self.scaler, f)
                
                # Save feature columns
                with open(f"{model_path}_features.pkl", 'wb') as f:
                    pickle.dump(self.feature_columns, f)
            
            logger.info("Model saved to %s", model_path)
            return True
            
        except Exception as e:
            logger.exception("Error saving model: %s", str(e))
            return False
    
    def _load_model(self) -> bool:
        """
        Load a trained model from disk.
        
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            model_path = f"models/{self.model_type}_model"
            
            if self.model_type in ['random_forest', 'gradient_boosting']:
                # Check if model file exists
                if not os.path.exists(f"{model_path}.pkl"):
                    logger.warning("No saved model found at %s.pkl", model_path)
                    return False
                
                # Load sklearn model
                with open(f"{model_path}.pkl", 'rb') as f:
                    self.model = pickle.load(f)
                
                # Load scaler
                if os.path.exists(f"{model_path}_scaler.pkl"):
                    with open(f"{model_path}_scaler.pkl", 'rb') as f:
                        self.scaler = pickle.load(f)
                
                # Load feature columns
                if os.path.exists(f"{model_path}_features.pkl"):
                    with open(f"{model_path}_features.pkl", 'rb') as f:
                        self.feature_columns = pickle.load(f)
                
            elif self.model_type == 'neural_network':
                # Check if model file exists
                if not os.path.exists(f"{model_path}.h5"):
                    logger.warning("No saved model found at %s.h5", model_path)
                    return False
                
                # Load Keras model
                self.model = load_model(f"{model_path}.h5")
                
                # Load scaler
                if os.path.exists(f"{model_path}_scaler.pkl"):
                    with open(f"{model_path}_scaler.pkl", 'rb') as f:
                        self.scaler = pickle.load(f)
                
                # Load feature columns
                if os.path.exists(f"{model_path}_features.pkl"):
                    with open(f"{model_path}_features.pkl", 'rb') as f:
                        self.feature_columns = pickle.load(f)
            
            logger.info("Model loaded from %s", model_path)
            return True
            
        except Exception as e:
            logger.exception("Error loading model: %s", str(e))
            return False
    
    def backtest(self, df: pd.DataFrame, target_column: str = 'target') -> Dict[str, float]:
        """
        Backtest the model on historical data.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of the target column
            
        Returns:
            Dictionary with backtest metrics
        """
        try:
            logger.info("Backtesting model...")
            
            # Prepare data
            X, y = self.prepare_data(df, target_column)
            
            if len(X) == 0 or y is None:
                logger.error("No data available for backtesting")
                return {}
            
            # Generate predictions
            if self.model_type in ['random_forest', 'gradient_boosting']:
                y_pred = self.model.predict(X)
                
            elif self.model_type == 'neural_network':
                # Convert target to 0, 1, 2 for softmax
                y_nn = y + 1  # -1 -> 0, 0 -> 1, 1 -> 2
                
                # Generate predictions
                probabilities = self.model.predict(X)
                y_pred = np.argmax(probabilities, axis=1) - 1
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            
            # Calculate trading metrics
            # Assuming df has 'close' column and is aligned with X and y
            df_backtest = df.iloc[-len(y):].copy()
            df_backtest['prediction'] = y_pred
            
            # Calculate returns based on predictions
            df_backtest['position'] = df_backtest['prediction'].shift(1)
            df_backtest['returns'] = df_backtest['close'].pct_change()
            df_backtest['strategy_returns'] = df_backtest['position'] * df_backtest['returns']
            
            # Calculate cumulative returns
            df_backtest['cumulative_returns'] = (1 + df_backtest['returns']).cumprod() - 1
            df_backtest['cumulative_strategy_returns'] = (1 + df_backtest['strategy_returns']).cumprod() - 1
            
            # Calculate win rate
            df_backtest['win'] = (df_backtest['strategy_returns'] > 0).astype(int)
            win_rate = df_backtest['win'].mean()
            
            # Calculate profit factor
            gross_profit = df_backtest.loc[df_backtest['strategy_returns'] > 0, 'strategy_returns'].sum()
            gross_loss = abs(df_backtest.loc[df_backtest['strategy_returns'] < 0, 'strategy_returns'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Calculate Sharpe ratio
            sharpe_ratio = df_backtest['strategy_returns'].mean() / df_backtest['strategy_returns'].std() * np.sqrt(252)
            
            # Calculate max drawdown
            df_backtest['peak'] = df_backtest['cumulative_strategy_returns'].cummax()
            df_backtest['drawdown'] = df_backtest['peak'] - df_backtest['cumulative_strategy_returns']
            max_drawdown = df_backtest['drawdown'].max()
            
            # Compile metrics
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_return': df_backtest['cumulative_strategy_returns'].iloc[-1]
            }
            
            logger.info("Backtest results:")
            logger.info("  Accuracy: %.4f", accuracy)
            logger.info("  Win Rate: %.4f", win_rate)
            logger.info("  Profit Factor: %.4f", profit_factor)
            logger.info("  Sharpe Ratio: %.4f", sharpe_ratio)
            logger.info("  Max Drawdown: %.4f", max_drawdown)
            logger.info("  Final Return: %.4f", metrics['final_return'])
            
            return metrics
            
        except Exception as e:
            logger.exception("Error backtesting model: %s", str(e))
            return {}
