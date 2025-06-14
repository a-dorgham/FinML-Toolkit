import pandas as pd
import numpy as np
import tensorflow as tf # Assuming tensorflow.keras.Model is used
from datetime import timedelta
from typing import Tuple, Dict

# Assuming ErrorHandler and MLToolkitError are defined in ml_toolkit/error_handler.py
from ml_toolkit.error_handler import ErrorHandler


class ModelForecaster:
    """
    Handles forecasting future values using trained machine learning models,
    especially designed for time series models like LSTMs.
    """

    def __init__(self):
        """
        Initializes the ModelForecaster.
        """
        pass

    def forecast_LSTM(self, model: tf.keras.Model, df: pd.DataFrame,
                      period_min: int = 60, window_size: int = 50) -> Tuple[pd.Series, np.ndarray]:
        """
        Forecasts future 'Close' prices using a trained LSTM model.

        This function performs step-by-step (one-step-ahead) forecasting. It takes
        the last `window_size` data points from the historical DataFrame, uses them
        to predict the next value, then appends that prediction to the sequence
        to predict the subsequent value, and so on, for `period_min` steps.

        Args:
            model (tf.keras.Model): The trained LSTM model (e.g., from TensorFlow/Keras).
                                    Expected to be compiled and fitted.
            df (pd.DataFrame): The full historical dataset used for training, containing
                                at least 'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                'MA_50', 'RSI'. This DataFrame should be in its *final
                                preprocessed form* (e.g., scaled if the model expects scaled input).
            period_min (int): The number of future time steps (minutes) to forecast. Defaults to 60.
            window_size (int): The sequence length (number of past time steps) the LSTM model
                               was trained with for a single prediction. Defaults to 50.

        Returns:
            Tuple[pd.Series, np.ndarray]:
                - forecast_times (pd.Series): A Series of datetime objects representing the
                                              future time steps for which forecasts are made.
                - forecast_values (np.ndarray): A NumPy array of the forecasted 'Close' prices.

        Raises:
            MLToolkitError: If inputs are invalid (e.g., model is not Keras Model, DataFrame
                            is missing columns/empty, window_size/period_min are invalid).
        """
        try:
            # Input Validations
            if not isinstance(model, tf.keras.Model):
                raise TypeError("Input 'model' must be a tensorflow.keras.Model.")
            ErrorHandler.validate_dataframe(df, ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA_50', 'RSI'])
            if not isinstance(period_min, int) or period_min <= 0:
                raise ValueError("period_min must be a positive integer.")
            if not isinstance(window_size, int) or window_size <= 0:
                raise ValueError("window_size must be a positive integer.")
            if len(df) < window_size:
                raise ValueError(f"DataFrame length ({len(df)}) must be at least 'window_size' ({window_size}) "
                                 "to create the initial input sequence for forecasting.")
            
            # Ensure 'Time' column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                df['Time'] = pd.to_datetime(df['Time'])

            forecast_times_list = []
            forecast_values_list = []
            
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_50', 'RSI']
            num_features = len(feature_cols)

            current_sequence = df[feature_cols].iloc[-window_size:].values
            
            last_known_time = df['Time'].iloc[-1]

            # Predict for the next 'period_min' minutes
            for i in range(1, period_min + 1): 
                input_for_prediction = current_sequence.reshape(1, window_size, num_features)
                predicted_output = model.predict(input_for_prediction, verbose=0) # verbose=0 to suppress predict output
                predicted_close = predicted_output[0][0] 
                forecast_values_list.append(predicted_close)
                next_time_step = last_known_time + timedelta(minutes=i)
                forecast_times_list.append(next_time_step)
                last_features_except_close = current_sequence[-1, :][np.array([col != 'Close' for col in feature_cols])]
                new_row_values = []
                close_idx = feature_cols.index('Close')
                other_feature_values_idx = 0
                for idx, col in enumerate(feature_cols):
                    if col == 'Close':
                        new_row_values.append(predicted_close)
                    else:
                        new_row_values.append(current_sequence[-1, idx]) 
                        other_feature_values_idx += 1
                
                new_row = np.array(new_row_values).reshape(1, num_features)
                current_sequence = np.vstack((current_sequence[1:], new_row))

            return pd.Series(forecast_times_list), np.array(forecast_values_list).reshape(-1, 1)

        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to perform LSTM forecasting")
            


    @staticmethod
    def process_signals(df: pd.DataFrame, threshold: int = 10) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Process predicted signals ('y_pred' column) in a DataFrame to identify trades,
        calculate profit, and derive performance metrics.

        This function simulates a simple trading strategy where a trade is opened
        if a 'Buy' (0) or 'Sell' (2) signal is maintained for a 'threshold' number
        of consecutive periods. Trades are closed when an opposite signal or a neutral (1)
        signal occurs.

        Args:
            df (pd.DataFrame): DataFrame expected to contain at least the following columns:
                - 'y_pred': Predicted signal (0=Buy, 1=Neutral, 2=Sell).
                - 'Time': Timestamp of the price data (datetime type).
                - 'Close': The closing price at the given timestamp.
            threshold (int): The minimum number of consecutive identical signals required
                             to open a trade. Defaults to 10.

        Returns:
            Tuple[pd.DataFrame, Dict[str, float]]:
                - trade_log (pd.DataFrame): DataFrame detailing all simulated trades,
                                            including entry/exit times/prices, profit, and direction.
                - metrics (dict): Dictionary of accumulated strategy performance metrics:
                                  "Total Profit", "Sharpe Ratio", "Total Trades".

        Raises:
            MLToolkitError: If required columns are missing, DataFrame is empty,
                            or inputs are invalid.
            ValueError: If 'threshold' is not a positive integer.
        """
        try:
            required_columns = ['y_pred', 'Time', 'Close']
            ErrorHandler.validate_dataframe(df, required_columns)

            if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                df['Time'] = pd.to_datetime(df['Time'])

            if not isinstance(threshold, int) or threshold <= 0:
                raise ValueError("Threshold must be a positive integer.")

            # Initialize variables
            potential_signals_buffer = [] # Buffer to store consecutive signals
            is_open_trade = False
            trade_direction = None # Store 0 for Buy, 2 for Sell
            entry_price = None
            entry_time = None
            
            trades = []  # List to hold trade details
            profits = []  # For Sharpe ratio calculation

            for i in range(len(df)):
                current_signal = df['y_pred'].iloc[i]
                current_price = df['Close'].iloc[i]
                current_time = df['Time'].iloc[i]

                # Skip neutral signals unless a trade is open
                if current_signal == 1:
                    if is_open_trade: 
                        exit_price = current_price
                        exit_time = current_time
                        profit = (exit_price - entry_price) if trade_direction == 0 else (entry_price - exit_price)
                        profits.append(profit)
                        trades.append({
                            "Entry Date": entry_time,
                            "Entry Price": entry_price,
                            "Exit Date": exit_time,
                            "Exit Price": exit_price,
                            "Profit": profit,
                            "Direction": "Buy" if trade_direction == 0 else "Sell",
                            "Exit Reason": "Neutral Signal"
                        })
                        # Reset for next trade
                        is_open_trade = False
                        entry_price = None
                        entry_time = None
                        trade_direction = None
                    potential_signals_buffer = [] 
                    continue

                # If no trade is open
                if not is_open_trade:
                    if not potential_signals_buffer:
                        potential_signals_buffer.append(current_signal)
                    elif current_signal == potential_signals_buffer[-1]:
                        potential_signals_buffer.append(current_signal)
                        if len(potential_signals_buffer) >= threshold:
                            # Open trade
                            entry_price = current_price
                            entry_time = current_time
                            trade_direction = current_signal 
                            is_open_trade = True
                            potential_signals_buffer = [] 
                    else: 
                        potential_signals_buffer = [current_signal]
                
                # If a trade is open, check for closing conditions
                else: 
                    if current_signal != trade_direction:
                        exit_price = current_price
                        exit_time = current_time
                        profit = (exit_price - entry_price) if trade_direction == 0 else (entry_price - exit_price)
                        profits.append(profit)
                        trades.append({
                            "Entry Date": entry_time,
                            "Entry Price": entry_price,
                            "Exit Date": exit_time,
                            "Exit Price": exit_price,
                            "Profit": profit,
                            "Direction": "Buy" if trade_direction == 0 else "Sell",
                            "Exit Reason": "Opposite Signal"
                        })
                        # Reset for next trade and start new potential sequence
                        is_open_trade = False
                        entry_price = None
                        entry_time = None
                        trade_direction = None
                        potential_signals_buffer = [current_signal] 

            # Handle open trade at the end of the DataFrame
            if is_open_trade:
                print("Warning: Trade was still open at the end of the data. Not included in trade_log.")

            # Create trade log DataFrame
            trade_log = pd.DataFrame(trades)

            # Calculate metrics
            total_profit = np.sum(profits) if profits else 0
            mean_profit_per_trade = np.mean(profits) if profits else 0
            std_dev_profit_per_trade = np.std(profits) if len(profits) > 1 else 0 
            
            # Simple Sharpe Ratio (per trade, not annualized)
            sharpe_ratio = mean_profit_per_trade / std_dev_profit_per_trade if std_dev_profit_per_trade != 0 else 0

            metrics = {
                "Total Profit": total_profit,
                "Sharpe Ratio (Per Trade)": sharpe_ratio,
                "Total Trades": len(trades),
                "Mean Profit Per Trade": mean_profit_per_trade,
                "Std Dev Profit Per Trade": std_dev_profit_per_trade
            }

            return trade_log, metrics

        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to process signals and calculate trade metrics")
            
            