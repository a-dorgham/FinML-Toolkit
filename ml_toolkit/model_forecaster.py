import pandas as pd
import numpy as np
import tensorflow as tf # Assuming tensorflow.keras.Model is used
from datetime import timedelta
from typing import Tuple

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