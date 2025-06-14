# ---------------------------------------------
# FINANCIAL MACHINE LEARNING TOOLKIT
# ---------------------------------------------

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle

from ml_toolkit.error_handler import ErrorHandler, MLToolkitError

# ---------------------------------------------
# DATA HANDLING
# ---------------------------------------------


class DataHandler:
    """Handles loading, splitting, scaling, and preprocessing of financial data."""

    @staticmethod
    def load_data(source: Union[str, pd.DataFrame], start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Loads financial time series data from a pickle file or uses an existing DataFrame,
        then filters it by a specified date range and standardizes column names.

        This method is designed to:
        1. Load data from a `.pkl` file path or accept a pre-loaded pandas DataFrame.
        2. Ensure the 'time' column is in datetime format and localized to be timezone-naive
           for consistent date comparisons.
        3. Filter the data based on `start_date` and `end_date`. If these dates are not
           provided, it intelligently defaults the `end_date` to the latest timestamp in the
           data and the `start_date` to 30 days prior.
        4. Standardize key financial columns to 'Time', 'Open', 'High', 'Low', 'Close', and 'Volume'.
        5. Sort the resulting DataFrame by 'Time' in ascending order.

        Args:
            source (Union[str, pd.DataFrame]): The data source. This can be:
                                               - A `str`: Path to a pickle file (`.pkl`) containing the raw data.
                                               - A `pd.DataFrame`: An already loaded pandas DataFrame.
            start_date (Optional[str]): The inclusive start date for filtering the data.
                                        Expected format: 'YYYY-MM-DD'.
                                        If `None`, it defaults to 30 days before the `end_date` (or the
                                        latest date in the data if `end_date` is also `None`).
            end_date (Optional[str]): The inclusive end date for filtering the data.
                                      Expected format: 'YYYY-MM-DD'.
                                      If `None`, it defaults to the latest date found in the raw data.

        Returns:
            pd.DataFrame: A new DataFrame containing the filtered and cleaned financial data
                          with standardized column names ('Time', 'Open', 'High', 'Low', 'Close', 'Volume').

        Raises:
            MLToolkitError: If the data `source` is invalid (e.g., not a string path or DataFrame),
                            if the specified file cannot be loaded, if required raw columns
                            ('time', 'mid_o', 'mid_h', 'mid_l', 'mid_c', 'volume') are missing,
                            if date parsing fails, or if the filtered DataFrame becomes empty.
        """
        raw_df: pd.DataFrame
        
        try:
            # Load data from source (file or DataFrame)
            if isinstance(source, str):
                ErrorHandler.validate_file_exists(source)
                print(f"Loading raw data from pickle file: '{source}'...")
                raw_df = pd.read_pickle(source)
            elif isinstance(source, pd.DataFrame):
                raw_df = source.copy()
                print("Using provided raw DataFrame.")
            else:
                # Raise error for unsupported source type
                raise MLToolkitError("Invalid 'source'. Must be a file path (str) or a pandas DataFrame.")

            ErrorHandler.validate_not_empty(raw_df, "Raw financial data DataFrame")

            required_raw_cols = ['time', 'mid_o', 'mid_h', 'mid_l', 'mid_c', 'volume']
            ErrorHandler.validate_columns_exist(raw_df, required_raw_cols)

            if not pd.api.types.is_datetime64_any_dtype(raw_df['time']):
                raw_df['time'] = pd.to_datetime(raw_df['time'])
            
            if raw_df['time'].dt.tz is not None:
                raw_df['time'] = raw_df['time'].dt.tz_localize(None)

            latest_data_date = raw_df['time'].max()
            earliest_data_date = raw_df['time'].min()

            if end_date is None:
                end_datetime = latest_data_date
            else:
                end_datetime = pd.to_datetime(end_date)
                ErrorHandler.validate_date_in_range(end_datetime, earliest_data_date, latest_data_date, "End date")

            if start_date is None:
                start_datetime = end_datetime - timedelta(days=30)
            else:
                start_datetime = pd.to_datetime(start_date)
                ErrorHandler.validate_date_in_range(start_datetime, earliest_data_date, latest_data_date, "Start date")
            
            # Ensure start date is not after end date
            ErrorHandler.validate_date_order(start_datetime, end_datetime)

            print(f"Filtering data from {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}.")

            mask = (raw_df['time'] >= start_datetime) & (raw_df['time'] <= end_datetime)
            df_filtered = raw_df.loc[mask].copy()

            ErrorHandler.validate_not_empty(df_filtered, f"Filtered DataFrame for range {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}")

            # Create new DataFrame with standardized column names
            df_cleaned = pd.DataFrame({
                'Time': df_filtered['time'],
                'Open': df_filtered['mid_o'],
                'High': df_filtered['mid_h'],
                'Low': df_filtered['mid_l'],
                'Close': df_filtered['mid_c'],
                'Volume': df_filtered['volume']
            })

            # Sort by 'Time' and reset index for a clean output
            df_cleaned.sort_values(by='Time', ascending=True, inplace=True)
            df_cleaned.reset_index(drop=True, inplace=True)
            
            print(f"Data successfully loaded and filtered. Resulting shape: {df_cleaned.shape}")
            print(f"Time range in cleaned data: {df_cleaned['Time'].min().strftime('%Y-%m-%d %H:%M:%S')} to {df_cleaned['Time'].max().strftime('%Y-%m-%d %H:%M:%S')}")

            return df_cleaned

        except Exception as e:
            # Centralized error handling
            ErrorHandler.handle_error(e, "Failed to load and process financial data")
            
    @staticmethod
    def split_custom_data(X: pd.DataFrame, y: pd.Series, non_zero_train_ratio: float = 0.9,
                          random_zero_samples: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
        """
        Split dataset with a custom strategy for imbalanced data.

        Args:
            X: Feature matrix.
            y: Target variable.
            non_zero_train_ratio: Proportion of non-zero target data for training.
            random_zero_samples: Number of random zero-target samples for training.

        Returns:
            Tuple containing X_train, X_test, y_train, y_test, train_indices, test_indices.

        Raises:
            MLToolkitError: If data splitting fails or inputs are invalid.
        """
        try:
            ErrorHandler.validate_not_empty(X, "Feature matrix X")
            ErrorHandler.validate_not_empty(y, "Target variable y")
            if not 0 < non_zero_train_ratio < 1:
                raise MLToolkitError("non_zero_train_ratio must be between 0 and 1")

            non_zero_indices = y[y != 0].index
            zero_indices = y[y == 0].index
            non_zero_indices = shuffle(non_zero_indices, random_state=42)
            zero_indices = shuffle(zero_indices, random_state=42)

            num_non_zero_train = int(len(non_zero_indices) * non_zero_train_ratio)
            non_zero_train_indices = non_zero_indices[:num_non_zero_train]
            non_zero_test_indices = non_zero_indices[num_non_zero_train:]
            random_zero_train_indices = zero_indices[:random_zero_samples]
            remaining_zero_indices = zero_indices[random_zero_samples:]

            train_indices = np.concatenate([non_zero_train_indices, random_zero_train_indices])
            test_indices = np.concatenate([remaining_zero_indices, non_zero_test_indices])

            train_indices = shuffle(train_indices, random_state=42)
            test_indices = shuffle(test_indices, random_state=42)

            X_train, y_train = X.loc[train_indices], y.loc[train_indices]
            X_test, y_test = X.loc[test_indices], y.loc[test_indices]
            return X_train, X_test, y_train, y_test, train_indices, test_indices
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to split data")

    @staticmethod
    def split_data(time: pd.Series, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, range, range]:
        """
        Split dataset based on time, with recent data for testing.

        Args:
            time: Time column for sorting.
            X: Feature matrix.
            y: Target variable.
            test_size: Proportion of data for testing.

        Returns:
            Tuple containing X_train, X_test, y_train, y_test, train_indices, test_indices.

        Raises:
            MLToolkitError: If data splitting fails or inputs are invalid.
        """
        try:
            ErrorHandler.validate_not_empty(time, "Time series")
            ErrorHandler.validate_not_empty(X, "Feature matrix X")
            ErrorHandler.validate_not_empty(y, "Target variable y")
            if not 0 < test_size < 1:
                raise MLToolkitError("test_size must be between 0 and 1")

            time = pd.to_datetime(time)
            test_cutoff = int(len(time) * test_size)
            test_indices = np.array(range(0, test_cutoff))
            train_indices = range(test_cutoff, len(time))

            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            return X_train, X_test, y_train, y_test, train_indices, test_indices
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to split data by time")

    @staticmethod
    def scale_data(data: Union[pd.DataFrame, pd.Series], data_type: str, scaler: Optional[MinMaxScaler] = None,
                   is_categorical: bool = False, encoder: Optional[LabelEncoder] = None,
                   reshape: bool = False) -> Tuple[np.ndarray, Union[MinMaxScaler, LabelEncoder]]:
        """
        Scale input data for features or targets.

        Args:
            data: Input data to scale (DataFrame for X, Series for y).
            data_type: 'train' or 'test'.
            scaler: Predefined scaler for features or targets.
            is_categorical: Whether the target is categorical.
            encoder: Predefined label encoder for categorical targets.
            reshape: Whether to reshape data for LSTM.

        Returns:
            Tuple of scaled data and updated scaler or encoder.

        Raises:
            MLToolkitError: If scaling fails or inputs are invalid.
        """
        try:
            ErrorHandler.validate_not_empty(data, "Input data")
            if data_type not in ['train', 'test']:
                raise MLToolkitError("data_type must be 'train' or 'test'")

            if isinstance(data, pd.DataFrame):
                if scaler is None:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data) if data_type == 'train' else scaler.transform(data)
                if reshape:
                    scaled_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1])
                return scaled_data, scaler
            else:
                if is_categorical:
                    if data_type == 'train':
                        encoder = LabelEncoder()
                        scaled_data = encoder.fit_transform(data).astype(int)
                    else:
                        if encoder is None:
                            raise MLToolkitError("Encoder must be provided for test data when is_categorical=True")
                        scaled_data = encoder.transform(data).astype(int)
                    return scaled_data, encoder
                else:
                    if scaler is None:
                        scaler = MinMaxScaler(feature_range=(-1, 1))
                    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)) if data_type == 'train' else scaler.transform(data.values.reshape(-1, 1))
                    if reshape:
                        scaled_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1])
                    return scaled_data, scaler
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to scale data")

    @staticmethod
    def check_nan(X: pd.DataFrame, y: pd.Series) -> None:
        """
        Check for NaN values in features and target.

        Args:
            X: Feature matrix.
            y: Target variable.

        Raises:
            MLToolkitError: If NaN values are detected.
        """
        try:
            if X.isnull().values.any() or y.isnull().values.any():
                raise MLToolkitError("NaN values detected in dataset")
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to check for NaN values")

    @staticmethod
    def handle_nan(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaN values by forward filling and dropping remaining NaNs.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with NaN values handled.

        Raises:
            MLToolkitError: If NaN handling fails.
        """
        try:
            ErrorHandler.validate_not_empty(df, "DataFrame")
            df.ffill(inplace=True)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to handle NaN values")

    @staticmethod
    def normalize_signal_to_price_range(signal_series: pd.Series, price_series: pd.Series) -> pd.Series:
        """
        Normalizes a given signal series to fit within the min/max range of a specified price series.

        This is useful for plotting signals on the same scale as price data.
        Handles edge cases where the signal or price series might be constant.

        Args:
            signal_series (pd.Series): The signal series (e.g., STC_SIGNAL) to be normalized.
                                       Must be a pandas Series.
            price_series (pd.Series): The price series (e.g., Close prices) defining the target range.
                                      Must be a pandas Series.

        Returns:
            pd.Series: A new pandas Series with the signal normalized to the price range.

        Raises:
            MLToolkitError: If inputs are not pandas Series or are empty.
        """
        try:
            # Validate inputs
            if not isinstance(signal_series, pd.Series):
                raise TypeError("Input 'signal_series' must be a pandas Series.")
            if not isinstance(price_series, pd.Series):
                raise TypeError("Input 'price_series' must be a pandas Series.")

            ErrorHandler.validate_not_empty(signal_series, "Signal series")
            ErrorHandler.validate_not_empty(price_series, "Price series")

            signal_min = signal_series.min()
            signal_max = signal_series.max()
            
            price_min = price_series.min()
            price_max = price_series.max()

            # Handle edge case: signal_series is constant
            if signal_max == signal_min:
                print("Warning: Signal series is constant. Normalizing to the mean of the price range.")
                normalized_signal = pd.Series([price_series.mean()] * len(signal_series), index=signal_series.index)
            # Handle edge case: price_series is constant
            elif price_max == price_min:
                print("Warning: Price series is constant. Normalized signal will be a constant at the price level.")
                normalized_signal = pd.Series([price_min] * len(signal_series), index=signal_series.index)
            else:
                # Apply the min-max scaling formula
                normalized_signal = price_min + \
                                    (signal_series - signal_min) * \
                                    (price_max - price_min) / \
                                    (signal_max - signal_min)
            
            return normalized_signal

        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to normalize signal to price range")

    @staticmethod
    def scale_to_0_1_range(data_series: pd.Series) -> pd.Series:
        """
        Scales a pandas Series to a range between 0 and 1 (inclusive).

        Uses Min-Max scaling: X_scaled = (X - X_min) / (X_max - X_min).
        Handles edge cases where the series might be constant.

        Args:
            data_series (pd.Series): The series to be scaled.

        Returns:
            pd.Series: A new pandas Series with values scaled between 0 and 1.

        Raises:
            MLToolkitError: If the input is not a pandas Series or is empty.
        """
        try:
            if not isinstance(data_series, pd.Series):
                raise TypeError("Input 'data_series' must be a pandas Series.")
            ErrorHandler.validate_not_empty(data_series, "Data series for scaling")

            min_val = data_series.min()
            max_val = data_series.max()

            if max_val == min_val:
                print(f"Warning: Series '{data_series.name or 'Unnamed'}' is constant. "
                      "All values will be scaled to 0.5.")
                return pd.Series(0.5, index=data_series.index)
            
            scaled_series = (data_series - min_val) / (max_val - min_val)
            return scaled_series

        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to scale series to 0-1 range")