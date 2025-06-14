# ---------------------------------------------
# FINANCIAL MACHINE LEARNING TOOLKIT
# ---------------------------------------------

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Any
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks

from ml_toolkit.error_handler import ErrorHandler
from ml_toolkit.data_handler import DataHandler

# ---------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------


class FeatureEngineer:
    """Handles feature engineering for financial data."""

    @staticmethod
    def find_peaks_and_valleys(df: pd.DataFrame, distance: int = 10, prominence: float = 0.1) -> np.ndarray:
        """
        Identify peaks and valleys in financial data.

        Args:
            df: DataFrame with 'Time' and 'Close' columns.
            distance: Minimum number of data points between peaks/valleys.
            prominence: Minimum prominence of peaks/valleys.

        Returns:
            np.ndarray: Array of signals (0: buy, 1: neutral, 2: sell).

        Raises:
            MLToolkitError: If peak detection fails or required columns are missing.
        """
        try:
            ErrorHandler.validate_dataframe(df, ['Time', 'Close'])
            scaler = MinMaxScaler(feature_range=(0, 1))
            close_scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1)).flatten()

            peak_indices, _ = find_peaks(close_scaled, distance=distance, prominence=prominence)
            valley_indices, _ = find_peaks(-close_scaled, distance=distance, prominence=prominence)

            signal = np.ones_like(close_scaled)
            signal[peak_indices] = 2
            signal[valley_indices] = 0
            return signal
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to detect peaks and valleys")

    @staticmethod
    def rsi_features(price_data: pd.Series, period: int = 9) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            price_data: Series of price data.
            period: Lookback period for RSI calculation.

        Returns:
            pd.Series: RSI values.

        Raises:
            MLToolkitError: If RSI calculation fails.
        """
        try:
            ErrorHandler.validate_not_empty(price_data, "Price data")
            delta = price_data.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            exp1 = gain.ewm(com=period - 1, adjust=False).mean()
            exp2 = loss.ewm(com=period - 1, adjust=False).mean()
            RS = exp1 / exp2
            RSI = 100 - (100 / (1 + RS))
            return RSI
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to calculate RSI")

    @staticmethod
    def crossover(x: pd.Series, y: pd.Series, cross_distance: Optional[int] = None) -> pd.Series:
        """
        Detect crossover events between two series.

        Args:
            x: First series.
            y: Second series.
            cross_distance: Number of periods to shift for crossover detection.

        Returns:
            pd.Series: Boolean series indicating crossover events.

        Raises:
            MLToolkitError: If crossover detection fails.
        """
        try:
            ErrorHandler.validate_not_empty(x, "Series x")
            ErrorHandler.validate_not_empty(y, "Series y")
            shift_value = 1 if cross_distance is None else cross_distance
            return (x > y) & (x.shift(shift_value) < y.shift(shift_value))
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to detect crossover")

    @staticmethod
    def crossunder(x: pd.Series, y: pd.Series, cross_distance: Optional[int] = None) -> pd.Series:
        """
        Detect crossunder events between two series.

        Args:
            x: First series.
            y: Second series.
            cross_distance: Number of periods to shift for crossunder detection.

        Returns:
            pd.Series: Boolean series indicating crossunder events.

        Raises:
            MLToolkitError: If crossunder detection fails.
        """
        try:
            ErrorHandler.validate_not_empty(x, "Series x")
            ErrorHandler.validate_not_empty(y, "Series y")
            shift_value = 1 if cross_distance is None else cross_distance
            return (x < y) & (x.shift(shift_value) > y.shift(shift_value))
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to detect crossunder")

    @staticmethod
    def get_signal(row: pd.Series) -> int:
        """
        Generate trading signal based on stochastic oscillator.

        Args:
            row: DataFrame row containing '%K', 'CO', and 'CU' columns.

        Returns:
            int: Signal value (0: buy, 1: neutral, 2: sell).

        Raises:
            MLToolkitError: If signal generation fails.
        """
        try:
            if row['%K'] < 20 and row['CO']:
                return 0
            elif row['%K'] > 80 and row['CU']:
                return 2
            return 1
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to generate stochastic signal")

    @staticmethod
    def stochastic_features(df: pd.DataFrame, k_length: int = 20, k_period: int = 1, d_period: int = 3) -> pd.DataFrame:
        """
        Calculate stochastic oscillator features (%K, %D, CO, CU, STC_SIGNAL).

        Args:
            df: DataFrame with 'High', 'Low', 'Close' columns.
            k_length: Lookback period for %K calculation.
            k_period: Smoothing period for %K.
            d_period: Smoothing period for %D.

        Returns:
            pd.DataFrame: DataFrame with stochastic features.

        Raises:
            MLToolkitError: If stochastic feature calculation fails.
        """
        try:
            ErrorHandler.validate_dataframe(df, ['High', 'Low', 'Close'])
            df = df.copy()
            df['L14'] = df['Low'].rolling(window=k_length).min()
            df['H14'] = df['High'].rolling(window=k_length).max()
            df['%K0'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))
            df['%K'] = df['%K0'].rolling(window=k_period).mean()
            df['%D'] = df['%K'].rolling(window=d_period).mean()
            df['CO'] = FeatureEngineer.crossover(df['%K'], df['%D'])
            df['CU'] = FeatureEngineer.crossunder(df['%K'], df['%D'])
            df['STC_SIGNAL'] = df.apply(FeatureEngineer.get_signal, axis=1)
            return df[['%K', '%D', 'CO', 'CU', 'STC_SIGNAL']]
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to calculate stochastic features")

    @staticmethod
    def macd_features(df: pd.DataFrame, short_window: int = 1, long_window: int = 26, signal_window: int = 9) -> pd.Series:
        """
        Calculate MACD indicator.

        Args:
            df: DataFrame with 'Close' column.
            short_window: Short-term EMA window.
            long_window: Long-term EMA window.
            signal_window: Signal line EMA window.

        Returns:
            pd.Series: MACD values.

        Raises:
            MLToolkitError: If MACD calculation fails.
        """
        try:
            ErrorHandler.validate_dataframe(df, ['Close'])
            df = df.copy()
            df['Short_EMA'] = df['Close'].ewm(span=short_window, adjust=False).mean()
            df['Long_EMA'] = df['Close'].ewm(span=long_window, adjust=False).mean()
            df['MACD_'] = df['Short_EMA'] - df['Long_EMA']
            df['MACD'] = df['MACD_'].ewm(span=signal_window, adjust=False).mean()
            return df['MACD']
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to calculate MACD")

    @staticmethod
    def add_features(df: pd.DataFrame, features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Add technical indicators as features to the DataFrame.

        Args:
            df: Input DataFrame with 'Time', 'Open', 'High', 'Low', 'Close', 'Volume' columns.
            features: List of feature names to include (optional).

        Returns:
            Tuple containing updated DataFrame, time series, feature matrix (X), and target variable (y).

        Raises:
            MLToolkitError: If feature engineering fails.
        """
        try:
            ErrorHandler.validate_dataframe(df, ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df = df.copy()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = FeatureEngineer.rsi_features(df['Close'])
            df = DataHandler.handle_nan(df)
            stoch_features = FeatureEngineer.stochastic_features(df[['High', 'Low', 'Close']])
            df = pd.concat([df, stoch_features], axis=1)
            df = DataHandler.handle_nan(df)
            df['MACD'] = FeatureEngineer.macd_features(df[['Close']])
            df['Signal'] = FeatureEngineer.find_peaks_and_valleys(df[['Time', 'Close']], distance=5, prominence=0.01)
            df = DataHandler.handle_nan(df)

            time = df['Time']
            features = features or ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']
            if isinstance(features, tuple):
                features = list(features)
            X = df[features]
            y = df['Signal']
            return df, time, X, y
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to add features")

