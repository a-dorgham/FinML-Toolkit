# ---------------------------------------------
# FINANCIAL MACHINE LEARNING TOOLKIT
# ---------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from typing import Tuple, List, Optional, Union, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from IPython.display import display, HTML

from ml_toolkit.error_handler import ErrorHandler, MLToolkitError
from ml_toolkit.data_handler import DataHandler
from ml_toolkit.model_forecaster import ModelForecaster

# ---------------------------------------------
# VISUALIZATION
# ---------------------------------------------


class Visualizer:
    """Handles visualization of financial data and model results."""

    @staticmethod
    def plot_class_distribution(y: np.ndarray, title: str = 'Class Distribution') -> None:
        """
        Plot class distribution as a bar plot.

        Args:
            y: Labels to plot.
            title: Plot title.

        Raises:
            MLToolkitError: If plotting fails.
        """
        try:
            ErrorHandler.validate_not_empty(y, "Labels")
            sns.countplot(x=y)
            plt.title(title)
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.show()
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to plot class distribution")

    @staticmethod
    def plot_pie_charts(y: Union[np.ndarray, List[np.ndarray]], titles: Union[str, List[str]]) -> None:
        """
        Plot pie charts for class distributions.

        Args:
            y: Single or list of label arrays.
            titles: Single or list of plot titles.

        Raises:
            MLToolkitError: If plotting fails.
        """
        try:
            y_list = y if isinstance(y, list) else [y]
            titles = titles if isinstance(titles, list) else [titles]
            if len(y_list) != len(titles):
                raise MLToolkitError("Number of datasets must match number of titles")
            for i, y_data in enumerate(y_list):
                if len(np.array(y_data).shape) > 1:
                    raise MLToolkitError(f"Dataset at index {i} must be 1D")
            fig, axs = plt.subplots(1, len(y_list), figsize=(3.5 * len(y_list), 5))
            axs = [axs] if len(y_list) == 1 else axs
            for i, (y_data, title) in enumerate(zip(y_list, titles)):
                value_counts = pd.Series(y_data).value_counts()
                axs[i].pie(value_counts, autopct="%.2f", labels=value_counts.index.tolist())
                axs[i].set_title(title)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to plot pie charts")

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix",
                              figsize: Tuple[int, int] = (4, 4), cmap: str = "Blues") -> None:
        """
        Plot confusion matrix as a heatmap.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            title: Plot title.
            figsize: Figure size.
            cmap: Colormap for heatmap.

        Raises:
            MLToolkitError: If plotting fails.
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=figsize)
            plt.title(title, size=11)
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, annot_kws={"size": 10})
            plt.xlabel('Predicted Label', fontsize=10)
            plt.ylabel('True Label', fontsize=10)
            plt.show()
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to plot confusion matrix")

    @staticmethod
    def print_html(text: str, font_size: str = "18px", font_family: str = "Arial", color: str = "blue",
                   align: str = "left", weight: str = "bold") -> None:
        """
        Display styled HTML text in Jupyter Notebook.

        Args:
            text: Text to display.
            font_size: Font size (e.g., "18px").
            font_family: Font family (e.g., "Arial").
            color: Text color.
            align: Text alignment.
            weight: Font weight.

        Raises:
            MLToolkitError: If HTML display fails.
        """
        try:
            html_code = f"""
            <div style="font-size:{font_size}; font-family:{font_family}; color:{color}; font-weight:{weight}; text-align:{align};">
                {text}
            </div>
            """
            display(HTML(html_code))
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to display HTML text")

    @staticmethod
    def print_hline(text: str, line_color: str = "black", line_thickness: str = "2px", font_size: str = "18px",
                    font_family: str = "Arial", color: str = "blue", align: str = "left", weight: str = "bold",
                    line_position: str = "bottom") -> None:
        """
        Display text with a horizontal line in Jupyter Notebook.

        Args:
            text: Text to display.
            line_color: Color of the horizontal line.
            line_thickness: Thickness of the line.
            font_size: Font size.
            font_family: Font family.
            color: Text color.
            align: Text alignment.
            weight: Font weight.
            line_position: Position of the line ('top' or 'bottom').

        Raises:
            MLToolkitError: If HTML display fails.
        """
        try:
            html_code = f"""
            <div style="font-size:{font_size}; font-family:{font_family}; color:{color}; font-weight:{weight}; text-align:{align}; position: relative; padding-top: 2px;">
                <div style="position: absolute; {line_position}: -7px; left: 0; right: 0; height: {line_thickness}; background-color: {line_color};"></div>
                {text}
            </div>
            """
            display(HTML(html_code))
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to display text with horizontal line")

    @staticmethod
    def print_hline_with_background(text: str, line_color: str = "black", line_thickness: str = "2px",
                                    line_position: str = "bottom", font_size: str = "20px",
                                    font_family: str = "Arial", color: str = "#1f77b4", align: str = "left",
                                    weight: str = "bold", background_color: str = "#D3D3D3") -> None:
        """
        Display text with a horizontal line and background color in Jupyter Notebook.

        Args:
            text: Text to display.
            line_color: Color of the horizontal line.
            line_thickness: Thickness of the line.
            line_position: Position of the line ('top' or 'bottom').
            font_size: Font size.
            font_family: Font family.
            color: Text color.
            align: Text alignment.
            weight: Font weight.
            background_color: Background color.

        Raises:
            MLToolkitError: If HTML display fails.
        """
        try:
            html_code = f"""
            <div style="background-color:{background_color}; width: 100%; padding: 7px; position: relative; text-align:{align};">
                <div style="position: absolute; {line_position}: -4px; left: 0; right: 0; height: {line_thickness}; background-color: {line_color};"></div>
                <span style="font-size:{font_size}; font-family:{font_family}; color:{color}; font-weight:{weight};">
                    {text}
                </span>
            </div>
            """
            display(HTML(html_code))
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to display text with background")

    @staticmethod
    def plot_with_peaks(df: pd.DataFrame, fig: go.Figure = go.Figure(), plot_data: bool = True,
                        plot_ma: bool = False, plot_stoch: bool = False, plot_peaks: bool = False, plot_ypred: bool = False,
                        plot_buysell: bool = False, plot_tclose: bool = False, symbol_size: int = 10,
                        fig_show: bool = False) -> go.Figure:
        """
        Plot financial data with peaks, valleys, and predictions using Plotly.

        Args:
            df: DataFrame with 'Time', 'Open', 'High', 'Low', 'Close', 'Signal', 'y_pred', 'BuySell', 'TClose' columns.
            fig: Plotly Figure object.
            plot_data: Whether to plot candlestick data.
            plot_ma: Whether to plot moving average.
            plot_stoch: Whether to plot stochastic signal.
            plot_peaks: Whether to plot peaks and valleys.
            plot_ypred: Whether to plot predicted signals.
            plot_buysell: Whether to plot buy/sell signals.
            plot_tclose: Whether to plot TClose signals.
            symbol_size: Size of marker symbols.
            fig_show: Whether to display the figure.

        Returns:
            go.Figure: Updated Plotly figure.

        Raises:
            MLToolkitError: If plotting fails.
        """
        try:
            ErrorHandler.validate_dataframe(df, ['Time', 'Open', 'High', 'Low', 'Close'])
            time = pd.to_datetime(df['Time'])
            close = df['Close']

            if plot_data:
                fig.add_trace(go.Candlestick(
                    x=time, open=df['Open'], high=df['High'], low=df['Low'], close=close,
                    name='Testing Data', increasing_line_color='green', decreasing_line_color='red'
                ))
            if plot_ma:
                fig.add_trace(go.Scatter(
                    x=time, y=close, name="Scaled Close Prices", mode='lines', line=dict(color='orange', width=1.5)
                ))

            if plot_stoch:
                stc_full = df['STC_SIGNAL']
                stc_norm = DataHandler.normalize_signal_to_price_range(stc_full, close)
                fig.add_trace(go.Scatter(
                    x=time, y=stc_norm, name="Stochastic Signal", mode='lines', line=dict(color='red', width=1.5)
                ))
               
            def add_peaks_valleys(signal_col: str, peak_marker: str, valley_marker: str, peak_color: str, valley_color: str, name_prefix: str) -> None:
                peak_indices = df[signal_col] > 1.9
                valley_indices = df[signal_col] < 0.1
                fig.add_trace(go.Scatter(
                    x=time[peak_indices], y=close[peak_indices], mode='markers',
                    marker=dict(color=peak_color, symbol=peak_marker, size=symbol_size, line_color="white", line_width=1),
                    name=f'{name_prefix} Peaks'
                ))
                fig.add_trace(go.Scatter(
                    x=time[valley_indices], y=close[valley_indices], mode='markers',
                    marker=dict(color=valley_color, symbol=valley_marker, size=symbol_size, line_color="white", line_width=1),
                    name=f'{name_prefix} Valleys'
                ))

            if plot_peaks:
                add_peaks_valleys('Signal', 'triangle-up', 'triangle-down', 'white', 'yellow', 'Signal')
            if plot_ypred:
                add_peaks_valleys('y_pred', 'star', 'circle', '#E4AFDC', 'LightSkyBlue', 'Predicted')
            if plot_buysell:
                add_peaks_valleys('BuySell', 'diamond', 'circle', '#E4AFDC', 'LightSkyBlue', 'BuySell')
            if plot_tclose:
                add_peaks_valleys('TClose', 'x', 'cross', '#79FB8C', '#79FB8C', 'TClose')

            fig.update_layout(
                title='Data Visualization with Predictions and Forecast',
                xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=True, template='plotly_white',
                width=1000, height=500, margin=dict(l=15, r=15, b=10), font=dict(size=10, color="#e1e1e1"),
                paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", legend_title_text="Elements", showlegend=False,
                xaxis=dict(gridcolor="#1f292f", showgrid=True, fixedrange=False, rangeslider=dict(visible=False),
                           rangebreaks=[dict(bounds=["sat", "mon"])])
            )
            if fig_show:
                fig.show()
            return fig
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to plot with peaks.")

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, features: list, target: list):
        """
        Generates and displays a correlation heatmap for the given features and target.

        Parameters:
            df (pd.DataFrame): Input dataframe containing features and target.
            features (list): List of feature column names.
            target (list): List of target column names.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not all(col in df.columns for col in features + target):
            raise ValueError("One or more specified features or target columns not found in the DataFrame.")

        plt.figure(figsize=(10, 8))
        sns.heatmap(df[features + target].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    @staticmethod
    def plot_prediction(df: pd.DataFrame, test_indices: Union[List[int], np.ndarray],
                        y_pred_imputed: np.ndarray, plot_type: str = 'test',
                        period_min: int = 60, model: Optional[Any] = None) -> None:
        """
        Plots the original data and predictions with flexibility to show test, train, forecast, or both datasets.
        This function uses Plotly for interactive visualizations.

        Args:
            df (pd.DataFrame): The full dataset containing the columns 'Time', 'Open', 'High', 'Low', 'Close'.
                               It should already be loaded and processed by DataHandler.
            test_indices (Union[List[int], np.ndarray]): Indices (integer locations) of the test data
                                                        within the full dataset `df`.
            y_pred_imputed (np.ndarray): Predicted values corresponding to the test data indices.
                                         These values should be on the same scale as the original prices.
            plot_type (str): The type of plot to display.
                             Options: 'test' (only actual test data and predictions),
                                      'train' (only training data),
                                      'test+train' (both training and testing data with predictions),
                                      'forecast' (only forecast data if `model` is provided).
                             Defaults to 'test'.
            period_min (int): For `plot_type='forecast'`, this specifies the number of minutes
                              to forecast beyond the last data point in the test data. Defaults to 60.
            model (Optional[Any]): The predictive model object (e.g., a Keras model) to be used
                                   for forecasting when `plot_type='forecast'`. Required for 'forecast'.

        Raises:
            MLToolkitError: If `df` is invalid, indices are malformed,
                            `y_pred_imputed` is not a NumPy array,
                            `plot_type` is invalid, or `model` is missing for 'forecast' type.
        """
        try:
            # Input Validations
            ErrorHandler.validate_dataframe(df, ['Time', 'Open', 'High', 'Low', 'Close'])
            
            if not isinstance(test_indices, (list, np.ndarray)):
                raise TypeError("test_indices must be a list or numpy array.")
            if not isinstance(y_pred_imputed, np.ndarray):
                raise TypeError("y_pred_imputed must be a numpy array.")
            if len(test_indices) != len(y_pred_imputed):
                raise ValueError("Length of test_indices must match length of y_pred_imputed.")
            
            valid_plot_types = ['test', 'train', 'test+train', 'forecast']
            if plot_type not in valid_plot_types:
                raise ValueError(f"Invalid plot_type: '{plot_type}'. Must be one of {valid_plot_types}.")
            
            if plot_type == 'forecast' and model is None:
                raise ValueError("For plot_type 'forecast', a 'model' must be provided.")
            if not isinstance(period_min, int) or period_min <= 0:
                raise ValueError("period_min must be a positive integer for forecasting.")

            # Ensure 'Time' column is datetime type for reliable indexing and plotting
            if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                df['Time'] = pd.to_datetime(df['Time'])

            # Ensure test_indices are sorted and valid
            test_indices_series = pd.Series(test_indices).sort_values().reset_index(drop=True)
            if not all(idx in df.index for idx in test_indices_series):
                raise ValueError("Some test_indices are not present in the DataFrame's index.")

            # Separate training and testing data based on indices
            train_indices = df.index.difference(test_indices_series)
            
            x_train = df.loc[train_indices, 'Time']
            x_test = df.loc[test_indices_series, 'Time']

            fig = go.Figure()

            # Add data based on plot_type
            if plot_type in ['train', 'test+train']:
                fig.add_trace(go.Candlestick(
                    x=x_train,
                    open=df.loc[train_indices, 'Open'],
                    high=df.loc[train_indices, 'High'],
                    low=df.loc[train_indices, 'Low'],
                    close=df.loc[train_indices, 'Close'],
                    name='Training Data',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ))

            if plot_type in ['test', 'test+train']:
                fig.add_trace(go.Candlestick(
                    x=x_test,
                    open=df.loc[test_indices_series, 'Open'],
                    high=df.loc[test_indices_series, 'High'],
                    low=df.loc[test_indices_series, 'Low'],
                    close=df.loc[test_indices_series, 'Close'],
                    name='Testing Data',
                    increasing_line_color='darkblue', 
                    decreasing_line_color='darkorange'
                ))

                # Add predicted data markers
                fig.add_trace(go.Scatter(
                    x=x_test,
                    y=y_pred_imputed.flatten(), 
                    mode='markers',
                    name='Predicted Data',
                    marker=dict(color='purple', size=5, symbol='circle-open') 
                ))

            if plot_type == 'forecast' and model is not None:
                forecast_times, forecast_values = ModelForecaster.forecast_LSTM(model, df, period_min)

                # Add forecast data (lines and markers)
                fig.add_trace(go.Scatter(
                    x=forecast_times,
                    y=forecast_values.flatten(), 
                    mode='lines+markers',
                    name='Forecast Data',
                    marker=dict(color='black', size=4),
                    line=dict(width=2)
                ))

                # Add the historical data leading up to the forecast for context
                fig.add_trace(go.Candlestick(
                    x=df['Time'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Historical Data',
                    increasing_line_color='green',
                    decreasing_line_color='red',
                    showlegend=True
                ))

            fig.update_layout(
                title_text='Financial Data Visualization: Actuals, Predictions & Forecast',
                xaxis_title='Time',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False, 
                template='plotly_white',
                hovermode='x unified', 
                height=600 
            )

            fig.update_layout(
                yaxis=dict(
                    autorange=True,
                    fixedrange=False
                )
            )

            fig.show()

        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to generate prediction plot")

    @staticmethod
    def plot_predictions_vs_actual(y_test_original: pd.DataFrame, predictions: pd.DataFrame, target_cols: list):
        """
        Plots predicted values against actual values for each target column.

        Parameters:
            y_test_original (pd.DataFrame): Original actual target values.
            predictions (pd.DataFrame): Predicted target values.
            target_cols (list): List of target column names.
        """
        if not isinstance(y_test_original, (pd.DataFrame, np.ndarray)) or not isinstance(predictions, (pd.DataFrame, np.ndarray)):
            raise TypeError("y_test_original and predictions must be pandas DataFrame or numpy array.")
        if y_test_original.shape != predictions.shape:
            raise ValueError("Shape of y_test_original and predictions must be the same.")

        plt.figure(figsize=(10, 6))
        for i, col in enumerate(target_cols):
            plt.plot(y_test_original[:, i], label=f"Actual {col}")
            plt.plot(predictions[:, i], label=f"Predicted {col}")
        plt.legend()
        plt.title("Predicted vs Actual Values")
        plt.show()
        

    @staticmethod
    def plot_actual_vs_predicted_lines(y_actual: np.ndarray, y_pred: np.ndarray,
                                       title: str = "Comparison of Actual and Predicted Values",
                                       x_label: str = "Index", y_label: str = "Value",
                                       actual_name: str = "Actual Values",
                                       predicted_name: str = "Predicted Values") -> None:
        """
        Plots two NumPy arrays, representing actual and predicted values, against their indices
        using Plotly for interactive visualization.

        This function is useful for quickly comparing the output of a model against ground truth,
        especially for regression tasks where outputs are continuous values.

        Args:
            y_actual (np.ndarray): The array of actual (ground truth) values.
                                   Expected to be a 1D or 2D array (e.g., (N,) or (N,1)).
            y_pred (np.ndarray): The array of predicted values.
                                 Expected to have the same shape as `y_actual`.
            title (str): The title of the plot. Defaults to "Comparison of Actual and Predicted Values".
            x_label (str): Label for the X-axis. Defaults to "Index".
            y_label (str): Label for the Y-axis. Defaults to "Value".
            actual_name (str): Name for the actual values series in the legend. Defaults to "Actual Values".
            predicted_name (str): Name for the predicted values series in the legend. Defaults to "Predicted Values".

        Raises:
            MLToolkitError: If inputs are not NumPy arrays, have different lengths/shapes,
                            or are empty.
        """
        try:
            # Validate inputs
            if not isinstance(y_actual, np.ndarray):
                raise TypeError("y_actual must be a numpy array.")
            if not isinstance(y_pred, np.ndarray):
                raise TypeError("y_pred must be a numpy array.")

            ErrorHandler.validate_not_empty(y_actual, "Actual values array")
            ErrorHandler.validate_not_empty(y_pred, "Predicted values array")

            # Ensure arrays are 1D for plotting
            y_actual_flat = y_actual.flatten()
            y_pred_flat = y_pred.flatten()

            if len(y_actual_flat) != len(y_pred_flat):
                raise ValueError("Input arrays y_actual and y_pred must have the same length after flattening.")
            
            x = np.arange(len(y_actual_flat))

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=x,
                y=y_actual_flat,
                mode='lines',
                name=actual_name,
                line=dict(color='blue', width=2)
            ))

            # Add trace for predicted values
            fig.add_trace(go.Scatter(
                x=x,
                y=y_pred_flat,
                mode='lines',
                name=predicted_name,
                line=dict(color='red', dash='dash', width=2)
            ))

            # Update layout
            fig.update_layout(
                title_text=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                hovermode='x unified',
                template='plotly_white',
                legend_title_text='Data Series'
            )

            fig.show()

        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to plot actual vs. predicted values with Plotly")

    @staticmethod
    def plot_signals_with_candlesticks(df: pd.DataFrame, test_indices: Union[List[int], np.ndarray],
                                       actual_signal: np.ndarray, predicted_signal: np.ndarray,
                                       actual_peak_threshold: float = 0.8, actual_valley_threshold: float = 0.2,
                                       predicted_peak_threshold: float = 0.8, predicted_valley_threshold: float = 0.2,
                                       min_marker_size: int = 5, max_marker_size: int = 15,
                                       title: str = "Financial Signals: Actual vs. Predicted Peaks and Valleys",
                                       show_candlesticks: bool = True, show_price_line: bool = True) -> None:
        """
        Plots candlestick data, a close price line, and highlights peaks and valleys
        identified from actual and predicted signals using Plotly.

        This function assumes `actual_signal` and `predicted_signal` are NumPy arrays,
        ideally scaled between 0 and 1, representing indicators or classification probabilities.
        The `size` of the peak/valley markers will smoothly transition based on how far
        the signal is from its threshold within the 0-1 range.

        Args:
            df (pd.DataFrame): The full dataset containing at least 'Time', 'Open', 'High', 'Low', 'Close'.
            test_indices (Union[List[int], np.ndarray]): Indices (integer locations) of the test data
                                                        within the full dataset `df`.
            actual_signal (np.ndarray): Actual signal values for the test period.
                                        Used to identify "true" peaks and valleys.
                                        Expected to be scaled (e.g., 0 to 1).
            predicted_signal (np.ndarray): Predicted signal values for the test period.
                                           Used to identify "predicted" peaks and valleys.
                                           Expected to be scaled (e.g., 0 to 1).
            actual_peak_threshold (float): Threshold for identifying a peak in `actual_signal`.
                                           Values > this threshold are considered peaks. Expected 0-1.
            actual_valley_threshold (float): Threshold for identifying a valley in `actual_signal`.
                                             Values < this threshold are considered valleys. Expected 0-1.
            predicted_peak_threshold (float): Threshold for identifying a peak in `predicted_signal`.
                                              Values >= this threshold are considered peaks. Expected 0-1.
            predicted_valley_threshold (float): Threshold for identifying a valley in `predicted_signal`.
                                                Values < this threshold are considered valleys. Expected 0-1.
            min_marker_size (int): Minimum size for peak/valley markers.
            max_marker_size (int): Maximum size for peak/valley markers.
            title (str): The main title of the plot.
            show_candlesticks (bool): If True, plots the candlestick chart for the test data.
            show_price_line (bool): If True, plots the 'Close' price line for the test data.

        Raises:
            MLToolkitError: If inputs are invalid (e.g., DataFrame missing columns,
                            indices/arrays have wrong types/lengths).
        """
        try:
            # Input Validations
            ErrorHandler.validate_dataframe(df, ['Time', 'Open', 'High', 'Low', 'Close'])
            
            if not isinstance(test_indices, (list, np.ndarray)):
                raise TypeError("test_indices must be a list or numpy array.")
            if not isinstance(actual_signal, np.ndarray):
                raise TypeError("actual_signal must be a numpy array.")
            if not isinstance(predicted_signal, np.ndarray):
                raise TypeError("predicted_signal must be a numpy array.")
            
            # Flatten arrays to ensure 1D for comparison and indexing
            actual_signal_flat = actual_signal.flatten()
            predicted_signal_flat = predicted_signal.flatten()

            if len(test_indices) != len(actual_signal_flat) or len(test_indices) != len(predicted_signal_flat):
                raise ValueError("Lengths of test_indices, actual_signal, and predicted_signal must match.")

            if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                df['Time'] = pd.to_datetime(df['Time'])

            time_test = df['Time'].loc[test_indices]
            close_test = df['Close'].loc[test_indices]

            fig = go.Figure()

            # Add testing data (candlestick) if requested
            if show_candlesticks:
                fig.add_trace(go.Candlestick(
                    x=time_test,
                    open=df.loc[test_indices, 'Open'],
                    high=df.loc[test_indices, 'High'],
                    low=df.loc[test_indices, 'Low'],
                    close=df.loc[test_indices, 'Close'],
                    name='Testing Data Candlesticks',
                    increasing_line_color='darkblue',
                    decreasing_line_color='darkorange'
                ))
            
            # Add Close Prices for test data if requested
            if show_price_line:
                fig.add_trace(go.Scatter(
                    x=time_test,
                    y=close_test,
                    name="Test Close Prices",
                    mode='lines',
                    line=dict(color='red', width=2)
                ))

            # --- Helper to process signals for peaks/valleys and marker sizes ---
            def _get_signal_markers(signal_arr, time_arr, price_arr,
                                    peak_thresh, valley_thresh,
                                    is_peak: bool, is_actual: bool):
                """Internal helper to identify and prepare marker data."""
                if is_peak:
                    indices_mask = signal_arr > peak_thresh
                    if (1.0 - peak_thresh) == 0: 
                         strengths = np.where(signal_arr > peak_thresh, 1.0, 0.0) 
                    else:
                        strengths = (signal_arr[indices_mask] - peak_thresh) / (1.0 - peak_thresh)
                    symbol = 'triangle-up' if is_actual else 'triangle-right'
                    name = 'Actual Peaks' if is_actual else 'Predicted Peaks'
                    color = 'blue' if is_actual else 'green' 
                else: # Valley
                    indices_mask = signal_arr < valley_thresh
                    if valley_thresh == 0: 
                        strengths = np.where(signal_arr < valley_thresh, 1.0, 0.0) 
                    else:
                        strengths = (valley_thresh - signal_arr[indices_mask]) / valley_thresh
                    symbol = 'triangle-down' if is_actual else 'triangle-left'
                    name = 'Actual Valleys' if is_actual else 'Predicted Valleys'
                    color = 'blue' if is_actual else 'violet' 

                marker_times = time_arr[indices_mask].reset_index(drop=True)
                marker_values = price_arr[indices_mask].reset_index(drop=True)
                
                # Scale marker size based on strength
                scaled_sizes = min_marker_size + (max_marker_size - min_marker_size) * np.clip(strengths, 0, 1)

                return go.Scatter(
                    x=marker_times,
                    y=marker_values,
                    mode='markers',
                    marker=dict(color=color, symbol=symbol, size=scaled_sizes, line=dict(color='white', width=1)),
                    name=name,
                    showlegend=True 
                )

            # Add traces for actual peaks and valleys
            fig.add_trace(_get_signal_markers(actual_signal_flat, time_test, close_test,
                                              actual_peak_threshold, actual_valley_threshold,
                                              is_peak=True, is_actual=True))
            fig.add_trace(_get_signal_markers(actual_signal_flat, time_test, close_test,
                                              actual_peak_threshold, actual_valley_threshold,
                                              is_peak=False, is_actual=True))

            # Add traces for predicted peaks and valleys
            fig.add_trace(_get_signal_markers(predicted_signal_flat, time_test, close_test,
                                              predicted_peak_threshold, predicted_valley_threshold,
                                              is_peak=True, is_actual=False))
            fig.add_trace(_get_signal_markers(predicted_signal_flat, time_test, close_test,
                                              predicted_peak_threshold, predicted_valley_threshold,
                                              is_peak=False, is_actual=False))


            # Update layout
            fig.update_layout(
                title_text=title,
                xaxis_title='Time',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                template='plotly_dark', 
                hovermode='x unified',
                height=500,
                width=1000,
                margin=dict(l=15,r=15,b=10),
                font=dict(size=10,color="#e1e1e1"),
                paper_bgcolor="#1e1e1e",
                plot_bgcolor="#1e1e1e",
                legend_title_text="Elements",
                showlegend=True 
            )

            # Update X-axis
            fig.update_xaxes(
                gridcolor="#1f292f",
                showgrid=True,
                fixedrange=False,
                rangeslider=dict(visible=False),
                rangebreaks=[
                    dict(bounds=["sat", "mon"])
                ]
            )
            
            # Update Y-axis
            fig.update_layout(
                yaxis=dict(
                    autorange=True,
                    fixedrange=False,
                    gridcolor="#1f292f",
                    showgrid=True
                )
            )
            
            fig.show()

        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to plot signals with candlesticks")


        
    @staticmethod
    def debug_report(y_train: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
                     y_pred_classes: Optional[np.ndarray] = None, y_pred_proba: Optional[np.ndarray] = None,
                     label_encoder: Optional[LabelEncoder] = None, imbalance: bool = False) -> None:
        """
        Generate a debugging report for classification models.

        Args:
            y_train: Training labels.
            y_test: Test labels.
            y_pred_classes: Predicted class labels.
            y_pred_proba: Predicted probabilities.
            label_encoder: Fitted LabelEncoder.
            imbalance: Whether the dataset is imbalanced.

        Raises:
            MLToolkitError: If report generation fails.
        """
        try:
            Visualizer.print_hline_with_background('Class Classifications:')
            titles = ['Training Set', 'Test Set', 'Predicted Set']
            non_none_data = [(y, title) for y, title in zip([y_train, y_test, y_pred_classes], titles) if y is not None]
            if non_none_data:
                filtered_data = [data[0] for data in non_none_data]
                filtered_titles = [data[1] for data in non_none_data]
                Visualizer.plot_pie_charts(filtered_data, filtered_titles)

            Visualizer.print_hline_with_background('Confusion Matrix:')
            Visualizer.plot_confusion_matrix(y_test, y_pred_classes, title="Confusion Matrix (with SMOTEENN)")

            Visualizer.print_hline_with_background('Debugging Report:')
            print(classification_report(y_test, y_pred_classes, target_names=['Sell', 'Neutral', 'Buy'], zero_division=0))
            Visualizer.print_hline(' ', line_position='middle', line_thickness='1px')

            if y_train is not None:
                print(f"{'Balanced' if imbalance else 'Original'} training label distribution:", Counter(y_train))
                print("Minimum label in y_train:", np.min(y_train))
                print("Unique classes in y_train:", np.unique(y_train))
                Visualizer.print_hline(' ', line_position='middle', line_thickness='1px')

            print("Minimum label in y_test:", np.min(y_test))
            print("Test label distribution:", Counter(y_test))
            print("Unique classes in y_test:", np.unique(y_test))
            print("Unique classes in y_pred_classes:", np.unique(y_pred_classes))
            print("Classes in LabelEncoder:", label_encoder.classes_)
            Visualizer.print_hline(' ', line_position='middle', line_thickness='1px')
            print("Sample y_pred_proba:", y_pred_proba[:5])
            print("Sample y_pred_classes:", y_pred_classes[:5])
            Visualizer.print_hline(' ', line_position='middle', line_thickness='1px')
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to generate debug report")


