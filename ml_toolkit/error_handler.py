import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union, List, Dict, Optional, Any

class MLToolkitError(Exception):
    """
    Custom exception class for the PredictiveFinance_ML Toolkit.
    All errors raised by the toolkit's internal components should be wrapped in this exception.
    """
    pass

class ErrorHandler:
    """
    Centralized error handling and validation utilities for the PredictiveFinance_ML Toolkit.
    This class provides static methods to validate inputs and handle exceptions consistently.
    """

    @staticmethod
    def handle_error(exception: Exception, message: str = "An unexpected error occurred.") -> None:
        """
        Logs the original exception and re-raises a custom MLToolkitError.

        This method acts as a central point for catching exceptions, logging them
        with a descriptive message, and then raising a uniform custom exception
        (`MLToolkitError`) throughout the toolkit.

        Args:
            exception (Exception): The original exception object caught.
            message (str): A descriptive message that provides context for the error.

        Raises:
            MLToolkitError: A custom exception containing the contextual message
                            and information about the original error.
        """
        full_message = f"{message} Original error: {type(exception).__name__}: {exception}"
        print(f"ERROR: {full_message}")
        raise MLToolkitError(full_message)

    @staticmethod
    def validate_not_empty(data: Union[pd.DataFrame, np.ndarray, List, Dict, Any], name: str) -> None:
        """
        Validates that a given data structure is not empty.

        This method checks if a pandas DataFrame, NumPy array, Python list, dictionary,
        or other iterable/sized object is empty. It's crucial for ensuring that
        functions don't proceed with empty inputs, which could lead to errors.

        Args:
            data: The data structure to check for emptiness. Supported types include
                  pd.DataFrame, np.ndarray, list, dict, and any object with `__len__` or `size` attribute.
            name (str): A descriptive name for the data being validated (e.g., "Training features",
                        "Filtered DataFrame"). This name is used in error messages.

        Raises:
            MLToolkitError: If the data structure is found to be empty or None.
        """
        try:
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    raise ValueError(f"{name} DataFrame is empty.")
            elif isinstance(data, np.ndarray):
                if data.size == 0:
                    raise ValueError(f"{name} NumPy array is empty.")
            elif isinstance(data, (list, dict)):
                if not data: 
                    raise ValueError(f"{name} is empty.")
            elif data is None: 
                raise ValueError(f"{name} is None.")
            # Generic checks for other types that might be considered "empty"
            elif hasattr(data, '__len__') and len(data) == 0:
                raise ValueError(f"{name} is empty.")
            elif hasattr(data, 'size') and data.size == 0: 
                raise ValueError(f"{name} is empty.")

        except Exception as e:
            ErrorHandler.handle_error(e, f"Validation failed for '{name}' emptiness check")

    @staticmethod
    def validate_file_exists(file_path: str) -> None:
        """
        Validates that a file at the given path exists on the filesystem.

        This is used before attempting to read data from a file, preventing
        `FileNotFoundError` exceptions from propagating directly.

        Args:
            file_path (str): The absolute or relative path to the file.

        Raises:
            MLToolkitError: If the file specified by `file_path` does not exist.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found at: '{file_path}'")
        except Exception as e:
            ErrorHandler.handle_error(e, f"File existence validation failed for path: '{file_path}'")

    @staticmethod
    def validate_columns_exist(df: pd.DataFrame, columns: List[str]) -> None:
        """
        Validates that all specified column names are present in a pandas DataFrame.

        This ensures that subsequent operations on the DataFrame (like column selection)
        do not fail due to missing columns.

        Args:
            df (pd.DataFrame): The DataFrame to inspect.
            columns (List[str]): A list of column names that are expected to be in `df`.

        Raises:
            MLToolkitError: If one or more required columns are missing from the DataFrame.
        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input 'df' must be a pandas DataFrame.")
            if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
                raise TypeError("Input 'columns' must be a list of strings.")

            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in DataFrame: {', '.join(missing_cols)}")
        except Exception as e:
            ErrorHandler.handle_error(e, "Column existence validation failed")

    @staticmethod
    def validate_date_order(start: datetime, end: datetime) -> None:
        """
        Validates that the start datetime is not strictly after the end datetime.

        This is essential for correctly defining time ranges and preventing
        logical errors in data filtering.

        Args:
            start (datetime): The start datetime object.
            end (datetime): The end datetime object.

        Raises:
            MLToolkitError: If the `start` datetime is after the `end` datetime.
        """
        try:
            if not isinstance(start, datetime) or not isinstance(end, datetime):
                raise TypeError("Start and end arguments must be datetime objects.")
            if start > end:
                raise ValueError(f"Start date ({start.strftime('%Y-%m-%d %H:%M:%S')}) cannot be after end date ({end.strftime('%Y-%m-%d %H:%M:%S')}).")
        except Exception as e:
            ErrorHandler.handle_error(e, "Date order validation failed")

    @staticmethod
    def validate_date_in_range(date_to_check: datetime, min_date: datetime, max_date: datetime, date_name: str) -> None:
        """
        Validates that a specific datetime falls within an inclusive minimum and maximum range.

        This is useful for ensuring that user-provided dates for filtering or analysis
        are sensible and fall within the bounds of available data.

        Args:
            date_to_check (datetime): The datetime object whose value is being validated.
            min_date (datetime): The minimum allowable datetime (inclusive).
            max_date (datetime): The maximum allowable datetime (inclusive).
            date_name (str): A descriptive name for the date being checked (e.g., "Start date", "End date").
                             Used in error messages.

        Raises:
            MLToolkitError: If `date_to_check` is outside the specified `min_date` and `max_date` range.
        """
        try:
            if not isinstance(date_to_check, datetime) or not isinstance(min_date, datetime) or not isinstance(max_date, datetime):
                raise TypeError("All date arguments must be datetime objects.")
            if not (min_date <= date_to_check <= max_date):
                raise ValueError(f"{date_name} ({date_to_check.strftime('%Y-%m-%d %H:%M:%S')}) must be within the data's available range "
                                 f"({min_date.strftime('%Y-%m-%d %H:%M:%S')} to {max_date.strftime('%Y-%m-%d %H:%M:%S')}).")
        except Exception as e:
            ErrorHandler.handle_error(e, f"Date range validation failed for '{date_name}'")
            

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> None:
        """
        Performs comprehensive validation on a pandas DataFrame.

        This method checks the following:
        1. Ensures the input is indeed a pandas DataFrame.
        2. Ensures the DataFrame is not empty.
        3. If `required_columns` are provided, ensures all specified columns exist in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            required_columns (Optional[List[str]]): A list of column names that must be present
                                                    in the DataFrame. If None, column existence
                                                    is not checked by this function (though
                                                    other functions might require specific columns).

        Raises:
            MLToolkitError: If the input is not a DataFrame, the DataFrame is empty,
                            or any required columns are missing.
        """
        try:
            # Validate input type
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input 'df' must be a pandas DataFrame.")

            # Validate DataFrame is not empty
            ErrorHandler.validate_not_empty(df, "Input DataFrame")

            # Validate required columns if provided
            if required_columns:
                ErrorHandler.validate_columns_exist(df, required_columns)
                
        except Exception as e:
            ErrorHandler.handle_error(e, "DataFrame validation failed")            