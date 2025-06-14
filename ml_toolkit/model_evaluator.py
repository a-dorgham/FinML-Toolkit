# ---------------------------------------------
# FINANCIAL MACHINE LEARNING TOOLKIT
# ---------------------------------------------

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix, f1_score, r2_score
from sklearn.impute import SimpleImputer
import tensorflow as tf
from collections import Counter
from scipy.optimize import minimize

from ml_toolkit.error_handler import ErrorHandler, MLToolkitError


# ---------------------------------------------
# MODEL PREDICTION AND EVALUATION
# ---------------------------------------------


class ModelEvaluator:
    """Evaluates model performance and generates predictions."""

    @staticmethod
    def predict_data(model: Any, X_test_scaled: np.ndarray, y_test_scaled: np.ndarray,
                     scaler_y: MinMaxScaler) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions and inverse transform to original scale.

        Args:
            model: Trained model.
            X_test_scaled: Scaled test features.
            y_test_scaled: Scaled test labels.
            scaler_y: Scaler for inverse transforming predictions.

        Returns:
            Tuple of inverse-transformed test and predicted values.

        Raises:
            MLToolkitError: If prediction fails.
        """
        try:
            y_pred_scaled = model.predict(X_test_scaled)
            if len(y_pred_scaled.shape) == 1:
                y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_test = scaler_y.inverse_transform(y_test_scaled)
            imputer = SimpleImputer(strategy='mean')
            y_test_imputed = imputer.fit_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_imputed = imputer.transform(y_pred.reshape(-1, 1)).flatten()
            return y_test_imputed, y_pred_imputed
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to generate predictions")

    @staticmethod
    def evaluate_model(y_test_imputed: np.ndarray, y_pred_imputed: np.ndarray, debug: bool = False) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Evaluate model performance for classification.

        Args:
            y_test_imputed: True test labels.
            y_pred_imputed: Predicted labels or probabilities.
            debug: Whether to print detailed evaluation metrics.

        Returns:
            Tuple containing accuracy, precision, recall, f1, confusion matrix, MSE, RMSE.

        Raises:
            MLToolkitError: If evaluation fails.
        """
        try:
            mse = mean_squared_error(y_test_imputed, y_pred_imputed)
            rmse = np.sqrt(mse)
            if len(y_pred_imputed.shape) == 2:
                y_pred_classes = np.argmax(y_pred_imputed, axis=1)
            else:
                y_pred_classes = y_pred_imputed
            accuracy = accuracy_score(y_test_imputed, y_pred_classes)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test_imputed, y_pred_classes, average=None, zero_division=0)
            confusion = confusion_matrix(y_test_imputed, y_pred_classes)
            if debug:
                print(f'Accuracy: {accuracy:.4f}')
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1-Score:", f1)
                print("Classification Report:")
                print(classification_report(y_test_imputed, y_pred_classes, target_names=['Sell', 'Neutral', 'Buy'], zero_division=0))
                print("Confusion Matrix:")
                print(confusion)
                print("True Labels Count:", Counter(y_test_imputed))
                print("Predicted Labels Count:", Counter(y_pred_classes))
            return accuracy, precision, recall, f1, confusion, mse, rmse
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to evaluate model")

    @staticmethod
    def adjust_predictions_with_regularization(y_pred_proba: np.ndarray, class_weights: Dict[int, float]) -> np.ndarray:
        """
        Adjust predicted probabilities with class-specific weights.

        Args:
            y_pred_proba: Predicted probabilities.
            class_weights: Dictionary of class weights.

        Returns:
            np.ndarray: Adjusted probabilities.

        Raises:
            MLToolkitError: If adjustment fails.
        """
        try:
            adjusted_proba = y_pred_proba * np.array([class_weights[i] for i in range(len(class_weights))])
            adjusted_proba /= np.sum(adjusted_proba, axis=1, keepdims=True)
            return adjusted_proba
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to adjust predictions")

    @staticmethod
    def class_specific_loss(class_weights: Dict[int, float] = {0: 2.0, 1: 0.5, 2: 0.5}) -> callable:
        """
        Create a custom loss function with class-specific regularization.

        Args:
            class_weights: Dictionary of class weights.

        Returns:
            Callable loss function.

        Raises:
            MLToolkitError: If loss function creation fails.
        """
        try:
            def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
                ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
                regularization = 0
                for cls, weight in class_weights.items():
                    cls_mask = tf.cast(tf.keras.backend.equal(y_true, cls), dtype=tf.float32)
                    regularization += weight * tf.keras.backend.sum(cls_mask * tf.square(1 - y_pred[:, cls]))
                return ce_loss + regularization
            return loss
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to create class-specific loss")

    @staticmethod
    def focal_loss(alpha: float = 0.25, gamma: float = 2.0) -> callable:
        """
        Create a focal loss function for imbalanced classification.

        Args:
            alpha: Scaling factor for focal loss.
            gamma: Focusing parameter for focal loss.

        Returns:
            Callable loss function.

        Raises:
            MLToolkitError: If loss function creation fails.
        """
        try:
            def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
                y_true = tf.keras.backend.cast(y_true, tf.float32)
                y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
                cross_entropy = -y_true * tf.keras.backend.log(y_pred)
                weight = alpha * tf.keras.backend.pow(1 - y_pred, gamma)
                return tf.keras.backend.mean(weight * cross_entropy)
            return loss
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to create focal loss")

    @staticmethod
    def get_initial_weights(y_test: np.ndarray, y_pred_proba: Optional[np.ndarray] = None,
                            scaling_factor: float = 1.0) -> Dict[int, float]:
        """
        Derive initial class weights based on class distribution.

        Args:
            y_test: True test labels.
            y_pred_proba: Predicted probabilities (optional).
            scaling_factor: Scaling factor for weights.

        Returns:
            Dictionary of initial class weights.

        Raises:
            MLToolkitError: If weight calculation fails.
        """
        try:
            class_counts = Counter(y_test)
            total_count = sum(class_counts.values())
            inverse_weights = {cls: (total_count / count) * scaling_factor for cls, count in class_counts.items()}
            if y_pred_proba is not None:
                avg_pred_proba = y_pred_proba.mean(axis=0)
                pred_bias_factor = {cls: scaling_factor / prob if prob > 0 else 1.0 for cls, prob in enumerate(avg_pred_proba)}
                combined_weights = {cls: inverse_weights.get(cls, 1.0) * pred_bias_factor.get(cls, 1.0) for cls in range(y_pred_proba.shape[1])}
            else:
                combined_weights = inverse_weights
            max_weight = max(combined_weights.values())
            return {cls: weight / max_weight for cls, weight in combined_weights.items()}
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to calculate initial weights")



    @staticmethod
    def optimize_class_weights(y_true: np.ndarray, y_pred_proba: np.ndarray, metric: str = "f1",
                              initial_weights: Optional[Dict[int, float]] = None,
                              bounds: Optional[List[Tuple[float, float]]] = None,
                              optimization_method: Optional[str] = None) -> Tuple[Dict[int, float], float]:
        """
        Optimize class weights to maximize a metric.

        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.
            metric: Metric to optimize ('f1' or 'accuracy').
            initial_weights: Initial class weights.
            bounds: Bounds for each class weight.
            optimization_method: Optimization method for scipy.optimize.minimize.

        Returns:
            Tuple of optimized class weights and best metric score.

        Raises:
            MLToolkitError: If optimization fails.
        """
        try:
            def objective(weights):
                # Ensure weights are treated as numpy array for calculations
                weights = np.asarray(weights)
                adjusted_proba = y_pred_proba * weights
                sum_proba = adjusted_proba.sum(axis=1, keepdims=True)
                adjusted_proba = adjusted_proba / (sum_proba + 1e-9) 
                y_pred_adjusted = adjusted_proba.argmax(axis=1)

                # Defensive check for NaNs/Infs after division and argmax
                if np.isnan(y_pred_adjusted).any() or np.isinf(y_pred_adjusted).any():
                    return np.inf 
                
                score = 0.0
                if metric == "f1":
                    score = f1_score(y_true, y_pred_adjusted, average="weighted", zero_division=0)
                elif metric == "accuracy":
                    score = accuracy_score(y_true, y_pred_adjusted)
                else:
                    raise MLToolkitError(f"Unsupported metric: {metric}")
                
                if np.isnan(score):
                    return np.inf

                return -score 

            num_classes = y_pred_proba.shape[1]

            # Default initial weights: ensure they are well within typical bounds.
            initial_weights_dict = initial_weights or {cls: 1.0 for cls in range(num_classes)}
            initial_weights_list = list(initial_weights_dict.values())

            bounds_list = bounds or [(0.1, 10.0)] * num_classes
            bounds_list = [(float(l), float(u)) for l, u in bounds_list]

            # Clamp initial_weights_list to bounds
            clamped_initial_weights_list = []
            for i, val in enumerate(initial_weights_list):
                lower, upper = bounds_list[i]
                clamped_val = np.clip(val, lower + 1e-9, upper - 1e-9) 
                clamped_initial_weights_list.append(clamped_val)
            
            result = minimize(objective, x0=clamped_initial_weights_list, bounds=bounds_list, method=optimization_method)

            if not result.success:
                print(f"Warning: Optimization for method '{optimization_method}' did not succeed. Message: {result.message}")
            if np.isnan(result.x).any() or np.isinf(result.x).any():
                 raise MLToolkitError("Optimization resulted in NaN or Inf weights.")
            if np.isnan(result.fun) or np.isinf(result.fun):
                raise MLToolkitError("Optimization objective resulted in NaN or Inf score.")

            optimized_weights = {cls: weight for cls, weight in enumerate(result.x)}
            return optimized_weights, -result.fun 
        except Exception as e:
            if isinstance(e, ValueError) and "`x0` violates bound constraints." in str(e):
                raise MLToolkitError(f"Failed to optimize class weights. Initial weights or bounds might be ill-defined: {str(e)}")
            else:
                ErrorHandler.handle_error(e, "Failed to optimize class weights")
                
    def evaluate_model_performance(self, y_test_original: pd.DataFrame, predictions: pd.DataFrame, target_cols: list, visualization: bool = True):
        """
        Evaluates the performance of the model and optionally visualizes predictions.

        Parameters:
            y_test_original (pd.DataFrame): Original actual target values.
            predictions (pd.DataFrame): Predicted target values.
            target_cols (list): List of target column names.
            visualization (bool): Whether to visualize predicted vs actual values.
        """
        if not isinstance(y_test_original, (pd.DataFrame, np.ndarray)) or not isinstance(predictions, (pd.DataFrame, np.ndarray)):
            raise TypeError("y_test_original and predictions must be pandas DataFrame or numpy array.")
        if y_test_original.shape != predictions.shape:
            raise ValueError("Shape of y_test_original and predictions must be the same.")

        print("\nModel Performance:")
        print("R² Score:", r2_score(y_test_original, predictions))
        print("RMSE:", np.sqrt(mean_squared_error(y_test_original, predictions)))

        if visualization:
            self.visualizer.plot_predictions_vs_actual(y_test_original, predictions, target_cols)