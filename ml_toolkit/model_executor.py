# ---------------------------------------------
# FINANCIAL MACHINE LEARNING TOOLKIT
# ---------------------------------------------

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from catboost import CatBoostClassifier
# import xgboost as xgb
# import lightgbm as lgb

from ml_toolkit.error_handler import ErrorHandler, MLToolkitError
from ml_toolkit.feature_engineer import FeatureEngineer
from ml_toolkit.data_handler import DataHandler
from ml_toolkit.model_builder import ModelBuilder
from ml_toolkit.model_evaluator import ModelEvaluator
from ml_toolkit.visualizer import Visualizer

# ---------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------


class ModelExecutor:
    """Main class for executing financial machine learning workflows."""

    @staticmethod
    def execute_model(file_path: str = './data/GBP_USD_M15.pkl', start_date: str = "2024-7-1",
                     end_date: str = "2024-7-23", test_size: float = 0.1, features: Optional[List[str]] = None,
                     imbalance: bool = True, imb_type: str = 'smote', model_type: str = 'LSTM',
                     model_style: str = 'Sequential', model_layers: int = 2, epochs: int = 10,
                     batch_size: int = 32, units: int = 50, dropout_rate: float = 0.2,
                     learning_rate: float = 0.001, plot_type: Optional[str] = None,
                     forecast_period_min: int = 60, num_classes: int = 3) -> Tuple[Any, pd.DataFrame, Dict[int, float], MinMaxScaler, LabelEncoder]:
        """
        Execute the full machine learning workflow for financial data.

        Args:
            file_path: Path to data file.
            start_date: Start date for data.
            end_date: End date for data.
            test_size: Proportion of test data.
            features: List of feature names.
            imbalance: Whether to handle class imbalance.
            imb_type: Imbalance handling method.
            model_type: Model type ('LSTM', 'randforest', 'xgboost', 'lightgbm', 'catboost').
            model_style: Keras model style ('Sequential' or 'Functional').
            model_layers: Number of model layers.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            units: Number of units in layers.
            dropout_rate: Dropout rate.
            learning_rate: Learning rate.
            plot_type: Type of plot ('test', 'train', 'test+train', 'forecast').
            forecast_period_min: Forecast period in minutes.
            num_classes: Number of output classes.

        Returns:
            Tuple containing trained model, DataFrame, optimized class weights, feature scaler, and label encoder.

        Raises:
            MLToolkitError: If execution fails.
        """
        try:
            # Load and preprocess data
            df = DataHandler.load_data(file_path, start_date, end_date)
            df, time, X, y = FeatureEngineer.add_features(df, features)
            DataHandler.check_nan(X, y)

            # Split data
            X_train_orig, X_test_orig, y_train_orig, y_test_orig, train_indices, test_indices = DataHandler.split_data(time, X, y, test_size)

            # Handle imbalance
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train_orig)
            y_test_encoded = label_encoder.transform(y_test_orig)

            if imbalance:
                sampler = {
                    'smote': SMOTE(random_state=42), 
                    'adasyn': ADASYN(random_state=42),
                    'random_oversampler': RandomOverSampler(random_state=42),
                    'random_undersampler': RandomUnderSampler(random_state=42),
                    'tomek_links': TomekLinks(), 
                    'nearmiss': NearMiss(),
                    'smote_tomek': SMOTETomek(random_state=42), 
                    'smote_enn': SMOTEENN(random_state=42)
                }.get(imb_type, None)
                if not sampler:
                    raise MLToolkitError(f"Unsupported imbalance type: {imb_type}")
                X_train, y_train = sampler.fit_resample(X_train_orig, y_train_encoded)
            else:
                X_train, y_train = X_train_orig, y_train_encoded
            X_test, y_test = X_test_orig, y_test_encoded

            # Scale data
            X_train_scaled, scaler_X = DataHandler.scale_data(X_train, 'train', reshape=True)
            X_test_scaled, _ = DataHandler.scale_data(X_test, 'test', scaler=scaler_X, reshape=True)

            # Build and train model
            if model_type.lower() == 'randforest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(class_weight='balanced', random_state=42)
                model.fit(X_train_scaled, y_train)
            elif model_type.lower() == 'xgboost':
                model = xgb.XGBClassifier(scale_pos_weight=0.5, random_state=42)
                model.fit(X_train_scaled, y_train)
            elif model_type.lower() == 'lightgbm':
                model = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
                model.fit(X_train_scaled, y_train)
            elif model_type.lower() == 'catboost':
                model = CatBoostClassifier(class_weights=[1, 10, 20], random_state=42, verbose=0)
                model.fit(X_train_scaled, y_train)
            else:
                model = ModelBuilder.create_flexible_model(
                    input_shape=(X_train_scaled.shape[1], 1), model_type=model_type, num_layers=model_layers,
                    model_style=model_style, units=units, learning_rate=learning_rate, dropout_rate=dropout_rate,
                    num_classes=num_classes
                )
                Visualizer.print_hline_with_background('Model Training Results:')
                ModelBuilder.train_model(model, X_train_scaled, y_train, epochs, batch_size, plot_epochs=True)

            # Predict and evaluate
            y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else model.predict(X_test_scaled)
            initial_class_weights = ModelEvaluator.get_initial_weights(y_test, y_pred_proba)
            optimized_class_weights, best_f1_score = ModelEvaluator.optimize_class_weights(
                y_true=y_test, y_pred_proba=y_pred_proba, metric="accuracy", initial_weights=initial_class_weights,
                bounds=[(0.1, 20.0)] * 3
            )
            y_pred_adjusted = ModelEvaluator.adjust_predictions_with_regularization(y_pred_proba, optimized_class_weights)
            y_pred_classes = np.argmax(y_pred_adjusted, axis=1)
            y_pred_reconstructed = label_encoder.inverse_transform(y_pred_classes)

            # Debug report
            Visualizer.debug_report(y_train, y_test, y_pred_classes, y_pred_proba, label_encoder, imbalance)
            print("Initial class weights:", initial_class_weights)
            print("Optimized Weights:", optimized_class_weights)
            print("Best accuracy score:", best_f1_score)
            Visualizer.print_hline(' ', line_position='middle', line_thickness='1px')

            # Evaluate model
            accuracy, precision, recall, f1, confusion, mse, rmse = ModelEvaluator.evaluate_model(y_test, y_pred_classes)

            # Plot predictions
            if plot_type:
                Visualizer.print_hline_with_background('Model Prediction:')
                Visualizer.plot_prediction(df, test_indices, y_pred_reconstructed, plot_type=plot_type, period_min=forecast_period_min, model=model)

            return model, mse, df, y_test, y_pred_classes, test_indices, label_encoder, optimized_class_weights, scaler_X#train_indices, test_indices
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to execute model workflow.")

    @staticmethod
    def scale_predict_data(model: Any, class_weights: Dict[int, float], scaler_X: MinMaxScaler,
                          label_encoder: LabelEncoder, X_test: pd.DataFrame,
                          y_test: Optional[np.ndarray] = None, plot: bool = False,
                          optimization_method: str = 'L-BFGS-B') -> Tuple[np.ndarray, Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float]]:
        """
        Scale data and generate predictions.

        Args:
            model: Trained model.
            class_weights: Class weights for prediction adjustment.
            scaler_X: Scaler for features.
            label_encoder: Encoder for labels.
            X_test: Test feature matrix.
            y_test: Test labels (optional).
            plot: Whether to plot predictions.
            optimization_method: Optimization method for weight optimization.

        Returns:
            Tuple containing predicted labels and evaluation metrics (if y_test provided).

        Raises:
            MLToolkitError: If prediction fails.
        """
        try:
            X_test_scaled, _ = DataHandler.scale_data(X_test, 'test', scaler=scaler_X, reshape=True)
            y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else model.predict(X_test_scaled)
            initial_class_weights = ModelEvaluator.get_initial_weights(y_test, y_pred_proba, scaling_factor=10) if y_test is not None else {0: 10.0, 1: 50.0, 2: 10.0}
            optimized_class_weights, best_f1_score = ModelEvaluator.optimize_class_weights(
                y_true=y_test, y_pred_proba=y_pred_proba, metric="accuracy", initial_weights=initial_class_weights,
                bounds=[(0.2, 100.0)] * 3, optimization_method=optimization_method
            )
            class_weights[1] = 5
            y_pred_adjusted = ModelEvaluator.adjust_predictions_with_regularization(y_pred_proba, optimized_class_weights)
            y_pred_classes = np.argmax(y_pred_adjusted, axis=1)
            y_pred_reconstructed = label_encoder.inverse_transform(y_pred_classes)

            Visualizer.debug_report(None, y_test, y_pred_classes, y_pred_proba, label_encoder, imbalance=False)
            print("Used class weights:", class_weights)
            print("Initial class weights:", initial_class_weights)
            print("Optimized Weights:", optimized_class_weights)
            print("Best accuracy score:", best_f1_score)
            Visualizer.print_hline(' ', line_position='middle', line_thickness='1px')

            if plot:
                Visualizer.plot_with_peaks(X_test, plot_data=True, plot_ypred=True, fig_show=True)

            if y_test is not None:
                accuracy, precision, recall, f1, confusion, mse, rmse = ModelEvaluator.evaluate_model(y_test, y_pred_classes)
                return y_pred_reconstructed, accuracy, precision, recall, f1, confusion, mse, rmse
            return y_pred_reconstructed, None, None, None, None, None, None, None
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to scale and predict data")