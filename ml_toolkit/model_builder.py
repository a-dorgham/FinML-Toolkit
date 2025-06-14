# ---------------------------------------------
# FINANCIAL MACHINE LEARNING TOOLKIT
# ---------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union, Any
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

from ml_toolkit.error_handler import ErrorHandler, MLToolkitError

# ---------------------------------------------
# MODEL ENGINEERING
# ---------------------------------------------

class ModelBuilder:
    """Builds and trains machine learning models for financial data."""

    @staticmethod
    def train_model(model: Any, X_train_scaled: np.ndarray, y_train: np.ndarray, epochs: int = 10,
                    batch_size: int = 32, plot_epochs: bool = False) -> Optional[tf.keras.callbacks.History]:
        """
        Train a model. This function is generalized for both Keras and sklearn models.

        Args:
            model: Model to train (Keras, sklearn, or others).
            X_train_scaled: Scaled training features.
            y_train: Training labels/targets.
            epochs: Number of training epochs (for Keras models).
            batch_size: Batch size for training (for Keras models).
            plot_epochs: Whether to plot training history (for Keras models).

        Returns:
            Optional History object for Keras models, None for sklearn models.

        Raises:
            MLToolkitError: If model training fails or unsupported model type.
        """
        try:
            ErrorHandler.validate_not_empty(X_train_scaled, "Training features")
            ErrorHandler.validate_not_empty(y_train, "Training labels/targets")

            if isinstance(model, tf.keras.Model):

                if len(y_train.shape) > 1 and y_train.shape[1] == 1:
                    y_train_flat = y_train.flatten()
                else:
                    y_train_flat = y_train 

                classes = np.unique(y_train_flat)
                class_weights = compute_class_weight('balanced', classes=classes, y=y_train_flat)
                class_weight_dict = dict(enumerate(class_weights))
                
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=3, restore_best_weights=True
                )
                history = model.fit(
                    X_train_scaled, y_train, epochs=epochs, batch_size=batch_size,
                    validation_split=0.1, callbacks=[early_stopping], class_weight=class_weight_dict
                )
                if plot_epochs:
                    ModelBuilder.plot_epochs_history(history)
                return history
            elif hasattr(model, 'fit'): 
                model.fit(X_train_scaled, y_train)
                return None
            else:
                raise MLToolkitError("Unsupported model type for training.")
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to train model")

    @staticmethod
    def plot_epochs_history(history: tf.keras.callbacks.History) -> None:
        """
        Plot training loss history for Keras models.

        Args:
            history: Keras History object.

        Raises:
            MLToolkitError: If plotting fails.
        """
        try:
            plt.plot(history.history['loss'])
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to plot training history")

    @staticmethod
    def create_flexible_model(input_shape: Tuple[int, ...], model_type: str = 'LSTM', units: int = 50,
                             dropout_rate: float = 0.2, learning_rate: float = 0.001, num_heads: int = 4,
                             key_dim: int = 64, num_layers: int = 2, model_style: str = 'Sequential',
                             num_classes: int = 3, output_activation: str = 'softmax') -> tf.keras.Model:
        """
        Create a flexible Keras model. Can be used for classification (default) or regression.

        Args:
            input_shape: Shape of input data.
            model_type: Type of layers ('LSTM', 'GRU', 'Conv1D', 'Transformer', 'Attention', or combinations).
            units: Number of units in recurrent or dense layers.
            dropout_rate: Dropout rate for regularization.
            learning_rate: Learning rate for optimizer.
            num_heads: Number of attention heads (for Transformer).
            key_dim: Size of attention heads (for Transformer).
            num_layers: Number of main sequential layers.
            model_style: 'Sequential' or 'Functional'.
            num_classes: Number of output units. For regression, set to 1 or `len(target)`.
            output_activation: Activation function for the output layer ('softmax' for classification,
                               'linear' for regression, 'sigmoid' for binary classification).

        Returns:
            tf.keras.Model: Compiled Keras model.

        Raises:
            MLToolkitError: If model creation fails or unsupported layer type/model style.
        """
        try:
            layers_list = model_type.split('+')
            
            # Determine loss and metrics based on output_activation
            if output_activation == 'softmax':
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
            elif output_activation == 'sigmoid':
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            elif output_activation == 'linear':
                loss = 'mse'
                metrics = ['mae']
            else:
                raise MLToolkitError(f"Unsupported output_activation: {output_activation}")

            if model_style == 'Sequential':
                model = tf.keras.Sequential([tf.keras.layers.InputLayer(shape=input_shape)])
                for i in range(num_layers):
                    for layer in layers_list:
                        if layer == 'LSTM':
                            model.add(tf.keras.layers.LSTM(units=units, return_sequences=(i < num_layers - 1)))
                            model.add(tf.keras.layers.Dropout(rate=dropout_rate))
                        elif layer == 'GRU':
                            model.add(tf.keras.layers.GRU(units=units, return_sequences=(i < num_layers - 1)))
                            model.add(tf.keras.layers.Dropout(rate=dropout_rate))
                        elif layer == 'Conv1D':
                            model.add(tf.keras.layers.Conv1D(filters=units, kernel_size=3, activation='relu', padding='same'))
                            model.add(tf.keras.layers.Dropout(rate=dropout_rate))
                        elif layer == 'Attention':
                            if len(model.layers[-1].output_shape) == 3: 
                                model.add(tf.keras.layers.Attention(use_scale=False)([model.layers[-1].output, model.layers[-1].output])) 
                            else:
                                raise MLToolkitError("Attention layer requires a sequence input.")
                        elif layer == 'Transformer':
                            if len(model.layers[-1].output_shape) == 3: 
                                x_temp = model.layers[-1].output
                                attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_temp, x_temp)
                                attn_output = tf.keras.layers.LayerNormalization()(x_temp + attn_output) 
                                ffn_output = tf.keras.layers.Dense(units, activation='relu')(attn_output)
                                ffn_output = tf.keras.layers.Dense(x_temp.shape[-1])(ffn_output) 
                                model.add(tf.keras.layers.LayerNormalization()(attn_output + ffn_output)) 
                                model.add(tf.keras.layers.Dropout(rate=dropout_rate))
                            else:
                                raise MLToolkitError("Transformer block requires a sequence input.")
                        elif layer == 'Dense':
                            if len(model.layers[-1].output_shape) == 3:
                                model.add(tf.keras.layers.Flatten()) 
                            model.add(tf.keras.layers.Dense(units=units, activation='relu'))
                            model.add(tf.keras.layers.Dropout(rate=dropout_rate))
                        else:
                            raise MLToolkitError(f"Unsupported layer type: {layer}")
                
                if len(model.layers[-1].output_shape) == 3:
                    model.add(tf.keras.layers.Flatten()) 

                model.add(tf.keras.layers.Dense(units=num_classes, activation=output_activation))

            elif model_style == 'Functional':
                inputs = tf.keras.layers.Input(shape=input_shape)
                x = inputs
                for i in range(num_layers):
                    for layer in layers_list:
                        if layer == 'LSTM':
                            x = tf.keras.layers.LSTM(units=units, return_sequences=(i < num_layers - 1))(x)
                            x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
                        elif layer == 'GRU':
                            x = tf.keras.layers.GRU(units=units, return_sequences=(i < num_layers - 1))(x)
                            x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
                        elif layer == 'Conv1D':
                            x = tf.keras.layers.Conv1D(filters=units, kernel_size=3, activation='relu', padding='same')(x)
                            x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
                        elif layer == 'Attention':
                            x = tf.keras.layers.Attention(use_scale=False)([x, x]) 
                        elif layer == 'Transformer':
                            attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
                            attn_output = tf.keras.layers.LayerNormalization()(x + attn_output)
                            ffn_output = tf.keras.layers.Dense(units, activation='relu')(attn_output)
                            ffn_output = tf.keras.layers.Dense(x.shape[-1])(ffn_output)
                            x = tf.keras.layers.LayerNormalization()(attn_output + ffn_output)
                            x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
                        elif layer == 'Dense':
                            # Flatten if preceding layer was sequence-outputting
                            if len(x.shape) == 3:
                                x = tf.keras.layers.Flatten()(x)
                            x = tf.keras.layers.Dense(units=units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
                            x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
                        else:
                            raise MLToolkitError(f"Unsupported layer type: {layer}")
                
                # After all main layers, output is 2D
                if len(x.shape) == 3: 
                    x = tf.keras.layers.Flatten()(x) 

                outputs = tf.keras.layers.Dense(units=num_classes, activation=output_activation)(x)
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
            else:
                raise MLToolkitError("model_style must be 'Sequential' or 'Functional'")
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            return model
    
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to create flexible Keras model")

    @staticmethod
    def create_lstm_model(input_shape: Tuple[int, ...], units: int = 50, dropout_rate: float = 0.2, output_units: int = 1, output_activation: str = 'linear') -> tf.keras.Model:
        """
        Create a basic LSTM model for regression or classification.

        Args:
            input_shape: Shape of input data (e.g., (timesteps, features)).
            units: Number of units in LSTM layers.
            dropout_rate: Dropout rate.
            output_units: Number of output units (e.g., 1 for regression, num_classes for classification).
            output_activation: Activation function for the output layer ('linear' for regression, 'softmax' for classification).

        Returns:
            tf.keras.Model: Compiled LSTM model.

        Raises:
            MLToolkitError: If model creation fails.
        """
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(shape=input_shape),
                tf.keras.layers.LSTM(units=units, return_sequences=True),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.LSTM(units=units), 
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(units=output_units, activation=output_activation)
            ])
            
            if output_activation == 'softmax':
                loss_func = 'sparse_categorical_crossentropy' 
                metrics_list = ['accuracy']
            elif output_activation == 'sigmoid':
                loss_func = 'binary_crossentropy'
                metrics_list = ['accuracy']
            elif output_activation == 'linear':
                loss_func = 'mean_squared_error'
                metrics_list = ['mae']
            else:
                raise MLToolkitError(f"Unsupported output_activation for LSTM model: {output_activation}")

            model.compile(optimizer='adam', loss=loss_func, metrics=metrics_list)
            return model
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to create LSTM model")

    @staticmethod
    def create_gru_model(input_shape: Tuple[int, ...], units: int = 50, dropout_rate: float = 0.2, output_units: int = 1, output_activation: str = 'linear') -> tf.keras.Model:
        """
        Create a basic GRU model for regression or classification.

        Args:
            input_shape: Shape of input data (e.g., (timesteps, features)).
            units: Number of units in GRU layers.
            dropout_rate: Dropout rate.
            output_units: Number of output units (e.g., 1 for regression, num_classes for classification).
            output_activation: Activation function for the output layer ('linear' for regression, 'softmax' for classification).

        Returns:
            tf.keras.Model: Compiled GRU model.

        Raises:
            MLToolkitError: If model creation fails.
        """
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(shape=input_shape),
                tf.keras.layers.GRU(units=units, return_sequences=True),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.GRU(units=units), 
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(units=output_units, activation=output_activation)
            ])

            if output_activation == 'softmax':
                loss_func = 'sparse_categorical_crossentropy'
                metrics_list = ['accuracy']
            elif output_activation == 'sigmoid':
                loss_func = 'binary_crossentropy'
                metrics_list = ['accuracy']
            elif output_activation == 'linear':
                loss_func = 'mean_squared_error'
                metrics_list = ['mae']
            else:
                raise MLToolkitError(f"Unsupported output_activation for GRU model: {output_activation}")

            model.compile(optimizer='adam', loss=loss_func, metrics=metrics_list)
            return model
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to create GRU model")


    @staticmethod
    def _build_simple_regression_nn(input_shape: Tuple[int, ...], output_units: int) -> tf.keras.Model:
        """
        Builds a simple feed-forward neural network for regression.
        This is a private helper method for specific regression NN architecture.

        Args:
            input_shape: Shape of input data (e.g., (num_features,)).
            output_units: Number of output units (e.g., len(target)).

        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(output_units, activation='linear')
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        except Exception as e:
            ErrorHandler.handle_error(e, "Failed to build simple regression NN")

    @staticmethod
    def build_and_train_regression_model(X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.DataFrame, np.ndarray],
                                         features: List[str], target: List[str], model_type: str = 'nn',
                                         nn_epochs: int = 50, nn_batch_size: int = 32) -> Any:
        """
        Builds and trains a predictive model (Random Forest or Neural Network) for regression tasks.

        Parameters:
            X_train (Union[pd.DataFrame, np.ndarray]): Training features (scaled).
            y_train (Union[pd.DataFrame, np.ndarray]): Training target values (scaled).
            features (List[str]): List of feature column names.
            target (List[str]): List of target column names.
            model_type (str): Type of model to build ('nn' for neural network, 'rf' for random forest).
            nn_epochs (int): Number of epochs for Neural Network training.
            nn_batch_size (int): Batch size for Neural Network training.

        Returns:
            model: Trained predictive model (sklearn model or Keras model).

        Raises:
            MLToolkitError: If input validation fails or an unsupported model type is specified.
        """
        try:
            model = None
            if model_type == 'rf':
                print("Building Random Forest Regressor...")
                model = RandomForestRegressor(random_state=42, n_estimators=100)
                ModelBuilder.train_model(model, X_train, y_train)
            elif model_type == 'nn':
                print("Building Neural Network Regressor...")
                input_shape = (X_train.shape[1],) 
                output_units = y_train.shape[1] if len(y_train.shape) > 1 else 1
                
                # Using the specific regression NN builder
                model = ModelBuilder._build_simple_regression_nn(input_shape, output_units)
                
                # Using the generalized train_model for Keras model
                ModelBuilder.train_model(model, X_train, y_train, epochs=nn_epochs, batch_size=nn_batch_size, plot_epochs=False) # plot_epochs can be an arg

            else:
                raise MLToolkitError(f"Invalid model_type: '{model_type}'. Use 'nn' for neural network or 'rf' for random forest.")

            return model
        except Exception as e:
            ErrorHandler.handle_error(e, f"Failed to build and train {model_type} regression model")