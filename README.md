# FinML-Toolkit

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange?logo=tensorflow)
![Plotly](https://img.shields.io/badge/Visualization-Plotly-darkorange?logo=plotly)
![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-ff69b4)

---

## 📊 FinML-Toolkit

**FinML-Toolkit** is a comprehensive Python library designed to streamline the full workflow of applying Machine Learning (ML) to financial time series data. It provides a modular and extensible framework for:

- **Data handling**
- **Feature engineering**
- **Model training & evaluation**
- **Signal generation**
- **Interactive visualization**

This empowers researchers and practitioners to build robust predictive models for financial markets.

---

## 🔧 Features

- **Robust Data Handling**: Load and filter financial time series from `.pkl` files using date ranges.
- **Advanced Feature Engineering**: Add technical indicators (e.g., RSI, MACD), volume analysis, and more.
- **Flexible ML Architectures**:
  - LSTM, GRU, Conv1D, Attention, Transformer (via TensorFlow/Keras)
  - RandomForest, XGBoost, CatBoost, LightGBM (via scikit-learn & boosting libraries)
- **Imbalanced Data Handling**: SMOTE, ADASYN, NearMiss, TomekLinks, SMOTETomek, SMOTEENN, etc.
- **Class Weight Optimization**: Optimize with `scipy.optimize.minimize` (Nelder-Mead, SLSQP, etc.)
- **Model Evaluation**: Accuracy, precision, recall, F1-score, MSE, RMSE, R², confusion matrix.
- **Interactive Plotting**: Plotly-based heatmaps, forecasting charts, actual vs. predicted lines/candlesticks, and trading signal visualizations.
- **Modular Design**: Clean structure with dedicated modules for each pipeline stage.

---

## 📁 Project Structure

```
FinML-Toolkit/
├── setup.py
├── examples/
│   ├── Basic_ML_Model_Training.ipynb
│   ├── ...
├── fin_data/
│   ├── EUR_USD_H4.pkl
│   └── ...
└── ml_toolkit/
    ├── __init__.py
    ├── data_handler.py
    ├── error_handler.py
    ├── feature_engineer.py
    ├── model_builder.py
    ├── model_evaluator.py
    ├── model_executor.py
    ├── model_forecaster.py
    └── visualizer.py
```

---

## 🚀 Getting Started

### 🔩 Prerequisites

- **Python 3.8+**
- **Recommended**: Use a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
```

### ▶️ Installation

From the project root (where `setup.py` is):

```bash
pip install -e .
```

---

## 📚 Usage Overview

### Load & Prepare Data

```python
from ml_toolkit.data_handler import DataHandler
df = DataHandler.load_data('GBP_USD_H4.pkl', start_date="2020-01-01", end_date="2024-07-29 21:00")
```

### Feature Engineering

```python
from ml_toolkit.feature_engineer import FeatureEngineer
df, time, X, y = FeatureEngineer.add_features(df=df, features=['RSI', 'MACD', 'Volume'])
```

### Build & Train Model

```python
from ml_toolkit.model_builder import ModelBuilder
input_shape = (X.shape[1], X.shape[2]) if X.ndim == 3 else (X.shape[1],)
model = ModelBuilder.create_lstm_model(input_shape=input_shape, output_units=3, output_activation='softmax')
```

### Execute & Evaluate

```python
from ml_toolkit.model_executor import ModelExecutor
model, acc, prec, rec, f1, conf, mse, rmse = ModelExecutor.execute_model(
    file_path='GBP_USD_H4.pkl',
    start_date="2020-01-01",
    end_date="2024-07-29 21:00",
    features=['RSI', 'MACD', 'Volume'],
    model_type='LSTM',
    epochs=10
)
```

### Visualization

```python
from ml_toolkit.visualizer import Visualizer
# visualizer = Visualizer()
# visualizer.plot_actual_vs_predicted_lines(y_test, y_pred)
```

---

## 📈 Example Notebooks

- `Basic_ML_Model_Training.ipynb`
- `Basic_Regression_Model_Training.ipynb`
- `Class_Weight_Optimization.ipynb`
- `Feature_Selection_Optimization.ipynb`
- `Imbalanced_Dataset_Evaluation.ipynb`
- `Correlation_Heatmap_Visualization.ipynb`
- `Peak_Detection_Visualization.ipynb`

---

## 📊 Model Output

Sample output from the notebook:
- Correlation maps visualization
![image](https://github.com/user-attachments/assets/d9911ef7-f1e9-491c-9a59-f8277fc8077b)

- Peaks detection
<img width="1014" alt="image" src="https://github.com/user-attachments/assets/76be0ad1-5909-491b-bad1-ccf9543d1612" />


- Actual vs predicted signals
  <img width="1063" alt="image" src="https://github.com/user-attachments/assets/0243a962-fa84-4146-a5ef-9f21c6ea4193" />

- Class classification
  <img width="1132" alt="image" src="https://github.com/user-attachments/assets/a5aa9858-df13-431b-855e-283af5cb031d" />

- Confusion matrix
<img width="1132" alt="image" src="https://github.com/user-attachments/assets/6526c7b8-913c-4848-bcbb-6b55ad2f4ccb" />

- Debugging report
<img width="1131" alt="image" src="https://github.com/user-attachments/assets/76d91977-adfd-45ce-a492-255830224eae" />


- 
---

## 🛠 Developer Notes

- Modular, extensible codebase.
- Add your own model logic in `model_builder.py`.
- Financial data expected in `.pkl` format in `fin_data/`.

---

## 🛑 Known Limitations

- Custom indicators must be added manually.
- Optimization methods may need tuning.

---

## 🌐 Roadmap

- Live data feed integration (e.g., Oanda, Binance)
- Backtesting module
- SHAP/LIME for explainability
- More advanced forecasting models
- Model deployment utilities

---

## 🤝 Contributing

We welcome contributions!

```bash
git clone https://github.com/a-dorgham/FinML-Toolkit.git
cd FinML-Toolkit
# Create a branch and submit a PR
```

---

## 📜 License

[MIT License](LICENSE)

---

## 📬 Contact

- Email: [a.k.y.dorgham@gmail.com](mailto:a.k.y.dorgham@gmail.com)
- GitHub Issues: [FinML-Toolkit Issues](https://github.com/a-dorgham/FinML-Toolkit/issues)
- Connect: [LinkedIn](https://www.linkedin.com/in/abdeldorgham) | [GoogleScholar](https://scholar.google.com/citations?user=EOwjslcAAAAJ&hl=en)  | [ResearchGate](https://www.researchgate.net/profile/Abdel-Dorgham-2) | [ORCiD](https://orcid.org/0000-0001-9119-5111)
