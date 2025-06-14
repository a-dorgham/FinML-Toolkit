from setuptools import setup, find_packages

setup(
    name='ml_toolkit',
    version='0.1.0', 
    packages=find_packages(), 
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'tensorflow', 
        'scikit-learn',
        'imbalanced-learn', 
        'catboost',
        'xgboost',
        'lightgbm',
        'seaborn',        
        'matplotlib',
        'plotly'
    ],
    python_requires='>=3.8', 
)