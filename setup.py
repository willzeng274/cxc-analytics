from setuptools import setup, find_packages

setup(
    name="runQL",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'plotly',
        'seaborn',
        'matplotlib',
        'pandas',
        'numpy',
        'joblib',
        'scikit-learn',
        'statsmodels',
        'prophet',
        'cvxopt'
    ]
)