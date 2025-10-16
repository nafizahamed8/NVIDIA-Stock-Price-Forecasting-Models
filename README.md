---
license: mit
language:
- en
---
tags:
- time-series
- forecasting
- stock-prediction
- arima
- lstm
- nvidia
- financial
- machine-learning
library_name: sklearn
pipeline_tag: time-series-forecasting
---

# NVIDIA Stock Price Forecasting Models

## Model Description

This repository contains two trained time series forecasting models for predicting NVIDIA Corporation (NVDA) stock closing prices:

- **ARIMA (AutoRegressive Integrated Moving Average)**: A statistical model for time series forecasting
- **LSTM (Long Short-Term Memory)**: A deep learning recurrent neural network model

Both models were trained on historical NVIDIA stock data from January 2020 to October 2025 and implement rolling window evaluation for realistic performance assessment.

## Model Details

### Model Architecture

**ARIMA Model:**
- **Type**: Statistical time series model
- **Order**: (0,1,0) - Random Walk model
- **Training Method**: Rolling window evaluation with expanding training set
- **Parameters**: Automatically determined through stationarity testing and ACF/PACF analysis

**LSTM Model:**
- **Type**: Deep learning recurrent neural network
- **Architecture**: Single-layer LSTM with 16 hidden units
- **Regularization**: Dropout (0.3) to prevent overfitting
- **Sequence Length**: 60 days (lookback period)
- **Training**: Rolling window retraining for each prediction

### Training Data

- **Stock**: NVIDIA Corporation (NVDA)
- **Time Period**: January 1, 2020 to October 2, 2025
- **Total Data Points**: 1,355 daily closing prices
- **Data Split**: 80% training (1,084 points), 20% testing (271 points)
- **Data Source**: Yahoo Finance via yfinance API

### Preprocessing

1. **Data Cleaning**: Handled missing values and date formatting
2. **Stationarity Testing**: Augmented Dickey-Fuller test confirmed non-stationarity
3. **Differencing**: First-order differencing applied to achieve stationarity
4. **Normalization**: MinMax scaling (0-1) for LSTM training
5. **Sequence Creation**: 60-day windows for LSTM training

## Performance

### Evaluation Metrics

| Model | MAE | RMSE | MAPE | Directional Accuracy |
|-------|-----|------|------|---------------------|
| **ARIMA** | $2.99 | $4.05 | 19.11% | - |
| **LSTM** | $7.89 | $9.34 | 5.71% | 52.34% |

### Performance Analysis

**ARIMA Model Strengths:**
- ✅ **Lower Absolute Error**: Better at predicting exact price values (MAE: $2.99)
- ✅ **More Consistent**: Lower error variance (RMSE: $4.05)
- ✅ **Computationally Efficient**: Faster training and prediction
- ❌ **Higher Percentage Error**: Less accurate in relative terms (MAPE: 19.11%)

**LSTM Model Strengths:**
- ✅ **Better Relative Accuracy**: Superior percentage performance (MAPE: 5.71%)
- ✅ **Directional Insight**: Can predict price direction with 52.34% accuracy
- ✅ **Complex Pattern Capture**: Better for non-linear relationships
- ❌ **Higher Absolute Error**: Less precise in dollar terms (MAE: $7.89)

### Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| Exact Price Prediction | 🎯 **ARIMA** | Lower absolute error (MAE: $2.99) |
| Percentage Accuracy | 🎯 **LSTM** | Lower MAPE (5.71%) |
| Trend Direction | 🎯 **LSTM** | 52.34% directional accuracy |
| Computational Efficiency | 🎯 **ARIMA** | Faster training & inference |
| Complex Pattern Capture | 🎯 **LSTM** | Better for non-linear relationships |
| Risk Management | 🎯 **ARIMA** | More consistent error distribution |

## Limitations

### ARIMA Limitations
- **Assumes linear relationships in data**
- **Limited ability to capture complex market dynamics**
- **Requires stationary data**
- **May underperform during high volatility periods**

### LSTM Limitations
- **Higher computational requirements**
- **Requires more data for optimal performance**
- **Black box nature limits interpretability**
- **Sensitive to hyperparameter tuning**

### General Limitations
- **Models trained on historical data only**
- **Cannot account for unexpected market events**
- **Performance may vary in different market conditions**
- **Not financial advice - use at own risk**

## Ethical Considerations

- **Financial Risk**: Stock predictions involve significant financial risk
- **No Investment Advice**: Models are for educational/research purposes only
- **Market Impact**: Large-scale use could potentially influence markets
- **Data Bias**: Historical biases may be reflected in predictions


### Requirements

Require libraries are listed in required.txt


### Hardware Recommendations
- **ARIMA**: Any modern CPU
- **LSTM**: CPU or GPU with ≥4GB RAM
- **Storage**: ≥100MB free space



