# Bitcoin Price Movement Prediction using CNN-LSTM Hybrid Model

## 📈 Introduction to Time-Series Forecasting

Time series forecasting is the process of using historical data to predict future values based on trends, seasonality, and patterns in temporal data. Unlike traditional regression, time series forecasting considers the temporal structure where past observations influence future values. This project implements advanced deep learning techniques for financial market prediction.

**Key Techniques Explored:**
- **Statistical Models**: ARIMA, Exponential Smoothing (Holt-Winters), VAR
- **Machine Learning**: Random Forest, XGBoost, Support Vector Regression
- **Deep Learning**: LSTM, CNN-LSTM Hybrids, Transformer-based models

## 🔬 Methodology and Research Design

This project evaluates the predictive performance of a **CNN-LSTM Hybrid** deep learning model for Bitcoin price movement prediction. The methodology integrates:

- **Technical Indicators**: Relative Strength Index (RSI), Simple/Exponential Moving Averages, MACD
- **Market Dynamics**: Seasonality patterns and underlying trend analysis
- **Neural Architecture**: Hybrid model combining CNN layers for feature extraction and LSTM layers for temporal sequence modeling
- **Target Prediction**: Binary classification for price direction (0=price falls, 1=price rises)

The research design focuses on creating a robust trading signal generation system using historical price data and technical analysis features.

## 📊 Data Acquisition

**Data Source**: Coinbase API  
**Time Period**: January 2024 - January 2025 (1 year)  
**Frequency**: Hourly data  
**Features**: OHLCV (Open, High, Low, Close, Volume)  
**Final Dataset**: 527,135 samples with 78 engineered features

The dataset provides comprehensive market information with sufficient temporal resolution for intraday pattern recognition and short-term price movement prediction.

## ⚙️ Feature Engineering

### Technical Indicators
- **RSI (Relative Strength Index)**: Momentum oscillator measuring price change velocity
- **Moving Averages**: SMA and EMA for trend identification
- **MACD**: Moving Average Convergence Divergence for trend changes

### Temporal Features
- **Cyclical Encoding**: Hour and day-of-week features using sine/cosine transformations
- **STL Decomposition**: Seasonal-Trend decomposition using LOESS for pattern extraction
- **Rolling Statistics**: 3, 6, and 24-hour rolling means and standard deviations

### Advanced Feature Engineering
- **Autocorrelation Analysis**: PACF analysis identifying significant lags (1, 2, 3, 9, 13, 16 hours)
- **Lag Features**: Historical price features based on statistically significant correlations
- **Normalization**: StandardScaler for feature scaling and model stability

### Data Preprocessing
- **Sequence Generation**: Conversion to LSTM-compatible format (samples, sequence_length, features)
- **Train/Validation/Test Split**: Temporal splitting to prevent data leakage
- **Missing Value Handling**: Removal of NaN values from technical indicator calculations

## 🤖 Model Selection

### Architecture: CNN-LSTM Hybrid
```
Input Layer → CNN Layers → LSTM Layers → Dense Layers → Binary Output
```

**Model Components:**
- **CNN Layers**: Extract local patterns from technical indicators
  - Multiple convolutional layers with batch normalization
  - Filters: 32, 64 with kernel size 5
- **LSTM Layers**: Capture temporal dependencies and long-term patterns
  - 3 LSTM layers with 256, 128, 128 units respectively
  - Dropout rate: 42.4% for regularization
- **Dense Layers**: Final classification with sigmoid activation

### Hyperparameter Optimization
- **Framework**: Optuna for automated hyperparameter tuning
- **Search Strategy**: 20 trials with 15 epochs each
- **Optimized Parameters**: Learning rate (0.002), batch size (128), dropout rates, L2 regularization (0.0002)

### Training Configuration
- **Loss Function**: Binary crossentropy
- **Optimizer**: Adam with learning rate 0.002
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Validation Strategy**: Time-based split maintaining temporal order

## 📊 Results and Performance

### Model Performance Metrics
- **Training Accuracy**: 74%
- **Test Accuracy**: 69%
- **AUC Score**: 0.76
- **F1-Score**: 0.68-0.69 (balanced across classes)

### Classification Performance
| Metric | Class 0 (Price Falls) | Class 1 (Price Rises) |
|--------|----------------------|----------------------|
| Precision | 0.68 | 0.69 |
| Recall | 0.71 | 0.66 |
| F1-Score | 0.69 | 0.68 |

### Key Observations
- **Moderate Overfitting**: 5% performance drop from training to testing
- **Balanced Classification**: Equal performance across both price direction classes
- **Decision Threshold**: Optimized to 0.47 (vs. standard 0.5)
- **Error Distribution**: Balanced false positives (8,904) and false negatives (7,543)

## 🔍 Conclusion

The CNN-LSTM hybrid model demonstrates **moderate predictive power** for Bitcoin price movement prediction with an AUC of 0.76, indicating substantially better performance than random classification. The model successfully integrates technical indicators and temporal patterns to generate trading signals.

### Strengths
- Balanced performance across price direction classes
- Effective integration of technical and temporal features
- Robust architecture combining CNN pattern recognition with LSTM temporal modeling
- Systematic hyperparameter optimization approach

### Areas for Improvement
- **Overfitting Mitigation**: Enhanced regularization techniques (L1/L2, advanced dropout)
- **Architecture Refinement**: Attention mechanisms, transformer components
- **Feature Enhancement**: Additional technical indicators, market sentiment data
- **Data Augmentation**: Synthetic data generation for improved generalization

### Future Work
The model provides a solid foundation for algorithmic trading applications but would benefit from ensemble methods, alternative architectures (Transformer-based models), and integration of external market factors for enhanced predictive accuracy.
