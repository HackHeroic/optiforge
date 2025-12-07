# OptiForge: Hybrid ML-Enhanced Black-Scholes Option Pricing  
**C Murali Madhav (230115) • Ravi Yadav (230131) • Abhijeet (230036)**  

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://www.tensorflow.org/)

---

### Problem Statement  
The Black-Scholes model assumes constant volatility and log-normal returns — assumptions that fail in real markets, especially during volatility spikes and for extreme moneyness levels.  

**OptiForge** aims to build a deep learning-based European call option pricer using MLP and LSTM architectures trained on both synthetic Black-Scholes data and real AAPL options data. The project includes comprehensive evaluation with GARCH volatility features and detailed visualizations comparing neural network predictions vs. Black-Scholes.

---

## 🎯 Project Overview

This project implements and evaluates multiple deep learning architectures for European call option pricing:

1. **Phase 1**: Synthetic Black-Scholes data validation
2. **Phase 2**: Real AAPL options data (2021-2023) with MLP and LSTM models
3. **Feature Engineering**: Standard features vs. GARCH-enhanced volatility features
4. **Target Variables**: Both `C/S` (normalized price) and `log(C/S)` (log-transformed) targets

---

## 📊 Current Progress & Achieved Results

### Phase 1: Synthetic Black-Scholes Data (Complete ✅)

| Milestone                        | Status       | Result                                                                                  |
|----------------------------------|--------------|------------------------------------------------------------------------------------------|
| Black-Scholes synthetic dataset | Done         | 420,000 samples (S₀=100, K∈[50-150], T∈[0.1-2.0], σ∈[10-200%], r∈[0-10%])                |
| MLP hyperparameter search       | Done         | Tested ReLU/ELU/LeakyReLU/Swish → Best: 8-layer with mixed LeakyReLU+ELU                |
| Final MLP training               | Done         | **MSE = 0.02648** → **Mean relative error ≈ –0.03%** (mean fraction = –0.0003)          |
| R² score                         | Done         | **R² ≈ 0.99998+**, MAE < 0.01 on test set                                               |
| Signature "Mountain Plot"        | Done         | Perfect smooth ridge pattern (matches Hutchinson 1994, Asadzadeh 2024) — proof of correct learning of the pricing surface |

**Visual proof that the neural network has perfectly learned the Black-Scholes function:**

*Sorted back to original generation order → reveals the famous smooth "mountain ridge" seen only in top-tier papers*

### Phase 2: Real AAPL Options Data (Complete ✅)

| Milestone                        | Status       | Result                                                                                  |
|----------------------------------|--------------|------------------------------------------------------------------------------------------|
| Real AAPL data processing        | Done         | Processed AAPL options data (Nov 2021 - Mar 2023), ~32,000+ samples                     |
| GARCH volatility estimation      | Done         | Conditional volatility calculated using GARCH(1,1) model (with IV fallback)            |
| MLP models (without GARCH)       | Done         | Trained on C/S and log(C/S) targets                                                    |
| MLP models (with GARCH)          | Done         | Trained with GARCH conditional volatility feature                                       |
| LSTM models (without GARCH)       | Done         | Trained on C/S and log(C/S) targets                                                    |
| LSTM models (with GARCH)          | Done         | Trained with GARCH conditional volatility feature                                       |
| Model evaluation & visualization  | Done         | Comprehensive metrics (RMSE, MAE, MSE, R²) and 5 visualization types per model          |

---

## 📁 Project Structure 

```
optiforge/
├── OptionPricingMLTests.ipynb                    # Hyperparameter tuning & activation experiments
├── OptionPricingSimulationUsingANN(MLP).ipynb   # Phase 1: MLP on synthetic Black-Scholes data
├── OptionsPricingRealData.ipynb                  # Phase 2: Real AAPL data processing & EDA
├── OptionsPricingRealData2.ipynb                # Phase 2: Initial model implementations
├── OptionsPricingRealData3.ipynb                # Phase 2: Final organized model training (8 models)
├── models/                                       # Trained model files (.h5 format)
│   ├── mlp_cs_without_garch.h5
│   ├── mlp_cs_with_garch.h5
│   ├── mlp_log_cs_without_garch.h5
│   ├── mlp_log_cs_with_garch.h5
│   ├── lstm_cs_without_garch.h5
│   ├── lstm_cs_with_garch.h5
│   ├── lstm_log_cs_without_garch.h5
│   └── lstm_log_cs_with_garch.h5
├── aapl_2021_2023.csv                            # Raw AAPL options data (download from [Google Drive](https://drive.google.com/file/d/1KxyOgvhGu9q0VK9JWwhaqvdRK0moDZP0/view?usp=sharing))
├── combine.csv                                   # Processed combined dataset
├── scaler_X.pkl                                  # Feature scaler (saved)
├── scaler_y.pkl                                  # Target scaler (saved)
├── install.sh                                    # Environment setup script
├── y_test_y_pred.csv                             # Sample predictions
├── dashboard/                                    # Interactive Streamlit dashboard
│   └── app.py                                   # Main dashboard application
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/HackHeroic/optiforge.git
cd optiforge

# Install all dependencies (creates a Conda environment automatically)
./install.sh

# Launch Jupyter Notebook
jupyter notebook

# Select kernel: Choose 'optiforge' kernel and run the notebooks
```

**Recommended execution order:**
1. `OptionPricingMLTests.ipynb` - Understand hyperparameter experiments
2. `OptionPricingSimulationUsingANN(MLP).ipynb` - Phase 1: Synthetic data
3. `OptionsPricingRealData.ipynb` - Phase 2: Data processing & EDA
4. `OptionsPricingRealData3.ipynb` - Phase 2: Model training (all 8 models)

### Running the Interactive Dashboard

```bash
# Navigate to dashboard directory
cd dashboard

# Run Streamlit app
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501` with:
- **Interactive parameter controls** for Spot Price, Strike Price, Time to Maturity, Risk-Free Rate, and Volatility
- **Model selection** from 8 trained models (MLP/LSTM with/without GARCH, C/S and log(C/S) targets)
- **Real-time predictions** comparing neural network models vs. Black-Scholes
- **Interactive visualizations**:
  - Call Price Heatmap C(S, σ) comparing model predictions vs. Black-Scholes
  - Price vs Spot Price (S) line charts
  - Price vs Volatility (σ) line charts
- **Dark/Light mode** toggle for comfortable viewing
- **Professional model name display** with human-readable formatting

---

## 📈 Phase 1: Model Performance on Synthetic Black-Scholes Data

We trained a deep Multi-Layer Perceptron (8 hidden layers, LeakyReLU/ELU activations, Adam optimizer) on 420,000 synthetically generated European call options (S₀ = 100, varying r ∈ [0–10%], K ∈ [50–150], T ∈ [0.1–2.0], σ ∈ [10–200%], q = 0).

**Key Achievements**
- Extremely tight fit: Mean relative error ≈ -0.03% (mean fraction = -0.0003)
- R² ≈ 0.99998+, MAE < 0.01 on test set
- Model learns the exact Black-Scholes pricing function almost perfectly across the entire parameter space

**Signature "Mountain Plot" – Relative Pricing Error**  
By sorting predictions back into the original nested-loop generation order (r → K → T → σ), the relative error plot reveals the characteristic smooth "mountain ridge" pattern that is a hallmark of state-of-the-art neural network option pricers on synthetic BS data (also seen in Hutchinson 1994, Kiana Asadzadeh's 2024 dissertation, etc.). This confirms the model has correctly captured the non-linear pricing surface rather than memorizing noise.

---

## 📊 Phase 2: Model Performance on Real AAPL Options Data

### Dataset Overview
- **Source**: AAPL (Apple Inc.) European call options
- **Time Period**: November 2021 - March 2023 (~17 months)
- **Total Samples**: ~32,000+ after cleaning and filtering
- **Features**: IV (Implied Volatility), K/S (Strike/Stock ratio), Maturity, r (risk-free rate), cond_vol (GARCH conditional volatility)
- **Targets**: C/S (normalized option price) and log(C/S) (log-transformed)
- **Data Download**: The original AAPL options CSV file can be downloaded from [Google Drive](https://drive.google.com/file/d/1KxyOgvhGu9q0VK9JWwhaqvdRK0moDZP0/view?usp=sharing)

### Model Architectures

#### MLP Architecture
- **Layers**: 3 hidden layers (30 → 60 → 90 neurons)
- **Activation**: LeakyReLU
- **Optimizer**: Adam (LegacyAdam for M1/M2 Mac compatibility)
- **Hyperparameters**: epochs=150, batch_size=40
- **Loss**: Mean Squared Error

#### LSTM Architecture
- **Layers**: 2 LSTM layers (100 units each) + Dense layers
- **Activation**: ELU
- **Dropout**: 0.2
- **Optimizer**: Adam (LegacyAdam for M1/M2 Mac compatibility)
- **Hyperparameters**: epochs=150, batch_size=16
- **Loss**: Mean Squared Error

### Performance Results

#### Models WITHOUT GARCH Features

| Model | Target | RMSE | MAE | MSE | R² | Training Time |
|-------|--------|------|-----|-----|----|--------------| 
| MLP | C/S | 0.0363 | 0.0266 | 0.00132 | **0.9468** | ~173s |
| MLP | log(C/S) | 0.9238 | 0.6239 | 0.8534 | **0.7157** | ~174s |
| LSTM | C/S | 0.0359 | 0.0259 | 0.00129 | **0.9480** | ~1190s |
| LSTM | log(C/S) | 0.9469 | 0.7120 | 0.8966 | **0.7013** | ~1195s |

#### Models WITH GARCH Features

| Model | Target | RMSE | MAE | MSE | R² | Training Time |
|-------|--------|------|-----|-----|----|--------------|
| MLP | C/S | 0.0367 | 0.0278 | 0.00135 | **0.9457** | ~190s |
| MLP | log(C/S) | 0.9239 | 0.6272 | 0.8537 | **0.7156** | ~187s |
| LSTM | C/S | 0.0360 | 0.0258 | 0.00130 | **0.9477** | ~1188s |
| LSTM | log(C/S) | 0.9463 | 0.7151 | 0.8954 | **0.7017** | ~1218s |

### Key Findings

1. **Best Overall Performance**: LSTM on C/S target achieves **R² = 0.9480** (without GARCH) and **R² = 0.9477** (with GARCH)
2. **C/S vs log(C/S)**: Models trained on C/S consistently outperform log(C/S) targets (R² ~0.95 vs ~0.70)
3. **GARCH Impact**: GARCH features show minimal impact on performance (slight degradation in most cases), suggesting that IV already captures much of the volatility information
4. **LSTM vs MLP**: LSTM models show slightly better performance on C/S targets, but require ~7× longer training time
5. **Model Robustness**: All models show consistent performance across different feature sets, indicating stable learning

### Visualizations Generated

For each of the 8 models, we generate 5 comprehensive visualizations:

1. **Loss Function Plot**: Training and validation loss curves over epochs
2. **Actual vs Predicted Scatter**: Regression plot with R² score
3. **Fraction Error by Index**: Error distribution across test samples
4. **Fraction Error vs Actual**: Error patterns by actual target value
5. **Fraction Error vs Days to Maturity**: Error patterns by time to expiration

These visualizations reveal:
- Models perform best for moderate C/S values (0.1-0.6)
- Higher errors occur for very low C/S values (deep out-of-the-money options)
- Consistent error patterns across different maturities
- Some overprediction tendency, especially for log(C/S) targets

---

## 🔧 Hyperparameter Tuning & Activation Experiments (OptionPricingMLTests.ipynb)

This notebook contains our full experimental log for finding the optimal MLP architecture and activation functions on raw Black-Scholes prices (no division by S, no log transform yet).

**Key Experiments & Findings**
- Tested ReLU, ELU, LeakyReLU, SELU, Swish, and sigmoid-family variants
- Architecture search: 4–10 hidden layers, 64–512 units per layer, various dropout rates
- Best configuration found: 8 layers with mixed LeakyReLU/ELU activations, Adam optimizer (lr ≈ 0.001 → 0.0001 decay)
- Final training MSE = **0.02648** (≈ RMSE 0.163, MAE ≈ 0.11 on raw prices in [0–35] range)
- Loss curve shows smooth, rapid convergence without overfitting

These runs confirmed that deep MLPs with advanced activations (ELU/LeakyReLU) dramatically outperform shallow networks and sigmoid/tanh, achieving >30× lower error than early baselines.  
This directly informed the final ultra-high-precision model in the main notebook (mean relative error ≈ -0.03%, R² ≈ 0.99998) that produces the perfect "mountain plot".

The GBM simulation code and detailed mathematical derivation are also included here as the foundation for LSTM sequence models on real-market paths.

---

## 🎓 Technical Details

### Data Preprocessing
- **Feature Scaling**: StandardScaler applied to all features
- **Target Scaling**: StandardScaler for C/S, no scaling for log(C/S)
- **Train/Test Split**: 75/25 split with random state for reproducibility
- **Missing Data**: Dropped rows with missing IV, prices, or maturity data
- **Outlier Filtering**: Removed options with price < $0.10, maturity > 2.4 years

### Feature Engineering
- **C/S**: Normalized option price (Call Price / Stock Price)
- **log(C/S)**: Log-transformed normalized price
- **K/S**: Strike-to-Stock ratio (moneyness measure)
- **Maturity**: Time to expiration in years
- **IV**: Implied Volatility (from market data)
- **cond_vol**: GARCH(1,1) conditional volatility (with IV fallback if GARCH unavailable)

### Model Training Details
- **Early Stopping**: Not used (fixed epochs for consistency)
- **Validation Split**: 10% of training data used for validation
- **Random Seed**: Fixed for reproducibility
- **Hardware**: Compatible with M1/M2 Macs (uses LegacyAdam optimizer)

---

## 🎨 Interactive Dashboard

OptiForge includes a comprehensive Streamlit-based interactive dashboard (`dashboard/app.py`) that provides real-time option pricing comparisons between neural network models and the Black-Scholes formula.

### Features

- **Interactive Parameter Controls**: Adjust Spot Price (S), Strike Price (K), Time to Maturity (T), Risk-Free Interest Rate (r), and Volatility (σ) using intuitive sliders
- **Model Selection**: Choose from 8 pre-trained models:
  - MLP Neural Network (Call-to-Stock Ratio) with/without GARCH Volatility
  - MLP Neural Network (Log Call-to-Stock Ratio) with/without GARCH Volatility
  - LSTM Neural Network (Call-to-Stock Ratio) with/without GARCH Volatility
  - LSTM Neural Network (Log Call-to-Stock Ratio) with/without GARCH Volatility
- **Real-Time Predictions**: Instant comparison between selected model predictions and Black-Scholes prices
- **Comprehensive Visualizations**:
  - **Call Price Heatmap C(S, σ)**: 2D heatmaps showing how option prices vary with Spot Price and Volatility
  - **Price vs Spot Price (S)**: Line charts comparing model predictions and Black-Scholes across different spot prices
  - **Price vs Volatility (σ)**: Line charts showing sensitivity to volatility changes
- **Dark/Light Mode**: Toggle between dark and light themes for comfortable viewing
- **Error Metrics**: Display absolute error and percentage error between model predictions and Black-Scholes

### Dashboard Architecture

The dashboard uses:
- **Streamlit** for the web interface
- **TensorFlow/Keras** for loading and running pre-trained models
- **Matplotlib** for generating visualizations
- **NumPy** and **SciPy** for Black-Scholes calculations and data processing

All models are loaded from the `models/` directory and predictions are made in real-time based on user inputs.

---

## 📝 Future Work

- [ ] Real-time option pricing API
- [ ] Ensemble methods combining MLP and LSTM predictions
- [ ] Advanced volatility models (EGARCH, GJR-GARCH)
- [ ] Multi-asset option pricing (beyond single stocks)

---

## 📚 References

- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Hutchinson, J. M., Lo, A. W., & Poggio, T. (1994). A nonparametric approach to pricing and hedging derivative securities via learning networks. *The Journal of Finance*, 49(3), 851-889.
- Asadzadeh, K. (2024). *Machine Learning Approaches to Option Pricing*. Dissertation.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

---

## 👥 Contributors

- **C Murali Madhav** (230115) - Model development, real data processing
- **Ravi Yadav** (230131) - Hyperparameter tuning, visualization
- **Abhijeet** (230036) - Data pipeline, documentation

---

## 📄 License

This project is open source and available under the MIT License.

---

**Last Updated**: December 2025
