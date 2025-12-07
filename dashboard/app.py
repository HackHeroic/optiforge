# -----------------------------
# 1. IMPORTS & PAGE CONFIG
# -----------------------------
import streamlit as st
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Option Pricing Dashboard", layout="wide")

# ------------------------------------------------------
# REMOVE STREAMLIT DEFAULT TOP WHITE NAVBAR
# ------------------------------------------------------
st.markdown("""
<style>
header[data-testid="stHeader"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------
# CUSTOM NAVBAR FUNCTION
# ------------------------------------------------------
def render_navbar(dark=False):
    bg = "#0f0f11" if dark else "#ffffff"
    text = "#ffffff" if dark else "#1f2937"

    st.markdown(f"""
    <style>
        .custom-nav {{
            background-color: {bg};
            width: 100%;
            height: 55px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding: 0 30px;
            border-bottom: 1px solid rgba(0,0,0,0.2);
            position: fixed;
            top: 0;
            left: 0;
            z-index: 9999;
        }}

        .nav-link {{
            font-size: 17px;
            font-weight: 600;
            color: {text};
            text-decoration: none;
        }}

        .nav-spacer {{
            margin-top: 65px;
        }}
    </style>

    <div class="custom-nav">
        <a class="nav-link" href="https://github.com" target="_blank">GitHub</a>
    </div>

    <div class="nav-spacer"></div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------
# 2. PREMIUM CSS (LIGHT + GRID)
# ------------------------------------------------------
st.markdown(
    """
<style>

    /* GRID BACKGROUND (LIGHT) */
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc;
        background-image:
            linear-gradient(to right, rgba(0,0,0,0.04) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(0,0,0,0.04) 1px, transparent 1px);
        background-size: 26px 26px;
    }

    /* SIDEBAR (LIGHT) */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb;
        color: #111827 !important;
    }

    /* TITLE + SUBTITLE */
    .title {
        font-size: 3rem !important;
        font-weight: 900 !important;
        text-align: center;
        margin-top: -30px;
        padding-bottom: 10px;
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 40px;
    }

    /* METRIC CARDS */
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transition: 0.3s ease-in-out;
        text-align: center;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0,0,0,0.13);
    }

    .metric-title {
        font-size: 1rem;
        color: #4F46E5;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #111827;
    }

    /* COMPARISON BOX */
    .comparison-box {
        background: linear-gradient(to right, #EEF2FF, #E0E7FF);
        border-left: 6px solid #4F46E5;
        padding: 18px 25px;
        border-radius: 15px;
        font-size: 1.1rem;
        margin-top: 15px;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.07);
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1E293B;
        margin-top: 30px;
        margin-bottom: 10px;
    }

    .footer {
        margin-top: 40px;
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
    }

</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# 3. MODEL LOADING & PREDICTION
# -----------------------------

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Model configuration
MODEL_CONFIG = {
    "MLP C/S (No GARCH)": {
        "path": "mlp_cs_without_garch.h5",
        "type": "mlp",
        "output": "log_cs",  # Models WITHOUT GARCH output log(C/S)
        "garch": False
    },
    "MLP C/S (With GARCH)": {
        "path": "mlp_cs_with_garch.h5",
        "type": "mlp",
        "output": "cs",  # Models WITH GARCH output C/S
        "garch": True
    },
    "MLP log(C/S) (No GARCH)": {
        "path": "mlp_log_cs_without_garch.h5",
        "type": "mlp",
        "output": "log_cs",  # Models WITHOUT GARCH output log(C/S)
        "garch": False
    },
    "MLP log(C/S) (With GARCH)": {
        "path": "mlp_log_cs_with_garch.h5",
        "type": "mlp",
        "output": "cs",  # Models WITH GARCH output C/S
        "garch": True
    },
    "LSTM C/S (No GARCH)": {
        "path": "lstm_cs_without_garch.h5",
        "type": "lstm",
        "output": "log_cs",  # Models WITHOUT GARCH output log(C/S)
        "garch": False
    },
    "LSTM C/S (With GARCH)": {
        "path": "lstm_cs_with_garch.h5",
        "type": "lstm",
        "output": "cs",  # Models WITH GARCH output C/S
        "garch": True
    },
    "LSTM log(C/S) (No GARCH)": {
        "path": "lstm_log_cs_without_garch.h5",
        "type": "lstm",
        "output": "log_cs",  # Models WITHOUT GARCH output log(C/S)
        "garch": False
    },
    "LSTM log(C/S) (With GARCH)": {
        "path": "lstm_log_cs_with_garch.h5",
        "type": "lstm",
        "output": "cs",  # Models WITH GARCH output C/S
        "garch": True
    }
}

# Cache models to avoid reloading
@st.cache_resource
def load_model_cached(model_path):
    """Load a model with caching"""
    full_path = os.path.join(MODELS_DIR, model_path)
    if not os.path.exists(full_path):
        st.warning(f"Model file not found: {full_path}")
        return None
    try:
        return load_model(full_path, compile=False)
    except Exception as e:
        st.error(f"Error loading model {model_path}: {str(e)}")
        return None


def prepare_features(S, K, T, r, sigma, use_garch=False):
    """
    Prepare input features for the models.
    Models expect: [IV, K/S, Maturity, r] or [IV, K/S, Maturity, r, cond_vol]
    Dashboard provides: S, K, T, r, sigma
    
    Note: Models were trained with StandardScaler, but scalers are not saved.
    Feature ranges should be similar to training data for best results.
    """
    # Map dashboard inputs to model features
    IV = sigma  # Implied volatility (maps to sigma from dashboard)
    K_S = K / S if S > 0 else 1.0  # Strike/Stock ratio
    Maturity = T  # Time to maturity (in years)
    r_feat = r  # Risk-free rate
    
    if use_garch:
        # For GARCH models, use sigma as proxy for conditional volatility
        # In practice, this would be calculated from historical data using GARCH(1,1)
        # Using a simple approximation: cond_vol ≈ sigma * adjustment factor
        cond_vol = sigma * 1.1  # Simple approximation
        return np.array([[IV, K_S, Maturity, r_feat, cond_vol]])
    else:
        return np.array([[IV, K_S, Maturity, r_feat]])


def predict_with_model(model, features, model_type, output_type):
    """
    Make prediction with the model
    
    Args:
        model: Loaded Keras model
        features: Input features array
        model_type: 'mlp' or 'lstm'
        output_type: 'cs' or 'log_cs'
    """
    if model is None:
        return None
    
    try:
        # LSTM models need 3D input: (samples, timesteps, features)
        if model_type == "lstm":
            features = features.reshape(1, 1, -1)
        
        # Make prediction
        prediction = model.predict(features, verbose=0)
        
        # Handle output transformation
        if output_type == "log_cs":
            # Model outputs log(C/S), need to convert back
            cs_ratio = exp(prediction[0][0])
        else:
            # Model outputs C/S directly
            cs_ratio = prediction[0][0]
        
        return max(cs_ratio, 0.0)  # Ensure non-negative
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


def get_model_cs_ratio(model_choice, S, K, T, r, sigma):
    """
    Get C/S ratio prediction from the selected model.
    Models with GARCH output C/S directly.
    Models without GARCH output log(C/S), which we convert to C/S.
    
    Returns: C/S ratio (float)
    """
    if model_choice not in MODEL_CONFIG:
        return None
    
    config = MODEL_CONFIG[model_choice]
    
    # Load model
    model = load_model_cached(config["path"])
    if model is None:
        # Fallback to Black-Scholes if model not available
        bs_price = black_scholes_call(S, K, T, r, sigma)
        return bs_price / S if S > 0 else 0.0
    
    # Prepare features
    features = prepare_features(S, K, T, r, sigma, use_garch=config["garch"])
    
    # Make prediction - this handles conversion from log(C/S) to C/S if needed
    cs_ratio = predict_with_model(
        model, 
        features, 
        config["type"], 
        config["output"]
    )
    
    if cs_ratio is None:
        # Fallback to Black-Scholes if prediction fails
        bs_price = black_scholes_call(S, K, T, r, sigma)
        return bs_price / S if S > 0 else 0.0
    
    return cs_ratio


def get_model_call_price(model_choice, S, K, T, r, sigma):
    """
    Get the actual call option price C from the model.
    This converts C/S ratio to C by multiplying by S.
    
    Returns: Call price C (float)
    """
    cs_ratio = get_model_cs_ratio(model_choice, S, K, T, r, sigma)
    if cs_ratio is None:
        return 0.0
    return S * cs_ratio


def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


# -----------------------------
# 4. RENDER NAVBAR (depends on dark mode later)
# -----------------------------


# -----------------------------
# 5. SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Input Parameters")

S = st.sidebar.slider("Spot Price (S)", 10.0, 500.0, 100.0)
K = st.sidebar.slider("Strike Price (K)", 10.0, 500.0, 120.0)
T = st.sidebar.slider("Time to Maturity (years)", 0.01, 2.0, 1.0)
r = st.sidebar.slider("Risk-Free Interest Rate (r)", 0.00, 0.20, 0.05)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.00, 0.20)

model_choice = st.sidebar.selectbox(
    "Select Model", 
    list(MODEL_CONFIG.keys()),
    index=0
)

dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

# NOW APPLY NAVBAR BASED ON DARK MODE
render_navbar(dark=dark_mode)

# -----------------------------
# 4. HEADER (RESTORED TITLE + SUBTITLE)
# -----------------------------
st.markdown(
    "<h1 class='title'>OptiForge Neural Pricing Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='subtitle'>Advanced Option Pricing using MLP, LSTM & Black-Scholes</p>",
    unsafe_allow_html=True,
)

# -----------------------------
# 5b. Extra Dark-mode CSS
# -----------------------------
if dark_mode:
    st.markdown(
        """
    <style>

    [data-testid="stAppViewContainer"] {
        background-color: #0f0f11 !important;
        background-image:
            linear-gradient(to right, rgba(255,255,255,0.06) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255,255,255,0.06) 1px, transparent 1px);
        background-size: 26px 26px;
    }

    [data-testid="stSidebar"] {
        background-color: #0c0c0d !important;
        border-right: 1px solid #1f1f1f !important;
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .css-17eq0hr,
    [data-testid="stSidebar"] .css-q8sbsg,
    [data-testid="stSidebar"] .css-1p4l8gs {
        color: #e5e7eb !important;
    }

    .metric-card,
    .comparison-box {
        background: rgba(25, 25, 25, 0.65) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.14);
    }

    .metric-title {
        color: #a5b4fc !important;
    }

    .metric-value {
        color: #ffffff !important;
    }

    .section-title,
    .subtitle,
    .footer,
    h2, h3, h4, h5, h6,
    .stMarkdown p,
    .stMarkdown strong {
        color: #ffffff !important;
    }

    </style>
    """,
        unsafe_allow_html=True,
    )


# -----------------------------
# 6. TOP METRICS
# -----------------------------
bs_price = black_scholes_call(S, K, T, r, sigma)
model_price = get_model_call_price(model_choice, S, K, T, r, sigma)

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class='metric-title'>Model Predicted Call Price (C) 💠</div>
            <div class='metric-value'>${model_price:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class='metric-title'>Black-Scholes Call Price (C) 📘</div>
            <div class='metric-value'>${bs_price:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


error = model_price - bs_price
percent_error = (error / bs_price) * 100 if bs_price > 0 else 0.0

st.markdown("<div class='section-title'>Comparison</div>", unsafe_allow_html=True)
st.markdown(
    f"""
<div class="comparison-box">
    <b>Model Call Price (C):</b> ${model_price:.4f} <br>
    <b>Black-Scholes Call Price (C):</b> ${bs_price:.4f} <br>
    <b>Absolute Error:</b> ${error:.4f} <br>
    <b>Percentage Error:</b> {percent_error:.3f}%
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# VISUALIZATION HELPERS
# -----------------------------
def get_price_vs_spot_data(model_choice, S, K, T, r, sigma):
    """Generate price vs spot price data for visualization.
    Returns call prices C (not C/S ratios).
    """
    S_vals = np.linspace(max(10, 0.5 * S), min(500, 1.5 * S), 60)
    bs_prices = np.array([black_scholes_call(s, K, T, r, sigma) for s in S_vals])
    
    model_prices = []
    for s in S_vals:
        # Get actual call price C (converts from C/S or log(C/S) internally)
        call_price = get_model_call_price(model_choice, s, K, T, r, sigma)
        model_prices.append(call_price)
    model_prices = np.array(model_prices)
    
    return S_vals, bs_prices, model_prices


def get_price_vs_vol_data(model_choice, S, K, T, r, sigma):
    """Generate price vs volatility data for visualization.
    Returns call prices C (not C/S ratios).
    """
    sigmas = np.linspace(max(0.01, sigma * 0.5), min(1.0, sigma * 1.5), 60)
    bs_prices = np.array([black_scholes_call(S, K, T, r, sig) for sig in sigmas])
    
    model_prices = []
    for sig in sigmas:
        # Get actual call price C (converts from C/S or log(C/S) internally)
        call_price = get_model_call_price(model_choice, S, K, T, r, sig)
        model_prices.append(call_price)
    model_prices = np.array(model_prices)
    
    return sigmas, bs_prices, model_prices


def get_heatmap_data(model_choice, S, K, T, r, sigma):
    """Generate heatmap data for S vs sigma.
    Returns call prices C (not C/S ratios).
    """
    S_vals = np.linspace(max(10, 0.7 * S), min(500, 1.3 * S), 10)
    sigmas = np.linspace(max(0.05, sigma * 0.5), min(0.9, sigma * 1.5), 10)

    bs_grid = np.zeros((len(sigmas), len(S_vals)))
    model_grid = np.zeros_like(bs_grid)

    for i, sig in enumerate(sigmas):
        for j, s in enumerate(S_vals):
            bs_grid[i, j] = black_scholes_call(s, K, T, r, sig)
            # Get actual call price C (converts from C/S or log(C/S) internally)
            model_grid[i, j] = get_model_call_price(model_choice, s, K, T, r, sig)

    return S_vals, sigmas, bs_grid, model_grid


# -----------------------------
# 8. VISUALIZATIONS
# -----------------------------
st.markdown("<div class='section-title'>Visualizations</div>", unsafe_allow_html=True)

# PRICE VS SPOT
st.markdown("#### 1. Price vs Spot Price (S)")
col_m1, col_bs1 = st.columns(2)

S_vals, bs_S, model_S = get_price_vs_spot_data(model_choice, S, K, T, r, sigma)

with col_m1:
    st.markdown("**Selected Model**")
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    ax.plot(S_vals, model_S, linewidth=2)
    ax.set_xlabel("Spot Price (S)")
    ax.set_ylabel("Call Price")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with col_bs1:
    st.markdown("**Black-Scholes**")
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    ax.plot(S_vals, bs_S, linewidth=2)
    ax.set_xlabel("Spot Price (S)")
    ax.set_ylabel("Call Price")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


# PRICE VS VOL
st.markdown("#### 2. Price vs Volatility (σ)")
col_m2, col_bs2 = st.columns(2)

sigmas, bs_sig, model_sig = get_price_vs_vol_data(model_choice, S, K, T, r, sigma)


with col_m2:
    st.markdown("**Selected Model**")
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    ax.plot(sigmas, model_sig, linewidth=2)
    ax.set_xlabel("Volatility (σ)")
    ax.set_ylabel("Call Price")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with col_bs2:
    st.markdown("**Black-Scholes**")
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    ax.plot(sigmas, bs_sig, linewidth=2)
    ax.set_xlabel("Volatility (σ)")
    ax.set_ylabel("Call Price")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


# HEATMAP
st.markdown("#### 3. Call Price Heatmap C(S, σ)")
col_m3, col_bs3 = st.columns(2)

S_grid, sig_grid, bs_grid, model_grid = get_heatmap_data(model_choice, S, K, T, r, sigma)


def draw_heatmap(grid, title):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(S_grid)))
    ax.set_yticks(range(len(sig_grid)))
    ax.set_xticklabels([f"{v:.0f}" for v in S_grid], fontsize=7)
    ax.set_yticklabels([f"{v:.2f}" for v in sig_grid], fontsize=7)

    # annotate
    mean_val = np.mean(grid)
    for i in range(len(sig_grid)):
        for j in range(len(S_grid)):
            color = "white" if grid[i, j] > mean_val else "black"
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


with col_m3:
    st.markdown("**Selected Model**")
    st.pyplot(draw_heatmap(model_grid, f"{model_choice} Heatmap"))

with col_bs3:
    st.markdown("**Black-Scholes**")
    st.pyplot(draw_heatmap(bs_grid, "Black-Scholes Heatmap"))


# FOOTER
st.markdown(
    "<div class='footer'>Built with ❤️ by OptiForge AI Team</div>",
    unsafe_allow_html=True,
)
