import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Food Inflation Tracker", layout="wide")

# --- 1. Universal Feature Generator ---
def create_model_features(df):
    """
    Ensures the dataframe has exactly the 44 features the model expects.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # --- A. Basic Date Features ---
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # --- B. Cyclic Features ---
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # --- C. Lag Features ---
    for lag in [1, 2, 3, 7, 14, 30]:
        val = df['price_mean'].shift(lag)
        df[f'lag_{lag}'] = val
        if lag in [7, 14, 30]:
            df[f'price_mean_lag_{lag}d'] = val

    # --- D. Rolling Windows ---
    for window in [7, 14, 30]:
        roll = df['price_mean'].shift(1).rolling(window)
        mean_val = roll.mean()
        std_val = roll.std()
        
        df[f'roll_mean_{window}'] = mean_val
        df[f'roll_std_{window}'] = std_val
        df[f'price_mean_ma_{window}d'] = mean_val
        df[f'price_mean_std_{window}d'] = std_val

    # --- E. Price Change Features ---
    df['price_change_abs'] = df['price_mean'].diff()
    df['price_change_pct'] = df['price_mean'].pct_change()
    
    # --- F. Fallback for "Raw" Columns ---
    # We add missing columns as standard types BEFORE converting to category
    required_raw_cols = [
        'price_median', 'price_std', 'price_min', 'price_max', 'transaction_count',
        'sentiment_mean', 'sentiment_std', 'negative_count', 'search_interest', 'keyword'
    ]
    
    for col in required_raw_cols:
        if col not in df.columns:
            if col == 'keyword':
                df[col] = 'unknown' 
            else:
                df[col] = 0.0
    
    # --- G. Final Type Conversion ---
    # Only NOW do we convert to categorical
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
        
    return df

# --- 2. Load Assets ---
@st.cache_resource
def load_assets():
    base_path = 'C:/Users/lenovo/Documents/food_inflation_ai/'
    
    # Load data
    df_raw = pd.read_csv(base_path + 'data/processed_prices.csv')
    
    # Generate ALL features
    df = create_model_features(df_raw)
    
    # Load Model
    model = joblib.load(base_path + 'outputs/final_lgbm_model.joblib')
    comparison = pd.read_csv(base_path + 'notebooks/model_comparison_results.csv')
    
    return df, model, comparison

def generate_7day_forecast(df, model):
    # 1. Setup
    # We start with the very last row of real data
    current_data = df.iloc[-1:].copy()
    last_real_date = df['date'].iloc[-1]
    last_real_price = df['price_mean'].iloc[-1]
    
    forecast_dates = []
    forecast_prices = []
    
    
    forecast_dates.append(last_real_date)
    forecast_prices.append(last_real_price)
    

    for i in range(1, 8):

        model_features = model.feature_name_
        X_pred = current_data[model_features].copy()
        
        # Smart Fill for NaNs (Numeric only)
        for col in X_pred.columns:
            if pd.api.types.is_numeric_dtype(X_pred[col]):
                X_pred[col] = X_pred[col].fillna(0)
        
        # B. Predict the Next Day
        pred_price = model.predict(X_pred)[0]
        
        # C. Store the Prediction
        next_date = last_real_date + pd.Timedelta(days=i)
        forecast_dates.append(next_date)
        forecast_prices.append(pred_price)
        
        # D. UPDATE THE FEATURES ("The Recursive Step")
        # This makes the AI "feel" time passing and prices changing
        
        # Update Lags: Today's prediction becomes tomorrow's "Lag_1"
        current_data['lag_1'] = pred_price 
        # (We could update all lags, but lag_1 is 80% of the signal)
        
        # Update Date Features
        current_data['date'] = next_date
        current_data['day_of_week'] = next_date.dayofweek
        current_data['month'] = next_date.month
        
        # Update Cyclic Features (Crucial for patterns like "Weekend Spikes")
        current_data['dow_sin'] = np.sin(2 * np.pi * current_data['day_of_week'] / 7)
        current_data['dow_cos'] = np.cos(2 * np.pi * current_data['day_of_week'] / 7)
        
    return pd.DataFrame({'date': forecast_dates, 'price_forecast': forecast_prices})

# --- 3. Main Dashboard UI ---
try:
    df, model, comparison = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

st.title("🛒 Food Price & Inflation Analysis")
st.markdown(f"**System Status:** Active | **Last Data Point:** {df['date'].max().strftime('%Y-%m-%d')}")

# Sidebar
st.sidebar.header("Settings")
budget = st.sidebar.number_input("Monthly Budget ($)", min_value=10, value=500)

recent_data = df.sort_values('date').tail(30)
monthly_change = ((recent_data['price_mean'].iloc[-1] - recent_data['price_mean'].iloc[0]) / recent_data['price_mean'].iloc[0]) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg Price", f"${recent_data['price_mean'].iloc[-1]:.2f}", f"{monthly_change:.2f}%")
with col2:
    power = budget / (1 + (monthly_change/100))
    st.metric("Budget Power", f"${power:.2f}", f"{power-budget:.2f} impact")
with col3:
    best_model = comparison.loc[comparison['MAPE (%)'].idxmin(), 'Model']
    st.metric("Top Model", best_model)
with col4:
    st.metric("Market Status", "High Volatility" if abs(monthly_change) > 5 else "Stable")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Price Trends", "🔮 AI Forecast", "💡 Advice"])

with tab1:
    df['SMA7'] = df['price_mean'].rolling(7).mean()
    fig = px.line(df, x='date', y=['price_mean', 'SMA7'], 
                  labels={'value': 'Price ($)', 'date': 'Date'},
                  title="Historical Price Movements",
                  color_discrete_map={'price_mean': '#CBD5E0', 'SMA7': '#E53E3E'})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    try:
        forecast_df = generate_7day_forecast(df, model)
        
        fig_f = go.Figure()
        hist_sub = df.tail(20)
        
        fig_f.add_trace(go.Scatter(x=hist_sub['date'], y=hist_sub['price_mean'], 
                                   name='Actual', line=dict(color='black')))
        
        fig_f.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['price_forecast'], 
                                   name='7-Day Forecast', line=dict(color='#3182ce', dash='dot', width=4)))
        
        fig_f.update_layout(title="7-Day Predicted Price Path", hovermode="x unified")
        st.plotly_chart(fig_f, use_container_width=True)
        
        f_change = ((forecast_df['price_forecast'].iloc[-1] - df['price_mean'].iloc[-1]) / df['price_mean'].iloc[-1]) * 100
        if f_change > 1:
            st.warning(f"Strategy: **Buy Now**. Prices expected to rise {f_change:.1f}% by next week.")
        elif f_change < -1:
            st.success(f"Strategy: **Wait**. Prices expected to drop {abs(f_change):.1f}% by next week.")
        else:
            st.info("Strategy: **Neutral**. No significant price movement predicted.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

with tab3:
    st.subheader("Consumer Recommendations")
    if monthly_change > 3:
        st.error("Inflation Alert: Purchasing power is decreasing.")
        st.write("- Prioritize long-shelf-life staples immediately.")
        st.write("- Compare unit prices across different retailers.")
    else:
        st.success("Stable Pricing: Standard shopping patterns recommended.")
        st.write("- No urgent bulk purchases required.")
    
    st.divider()
    st.write("### Model Performance Benchmarks")
    st.table(comparison.style.highlight_min(subset=['MAPE (%)', 'RMSE'], color='#d4edda'))

st.divider()
st.caption("Automated insights powered by LightGBM and SARIMA optimization.")
