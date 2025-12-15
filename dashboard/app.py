"""
ML Monitoring Dashboard - Product Return Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.data_drift import DataDriftDetector

# Page config
st.set_page_config(
    page_title="Product Return Monitoring",
    page_icon="🛍️",
    layout="wide"
)

# Title
st.markdown("# 🛍️ E-Commerce Return Monitoring System")
st.markdown("**Real-time monitoring for product return prediction model**")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    baseline = pd.read_csv('../data/train.csv')
    batch_1 = pd.read_csv('../data/batch_1_normal.csv')
    batch_2 = pd.read_csv('../data/batch_2_flash_sale.csv')
    batch_3 = pd.read_csv('../data/batch_3_supply_crisis.csv')
    return baseline, batch_1, batch_2, batch_3

baseline_df, batch_1, batch_2, batch_3 = load_data()

# Sidebar
st.sidebar.header("⚙️ Configuration")
selected_batch = st.sidebar.selectbox(
    "Select Batch to Monitor",
    ["Batch 1 (Normal)", "Batch 2 (Flash Sale)", "Batch 3 (Supply Crisis)"]
)

batch_map = {
    "Batch 1 (Normal)": batch_1,
    "Batch 2 (Flash Sale)": batch_2,
    "Batch 3 (Supply Crisis)": batch_3
}
current_batch = batch_map[selected_batch]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.info(
    "This dashboard monitors product return predictions and detects:\n"
    "- Data drift (input changes)\n"
    "- Return rate changes\n"
    "- Feature distribution shifts"
)

# Initialize detector
detector = DataDriftDetector()

# ===== MAIN METRICS =====
st.header("📊 Overview Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    baseline_return_rate = baseline_df['returned'].mean()
    current_return_rate = current_batch['returned'].mean()
    return_rate_change = ((current_return_rate - baseline_return_rate) / baseline_return_rate) * 100
    
    st.metric(
        "Return Rate",
        f"{current_return_rate:.1%}",
        f"{return_rate_change:+.1f}%",
        delta_color="inverse"
    )

with col2:
    baseline_discount = baseline_df['discount_percentage'].mean()
    current_discount = current_batch['discount_percentage'].mean()
    discount_change = current_discount - baseline_discount
    
    st.metric(
        "Avg Discount",
        f"{current_discount:.0f}%",
        f"{discount_change:+.0f}%",
        delta_color="normal"
    )

with col3:
    baseline_delivery = baseline_df['delivery_time_days'].mean()
    current_delivery = current_batch['delivery_time_days'].mean()
    delivery_change = current_delivery - baseline_delivery
    
    st.metric(
        "Avg Delivery Time",
        f"{current_delivery:.1f} days",
        f"{delivery_change:+.1f} days",
        delta_color="inverse"
    )

with col4:
    baseline_rating = baseline_df['review_rating'].mean()
    current_rating = current_batch['review_rating'].mean()
    rating_change = current_rating - baseline_rating
    
    st.metric(
        "Avg Rating",
        f"{current_rating:.1f}/5.0",
        f"{rating_change:+.1f}",
        delta_color="normal"
    )

# ===== DRIFT ANALYSIS =====
st.markdown("---")
st.header("🔍 Data Drift Analysis")

drift_results = detector.analyze_feature_drift(current_batch, baseline_df)
drift_score = detector.get_drift_score(drift_results)

col1, col2, col3 = st.columns(3)

severe_count = sum(1 for r in drift_results.values() if r['severity'] == 'Severe Drift')
moderate_count = sum(1 for r in drift_results.values() if r['severity'] == 'Moderate Drift')
no_drift_count = sum(1 for r in drift_results.values() if r['severity'] == 'No Drift')

with col1:
    st.metric("Overall Drift Score", f"{drift_score}/100",
              delta=f"{drift_score - 10:.1f}" if drift_score > 10 else None,
              delta_color="inverse")

with col2:
    st.metric("🚨 Severe Drift", severe_count,
              delta=severe_count if severe_count > 0 else None,
              delta_color="inverse")

with col3:
    st.metric("✅ No Drift", no_drift_count,
              delta=no_drift_count - 5 if no_drift_count < 5 else None,
              delta_color="normal")

# Status
if drift_score < 10:
    st.success("✅ **System Status: HEALTHY** - No significant drift detected")
elif drift_score < 20:
    st.warning("⚠️ **System Status: WARNING** - Moderate drift detected")
else:
    st.error("🚨 **System Status: CRITICAL** - Severe drift detected, investigate immediately!")

st.markdown("---")

# Feature drift table
st.subheader("📈 Feature-wise Drift Details")

drift_df = pd.DataFrame(drift_results).T
drift_df['feature'] = drift_df.index
drift_df = drift_df[['feature', 'psi', 'mean_shift_pct', 'severity', 'alert']]
drift_df.columns = ['Feature', 'PSI Score', 'Mean Shift %', 'Severity', 'Status']

st.dataframe(drift_df, use_container_width=True, height=300)

# Visualizations
col1, col2 = st.columns(2)

with col1:
    # PSI scores
    fig_psi = px.bar(
        drift_df,
        x='Feature',
        y='PSI Score',
        color='Severity',
        title="PSI Scores by Feature",
        color_discrete_map={
            'No Drift': 'green',
            'Moderate Drift': 'orange',
            'Severe Drift': 'red'
        }
    )
    fig_psi.add_hline(y=0.1, line_dash="dash", line_color="orange")
    fig_psi.add_hline(y=0.2, line_dash="dash", line_color="red")
    st.plotly_chart(fig_psi, use_container_width=True)

with col2:
    # Mean shifts
    fig_shift = px.bar(
        drift_df,
        x='Feature',
        y='Mean Shift %',
        title="Mean Value Shifts (%)",
        color='Mean Shift %',
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig_shift, use_container_width=True)

# Distribution comparison
st.markdown("---")
st.subheader("📉 Distribution Comparison")

selected_feature = st.selectbox(
    "Select feature to compare",
    ['product_price', 'customer_purchase_history', 'review_rating', 
     'delivery_time_days', 'discount_percentage', 'previous_returns', 
     'days_since_purchase']
)

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=baseline_df[selected_feature],
    name='Baseline (Training)',
    opacity=0.7,
    marker_color='blue',
    nbinsx=30
))
fig_dist.add_trace(go.Histogram(
    x=current_batch[selected_feature],
    name=f'Current ({selected_batch})',
    opacity=0.7,
    marker_color='red',
    nbinsx=30
))
fig_dist.update_layout(
    title=f'Distribution: {selected_feature}',
    xaxis_title=selected_feature,
    yaxis_title='Frequency',
    barmode='overlay',
    height=400
)
st.plotly_chart(fig_dist, use_container_width=True)

# ===== RECOMMENDATIONS =====
st.markdown("---")
st.header("💡 Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔍 Key Findings")
    
    if drift_score > 20:
        st.error(f"- Critical drift detected (score: {drift_score}/100)")
    elif drift_score > 10:
        st.warning(f"- Moderate drift detected (score: {drift_score}/100)")
    else:
        st.success(f"- Data distribution stable (score: {drift_score}/100)")
    
    st.info(f"- {severe_count} features show severe drift")
    st.info(f"- Return rate: {current_return_rate:.1%} (baseline: {baseline_return_rate:.1%})")

with col2:
    st.markdown("### 🎯 Recommended Actions")
    
    if drift_score > 20 or severe_count > 2:
        st.error("**URGENT:**")
        st.markdown("""
        1. ⚠️ Investigate root cause
        2. 🔄 Retrain model with recent data
        3. 🔔 Alert stakeholders
        4. 💾 Consider rollback
        """)
    elif drift_score > 10:
        st.warning("**MONITOR:**")
        st.markdown("""
        1. 👀 Monitor closely
        2. 📊 Collect more data
        3. 📅 Schedule retraining
        """)
    else:
        st.success("**MAINTAIN:**")
        st.markdown("""
        1. ✅ Continue monitoring
        2. 📝 Document patterns
        3. 🔄 Regular reviews
        """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: gray;">Built with Streamlit | ML Monitoring System</p>',
    unsafe_allow_html=True
)