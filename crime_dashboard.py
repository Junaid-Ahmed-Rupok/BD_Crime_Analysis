import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Crime Classification Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .feature-importance-plot {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_resource
def load_data():
    # In a real scenario, we would load the actual data and model
    # For demonstration, we'll create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample feature data
    feature_names = ['feature_' + str(i) for i in range(1, 21)]
    X = np.random.randn(n_samples, 20)
    
    # Create sample target (crime categories)
    crime_categories = ['murder', 'rape', 'assault', 'bodyfound', 'kidnap', 'robbery']
    y = np.random.choice(crime_categories, n_samples, p=[0.1, 0.15, 0.2, 0.1, 0.15, 0.3])
    
    # Create sample feature importance data
    feature_importance = np.random.rand(20)
    feature_importance = feature_importance / feature_importance.sum()
    
    # Create sample performance metrics
    metrics = {
        'accuracy': 0.93,
        'precision': 0.93,
        'recall': 0.93,
        'f1': 0.93
    }
    
    # Create sample confusion matrix
    cm = confusion_matrix(
        np.random.choice(range(6), n_samples),
        np.random.choice(range(6), n_samples)
    )
    
    return {
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'crime_categories': crime_categories,
        'feature_importance': feature_importance,
        'metrics': metrics,
        'confusion_matrix': cm
    }

data = load_data()

# Sidebar
st.sidebar.title("Crime Classification Dashboard")
st.sidebar.markdown("Explore the performance of the crime classification model.")

# Main content
st.markdown('<h1 class="main-header">Crime Classification Analysis</h1>', unsafe_allow_html=True)

# Overview section
st.header("Model Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Accuracy", f"{data['metrics']['accuracy']*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Precision", f"{data['metrics']['precision']*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Recall", f"{data['metrics']['recall']*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("F1 Score", f"{data['metrics']['f1']*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# Data distribution
st.header("Data Distribution")
col1, col2 = st.columns(2)

with col1:
    # Crime category distribution
    crime_counts = pd.Series(data['y']).value_counts()
    fig = px.pie(
        values=crime_counts.values,
        names=crime_counts.index,
        title="Distribution of Crime Categories"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': data['feature_names'][:10],
        'importance': data['feature_importance'][:10]
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title="Top 10 Most Important Features"
    )
    st.plotly_chart(fig, use_container_width=True)

# Model performance
st.header("Model Performance Evaluation")

# Confusion matrix
st.subheader("Confusion Matrix")
fig = px.imshow(
    data['confusion_matrix'],
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=data['crime_categories'],
    y=data['crime_categories'],
    aspect="auto"
)
fig.update_xaxes(side="top")
st.plotly_chart(fig, use_container_width=True)

# Classification report
st.subheader("Classification Report")
report_data = []
for i, crime in enumerate(data['crime_categories']):
    report_data.append({
        'Class': crime,
        'Precision': np.random.uniform(0.8, 1.0),
        'Recall': np.random.uniform(0.8, 1.0),
        'F1-Score': np.random.uniform(0.8, 1.0),
        'Support': np.random.randint(50, 200)
    })

report_df = pd.DataFrame(report_data)
st.dataframe(report_df, use_container_width=True)

# Model comparison
st.header("Model Comparison")
models = ['XGBoost', 'AdaBoost', 'Random Forest', 'Logistic Regression']
accuracy = [0.93, 0.73, 0.85, 0.68]
training_time = [12.5, 8.2, 6.7, 3.1]

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(x=models, y=accuracy, name="Accuracy", marker_color='#1f77b4'),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=models, y=training_time, name="Training Time (s)", marker_color='#ff7f0e'),
    secondary_y=True,
)
fig.update_layout(
    title_text="Model Comparison: Accuracy and Training Time"
)
fig.update_xaxes(title_text="Model")
fig.update_yaxes(title_text="Accuracy", secondary_y=False)
fig.update_yaxes(title_text="Training Time (seconds)", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

# Learning curves
st.header("Learning Curves")
learning_curve_data = {
    'Training Size': [100, 200, 300, 400, 500, 600, 700, 800],
    'XGBoost Train': [0.75, 0.82, 0.87, 0.90, 0.92, 0.93, 0.94, 0.94],
    'XGBoost Validation': [0.70, 0.78, 0.84, 0.88, 0.90, 0.91, 0.92, 0.93],
    'AdaBoost Train': [0.65, 0.70, 0.72, 0.73, 0.73, 0.73, 0.73, 0.73],
    'AdaBoost Validation': [0.63, 0.68, 0.71, 0.72, 0.73, 0.73, 0.73, 0.73]
}

lc_df = pd.DataFrame(learning_curve_data)
fig = px.line(
    lc_df,
    x='Training Size',
    y=['XGBoost Train', 'XGBoost Validation', 'AdaBoost Train', 'AdaBoost Validation'],
    title='Learning Curves for Different Models'
)
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### About This Dashboard")
st.markdown("""
This dashboard presents the analysis of a crime classification model that categorizes crimes into:
- Murder (0)
- Rape (1)
- Assault (2)
- Body Found (3)
- Kidnap (4)
- Robbery (5)

The XGBoost model achieved the best performance with 93% accuracy.
""")