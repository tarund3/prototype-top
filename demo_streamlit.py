#!/usr/bin/env python3
"""
Demo script showing the key features of the TOP Streamlit app.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def main():
    """Demo the key features."""
    st.set_page_config(
        page_title="TOP Demo",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("TOP Experiment Dashboard Demo")
    st.markdown("This is a preview of the interactive features you'll get with the full Streamlit app!")
    
    # Demo 1: Model Architecture
    st.header("Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parameter Breakdown")
        param_data = {
            'Component': ['Embedding', 'Position Embedding', 'Transformer', 'Heads'],
            'Parameters': [12800000, 262144, 2000000, 15000000],
            'Percentage': [40, 1, 6, 53]
        }
        
        df = pd.DataFrame(param_data)
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("Parameter Distribution")
        fig = px.pie(
            df, 
            values='Parameters', 
            names='Component',
            title="Model Parameter Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Demo 2: Training Progress
    st.header("Training Progress")
    
    # Simulate training data
    epochs = list(range(1, 11))
    train_losses = [5.0 * np.exp(-e * 0.3) + np.random.normal(0, 0.1) for e in epochs]
    val_losses = [5.2 * np.exp(-e * 0.25) + np.random.normal(0, 0.1) for e in epochs]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode='lines+markers', name='Training Loss'))
    fig.add_trace(go.Scatter(x=epochs, y=val_losses, mode='lines+markers', name='Validation Loss'))
    
    fig.update_layout(
        title="Training and Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Demo 3: Experiment Comparison
    st.header("Experiment Comparison")
    
    comparison_data = {
        'Experiment': ['NTP Only', 'NTP + MTP', 'NTP + TOP', 'All Objectives'],
        'Perplexity': [45.2, 43.8, 42.1, 41.5],
        'MRR': [0.0, 0.0, 0.15, 0.18],
        'Training Time': [120, 180, 150, 200]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Results Table")
        st.dataframe(df_comparison, use_container_width=True)
    
    with col2:
        st.subheader("Perplexity Comparison")
        fig = px.bar(
            df_comparison, 
            x='Experiment', 
            y='Perplexity',
            title="Perplexity by Experiment",
            color='Perplexity',
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Demo 4: Interactive Controls
    st.header("Interactive Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Model Configuration")
        n_layer = st.slider("Layers", 2, 8, 4)
        n_head = st.slider("Attention Heads", 4, 16, 8)
        d_model = st.slider("Model Dimension", 128, 512, 256)
    
    with col2:
        st.subheader("Training Configuration")
        max_epochs = st.slider("Max Epochs", 1, 20, 5)
        batch_size = st.slider("Batch Size", 2, 16, 8)
        learning_rate = st.selectbox("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3])
    
    with col3:
        st.subheader("TOP Configuration")
        lambda_top = st.slider("TOP Loss Weight", 0.1, 1.0, 0.5, 0.1)
        window_size = st.slider("Ranking Window", 32, 256, 128)
        k_future = st.slider("MTP Future Tokens", 2, 5, 3)
    
    # Show current configuration
    st.subheader("Current Configuration")
    config = {
        'Layers': n_layer,
        'Heads': n_head,
        'Dimensions': d_model,
        'Epochs': max_epochs,
        'Batch Size': batch_size,
        'Learning Rate': learning_rate,
        'TOP Weight': lambda_top,
        'Window Size': window_size,
        'Future Tokens': k_future
    }
    
    st.json(config)
    
    # Demo 5: Text Generation
    st.header("Text Generation")
    
    prompt = st.text_area("Enter your prompt:", "The quick brown fox")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
        max_length = st.slider("Max Length", 10, 100, 50)
    
    with col2:
        top_k = st.slider("Top-k", 1, 100, 50)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.1)
    
    if st.button("Generate Text"):
        # Mock generated text
        generated = f"{prompt} jumps over the lazy dog and runs through the forest. The trees sway gently in the wind as the fox continues its journey through the wilderness, searching for its next meal."
        
        st.subheader("Generated Text:")
        st.write(generated)
        
        # Show generation stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Generated Length", len(generated.split()))
        with col2:
            st.metric("Total Characters", len(generated))
        with col3:
            st.metric("Temperature", temperature)
    
    # Demo 6: Key Features
    st.header("Key Features")
    
    features = [
        "**Interactive Experiment Controls** - Adjust model parameters in real-time",
        "**Real-time Training Visualization** - Watch loss curves update as training progresses",
        "**Model Architecture Explorer** - Understand parameter distribution and model structure",
        "**Experiment Comparison** - Compare different objective combinations side-by-side",
        "**Text Generation Interface** - Generate text with different models and parameters",
        "**Comprehensive Configuration** - Fine-tune all aspects of the training process",
        "**Responsive Design** - Works on desktop, tablet, and mobile devices",
        "**One-Click Experiments** - Run experiments with a single button click"
    ]
    
    for feature in features:
        st.markdown(feature)
    
    st.success("This is just a preview! Run the full app with `python run_streamlit.py` to get the complete interactive experience!")

if __name__ == "__main__":
    main()
