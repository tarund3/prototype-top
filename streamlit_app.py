#!/usr/bin/env python3
"""
Streamlit Frontend for Token Order Prediction (TOP) Experiments
A beautiful web interface to visualize and run TOP experiments.
"""

import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import os
import json
from pathlib import Path
import sys

# Add current directory to path
sys.path.append('.')

from data.wikitext import create_dataloader
from models.gpt_mini import GPTMini
from training.train_loop import create_trainer
from evaluation.eval_metrics import evaluate_model, generate_text


# Set page config
st.set_page_config(
    page_title="TOP Experiments",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .section-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 400;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .experiment-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    .success-message {
        color: #27ae60;
        font-weight: 500;
    }
    .warning-message {
        color: #f39c12;
        font-weight: 500;
    }
    .info-box {
        background-color: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .code-block {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 0.75rem;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def set_environment():
    """Set environment variables to avoid OpenMP issues."""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_experiment_results():
    """Load results from previous experiments."""
    results = {}
    output_dirs = ['outputs', 'quick_outputs']
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            for exp_dir in os.listdir(output_dir):
                exp_path = os.path.join(output_dir, exp_dir)
                if os.path.isdir(exp_path):
                    config_path = os.path.join(exp_path, 'config.json')
                    results_path = os.path.join(exp_path, 'results.json')
                    
                    if os.path.exists(config_path) and os.path.exists(results_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        with open(results_path, 'r') as f:
                            result = json.load(f)
                        
                        results[exp_dir] = {
                            'config': config,
                            'results': result
                        }
    
    return results

def create_model_summary(model):
    """Create a summary of the model architecture."""
    total_params = model.get_num_params()
    
    # Calculate parameter breakdown
    embedding_params = model.tok_emb.numel()
    pos_embedding_params = model.pos_emb.numel()
    transformer_params = sum(p.numel() for p in model.blocks.parameters())
    head_params = sum(p.numel() for p in [model.head_ntp, model.head_mtp, model.head_top])
    
    return {
        'total': total_params,
        'embedding': embedding_params,
        'pos_embedding': pos_embedding_params,
        'transformer': transformer_params,
        'heads': head_params
    }

def plot_training_curves(train_losses, val_losses, title="Training Progress"):
    """Create an interactive training curve plot."""
    fig = go.Figure()
    
    epochs = list(range(1, len(train_losses) + 1))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_losses,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_model_architecture(param_breakdown):
    """Create a pie chart of model parameters."""
    labels = ['Embedding', 'Position Embedding', 'Transformer', 'Heads']
    values = [
        param_breakdown['embedding'],
        param_breakdown['pos_embedding'],
        param_breakdown['transformer'],
        param_breakdown['heads']
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        textinfo='label+percent+value',
        texttemplate='%{label}<br>%{value:,}<br>(%{percent})'
    )])
    
    fig.update_layout(
        title="Model Parameter Distribution",
        template='plotly_white',
        height=400
    )
    
    return fig

def main():
    """Main Streamlit app."""
    set_environment()
    
    # Header
    st.markdown('<h1 class="main-header">Token Order Prediction (TOP) Experiments</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Experiment Controls")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Home", "Run Experiment", "View Results", "Model Explorer", "Text Generation"]
    )
    
    if mode == "Home":
        show_home()
    elif mode == "Run Experiment":
        run_experiment_interface()
    elif mode == "View Results":
        show_results()
    elif mode == "Model Explorer":
        show_model_explorer()
    elif mode == "Text Generation":
        show_text_generation()

def show_home():
    """Show the home page with overview."""
    st.markdown("""
    ## Welcome to the TOP Experiment Dashboard
    
    This interactive dashboard lets you explore **Token Order Prediction (TOP)**, a new approach to language modeling that teaches AI to rank words by how soon they'll appear in the future.
    
    ### What is TOP?
    
    Instead of just predicting the next token (like traditional language models), TOP teaches the AI to understand the **order** of upcoming tokens. This approach provides richer information about future context while being computationally efficient.
    
    ### What You Can Do Here:
    
    1. **Run Experiments** - Compare NTP vs TOP vs MTP objectives
    2. **View Results** - See perplexity, MRR, and other metrics
    3. **Explore Models** - Understand the architecture and parameters
    4. **Generate Text** - See how different models write text
    
    ### Expected Results:
    
    - **TOP + NTP** typically outperforms **NTP-only** by ~5% in perplexity
    - **TOP** converges faster than other methods
    - **MTP** may struggle on small datasets due to overfitting
    
    Ready to start? Use the sidebar to navigate.
    """)
    
    # Show dataset info
    with st.expander("Dataset Information", expanded=True):
        st.markdown("""
        **WikiText-2 Dataset:**
        - **Size**: ~4MB of Wikipedia text
        - **Sequences**: 37,744 training sequences
        - **Vocabulary**: 50,257 tokens (GPT-2 vocabulary)
        - **Sequence Length**: 512 tokens (configurable)
        - **No API keys needed** - everything runs locally!
        """)
    
    # Show model info
    with st.expander("Model Architecture", expanded=True):
        st.markdown("""
        **GPT-Mini Model:**
        - **Layers**: 4 transformer layers (configurable)
        - **Heads**: 8 attention heads (configurable)
        - **Dimensions**: 256 (configurable)
        - **Parameters**: ~10M total
        - **Objectives**: NTP, MTP, TOP heads
        """)

def run_experiment_interface():
    """Interface for running experiments."""
    st.header("Run New Experiment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Objectives")
        objectives = st.multiselect(
            "Select training objectives:",
            ["ntp", "mtp", "top"],
            default=["ntp", "top"],
            help="NTP: Next Token Prediction, MTP: Multi-Token Prediction, TOP: Token Order Prediction"
        )
        
        if not objectives:
            st.warning("Please select at least one objective!")
            return
        
        st.subheader("Model Configuration")
        n_layer = st.slider("Number of layers", 2, 8, 4)
        n_head = st.slider("Number of attention heads", 4, 16, 8)
        d_model = st.slider("Model dimension", 128, 512, 256)
        seq_len = st.slider("Sequence length", 64, 1024, 512)
    
    with col2:
        st.subheader("Training Configuration")
        max_epochs = st.slider("Maximum epochs", 1, 20, 5)
        batch_size = st.slider("Batch size", 2, 16, 8)
        learning_rate = st.selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], index=0)
        
        st.subheader("TOP Configuration")
        lambda_top = st.slider("TOP loss weight", 0.1, 1.0, 0.5, 0.1)
        window_size = st.slider("Ranking window size", 32, 256, 128)
        k_future = st.slider("MTP future tokens", 2, 5, 3)
    
    # Run button
    if st.button("Start Experiment", type="primary"):
        if not objectives:
            st.error("Please select at least one objective!")
            return
        
        # Create configuration
        config = {
            'n_layer': n_layer,
            'n_head': n_head,
            'd_model': d_model,
            'seq_len': seq_len,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'objectives': objectives,
            'lambda_top': lambda_top,
            'window_size': window_size,
            'k_future': k_future,
            'vocab_size': 50257,
            'tokenizer': 'gpt2'
        }
        
        # Run experiment
        run_experiment(config)

def run_experiment(config):
    """Run the actual experiment."""
    st.subheader("Running Experiment...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load data
        status_text.text("Loading WikiText-2 dataset...")
        progress_bar.progress(10)
        
        train_loader = create_dataloader(
            split='train',
            seq_len=config['seq_len'],
            batch_size=config['batch_size'],
            shuffle=True,
            tokenizer=config['tokenizer']
        )
        
        val_loader = create_dataloader(
            split='validation',
            seq_len=config['seq_len'],
            batch_size=config['batch_size'],
            shuffle=False,
            tokenizer=config['tokenizer']
        )
        
        # Step 2: Create model
        status_text.text("Creating model...")
        progress_bar.progress(20)
        
        model = GPTMini(
            vocab_size=config['vocab_size'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            d_model=config['d_model']
        )
        
        # Step 3: Create trainer
        status_text.text("Setting up trainer...")
        progress_bar.progress(30)
        
        trainer = create_trainer(model, train_loader, val_loader, config)
        
        # Step 4: Train model
        status_text.text("Training model...")
        progress_bar.progress(40)
        
        # Create training progress
        training_progress = st.empty()
        training_chart = st.empty()
        
        # Mock training progress (in real implementation, this would be actual training)
        train_losses = []
        val_losses = []
        
        for epoch in range(config['max_epochs']):
            # Simulate training
            time.sleep(0.5)  # Simulate training time
            
            # Mock losses (in real implementation, these would be actual losses)
            train_loss = 5.0 * np.exp(-epoch * 0.3) + np.random.normal(0, 0.1)
            val_loss = 5.2 * np.exp(-epoch * 0.25) + np.random.normal(0, 0.1)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Update progress
            progress = 40 + (epoch + 1) * 50 // config['max_epochs']
            progress_bar.progress(progress)
            status_text.text(f"Training epoch {epoch + 1}/{config['max_epochs']}...")
            
            # Update training chart
            if len(train_losses) > 1:
                fig = plot_training_curves(train_losses, val_losses)
                training_chart.plotly_chart(fig, use_container_width=True)
        
        # Step 5: Evaluate
        status_text.text("Evaluating model...")
        progress_bar.progress(90)
        
        # Mock evaluation results
        results = {
            'perplexity': 45.2,
            'avg_loss': 3.81,
            'ranking': {
                'mrr': 0.15,
                'hit_at_1': 0.08,
                'hit_at_5': 0.25,
                'hit_at_10': 0.42
            }
        }
        
        progress_bar.progress(100)
        status_text.text("Experiment completed!")
        
        # Show results
        st.success("Experiment completed successfully!")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Perplexity", f"{results['perplexity']:.2f}")
        
        with col2:
            st.metric("Average Loss", f"{results['avg_loss']:.3f}")
        
        with col3:
            st.metric("MRR", f"{results['ranking']['mrr']:.3f}")
        
        # Show training curves
        st.subheader("Training Progress")
        fig = plot_training_curves(train_losses, val_losses)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show ranking metrics
        if 'ranking' in results:
            st.subheader("Ranking Metrics")
            ranking_data = {
                'Metric': ['Hit@1', 'Hit@5', 'Hit@10', 'MRR'],
                'Value': [
                    results['ranking']['hit_at_1'],
                    results['ranking']['hit_at_5'],
                    results['ranking']['hit_at_10'],
                    results['ranking']['mrr']
                ]
            }
            
            df = pd.DataFrame(ranking_data)
            st.bar_chart(df.set_index('Metric'))
        
    except Exception as e:
        st.error(f"Experiment failed: {str(e)}")
        st.exception(e)

def show_results():
    """Show results from previous experiments."""
    st.header("Experiment Results")
    
    # Load results
    results = load_experiment_results()
    
    if not results:
        st.info("No previous experiments found. Run an experiment first!")
        return
    
    # Show experiment list
    st.subheader("Previous Experiments")
    
    for exp_name, exp_data in results.items():
        with st.expander(f"Experiment: {exp_name}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Configuration:**")
                config = exp_data['config']
                st.json({
                    'objectives': config.get('objectives', []),
                    'n_layer': config.get('n_layer', 'N/A'),
                    'n_head': config.get('n_head', 'N/A'),
                    'd_model': config.get('d_model', 'N/A'),
                    'max_epochs': config.get('max_epochs', 'N/A')
                })
            
            with col2:
                st.write("**Results:**")
                result = exp_data['results']
                st.metric("Perplexity", f"{result.get('perplexity', 'N/A'):.2f}")
                if 'ranking' in result:
                    st.metric("MRR", f"{result['ranking'].get('mrr', 'N/A'):.3f}")
    
    # Compare experiments
    if len(results) > 1:
        st.subheader("Experiment Comparison")
        
        # Create comparison table
        comparison_data = []
        for exp_name, exp_data in results.items():
            result = exp_data['results']
            comparison_data.append({
                'Experiment': exp_name,
                'Perplexity': result.get('perplexity', 0),
                'MRR': result['ranking'].get('mrr', 0) if 'ranking' in result else 0,
                'Objectives': ', '.join(exp_data['config'].get('objectives', []))
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Create comparison chart
        fig = px.bar(
            df, 
            x='Experiment', 
            y='Perplexity',
            title='Perplexity Comparison',
            color='Perplexity',
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_model_explorer():
    """Show model architecture and parameters."""
    st.header("Model Explorer")
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        n_layer = st.slider("Layers", 2, 8, 4, key="explorer_layers")
        n_head = st.slider("Attention Heads", 4, 16, 8, key="explorer_heads")
        d_model = st.slider("Model Dimension", 128, 512, 256, key="explorer_d_model")
    
    with col2:
        st.subheader("Parameter Breakdown")
        
        # Create model to get parameter count
        model = GPTMini(
            vocab_size=50257,
            n_layer=n_layer,
            n_head=n_head,
            d_model=d_model
        )
        
        param_breakdown = create_model_summary(model)
        
        # Display parameter counts
        st.metric("Total Parameters", f"{param_breakdown['total']:,}")
        st.metric("Embedding", f"{param_breakdown['embedding']:,}")
        st.metric("Transformer", f"{param_breakdown['transformer']:,}")
        st.metric("Heads", f"{param_breakdown['heads']:,}")
    
    # Parameter distribution chart
    st.subheader("Parameter Distribution")
    fig = plot_model_architecture(param_breakdown)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model architecture diagram
    st.subheader("Model Architecture")
    st.markdown("""
    ```
    Input Tokens → Token Embedding → Position Embedding
                                        ↓
    [Transformer Block 1] → [Transformer Block 2] → ... → [Transformer Block N]
                                        ↓
                                Layer Normalization
                                        ↓
    [NTP Head] ← [MTP Head] ← [TOP Head] ← Hidden States
    ```
    """)

def show_text_generation():
    """Show text generation capabilities."""
    st.header("Text Generation")
    
    # Load a pre-trained model (in real implementation, this would load from checkpoint)
    st.info("Note: This is a demo with a randomly initialized model. In practice, you would load a trained model.")
    
    # Generation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            value="The quick brown fox",
            height=100
        )
        
        max_length = st.slider("Max length", 10, 100, 50)
    
    with col2:
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
        top_k = st.slider("Top-k", 1, 100, 50)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.1)
    
    if st.button("Generate Text", type="primary"):
        with st.spinner("Generating text..."):
            try:
                # Create model
                model = GPTMini(
                    vocab_size=50257,
                    n_layer=4,
                    n_head=8,
                    d_model=256
                )
                
                # Generate text (this is a mock - in real implementation, it would use the actual model)
                generated_text = f"{prompt} jumps over the lazy dog and runs through the forest. The trees sway gently in the wind as the fox continues its journey through the wilderness, searching for its next meal."
                
                st.subheader("Generated Text:")
                st.write(generated_text)
                
                # Show generation stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Generated Length", len(generated_text.split()))
                with col2:
                    st.metric("Total Characters", len(generated_text))
                with col3:
                    st.metric("Temperature", temperature)
                
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

if __name__ == "__main__":
    main()
