#!/usr/bin/env python3
"""
Launcher script for the Streamlit TOP experiment dashboard.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app."""
    print("Launching TOP Experiment Dashboard...")
    print("=" * 50)
    
    # Set environment variables
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("Streamlit is installed")
    except ImportError:
        print("Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas"])
        print("Streamlit installed successfully")
    
    # Launch the app
    print("Starting web interface...")
    print("Open your browser to: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error launching Streamlit: {e}")

if __name__ == "__main__":
    main()
