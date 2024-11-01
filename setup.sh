#!/bin/bash

# Create conda environment
conda create -n energy_dashboard python=3.9 -y

# Activate environment
conda activate energy_dashboard

# Install required packages
pip install streamlit pandas plotly openpyxl

echo "Setup complete! Now you can run: streamlit run app.py"