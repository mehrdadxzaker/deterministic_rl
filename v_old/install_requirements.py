"""
Installation script for Deterministic RL Inference Research
Run this first in your Colab environment
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Core ML packages
packages = [
    "torch>=1.9.0",
    "numpy>=1.21.0", 
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    
    # RL specific
    "gym>=0.21.0",
    "stable-baselines3>=1.6.0",
    "tensorboard>=2.7.0",
    
    # Visualization and analysis
    "plotly>=5.0.0",
    "tqdm>=4.62.0",
    "jupyter>=1.0.0",
    
    # For PDDL domains (optional, for later experiments)
    "pddlgym",
    
    # Utilities
    "python-dotenv>=0.19.0",
    "joblib>=1.1.0"
]

def main():
    print("Installing packages for Deterministic RL Inference Research...")
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            install_package(package)
            print(f"✓ {package} installed successfully")
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}")
    
    print("\n" + "="*50)
    print("Installation complete!")
    print("Next steps:")
    print("1. Run helper_functions.py to load utilities")
    print("2. Choose an experiment file to run")
    print("="*50)

if __name__ == "__main__":
    main()

# Quick test to verify key packages
def test_imports():
    """Test that key packages import correctly"""
    try:
        import torch
        import numpy as np
        import gym
        import stable_baselines3
        import matplotlib.pyplot as plt
        print("✓ All core packages imported successfully!")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Gym version: {gym.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

# Uncomment to run test after installation
# test_imports()