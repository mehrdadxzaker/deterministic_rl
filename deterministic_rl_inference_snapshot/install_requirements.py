
# install_requirements.py
# Minimal deps for Colab/local use.
import sys, subprocess

pkgs = [
    "torch>=2.0.0",
    "numpy>=1.23.0",
]

for p in pkgs:
    print("Installing", p)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

print("Done.")
