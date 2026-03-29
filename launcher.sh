#!/bin/bash

set -e  # crash immediately if anything fails (important)

echo "🚀 Starting setup..."

# ========= SYSTEM SETUP =========
sudo apt update -y
sudo apt install -y python3 python3-pip git

# ========= PYTHON ENV =========
python3 -m pip install --upgrade pip

# core deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate trl
pip install unsloth

# ========= OPTIONAL: HF LOGIN =========
# export HF_TOKEN="your_token_here"
# huggingface-cli login --token $HF_TOKEN

# ========= CLONE YOUR REPO =========
echo "📦 Cloning repo..."
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# ========= INSTALL EXTRA REQS =========
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# ========= RUN TRAINING =========
echo "🔥 Starting training..."
python train.py

echo "✅ Training complete!"

# ========= UPLOAD (if inside script skip this) =========
# python upload.py

# ========= CLEAN SHUTDOWN =========
echo "💤 Shutting down instance..."
sudo shutdown now
