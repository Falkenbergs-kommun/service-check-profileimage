#!/bin/bash
set -e
sleep 3

# Installera systemberoenden
apt-get update
apt-get install -y libgl1 build-essential cmake libboost-all-dev libopenblas-dev liblapack-dev libglib2.0-0 libsm6 libxrender1 libxext6

echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r /app/requirements.txt
