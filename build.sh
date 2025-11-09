#!/usr/bin/env bash

# exit on error
set -o errexit

# Install system dependencies
apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    pkg-config

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Django commands
python manage.py collectstatic --no-input
python manage.py migrate