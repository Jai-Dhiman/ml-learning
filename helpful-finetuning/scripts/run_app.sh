#!/usr/bin/env bash
set -euo pipefail

if [ -d .venv ]; then
  source .venv/bin/activate
fi

# Start Streamlit comparison app
streamlit run streamlit_app/app.py --server.port 8501 --server.address 127.0.0.1
