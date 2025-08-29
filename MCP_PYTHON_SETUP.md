# Python Runner MCP Setup Guide for WARP

This guide covers setting up Python code execution capabilities in WARP using MCP (Model Context Protocol) servers.

## What You Get

With these MCP servers configured, I (your AI assistant) can:
- Execute Python code snippets interactively
- Install packages dynamically 
- Test your ML models and data processing
- Debug code by running it step-by-step
- Generate visualizations and plots
- Validate tensor shapes and operations in JAX/Flax

## Option 1: MCP Safe Local Python Executor (RECOMMENDED)

✅ **Already installed** at: `~/mcp_servers/safe-python-executor`

### Features
- Safe local execution (no Docker required)
- Supports your existing Python environments
- Works with conda, venv, and uv environments
- Perfect for your ML workflow

### WARP Configuration

Add this to your WARP/Claude Desktop configuration:

```json
{
  "mcpServers": {
    "safe-python-executor": {
      "command": "uv",
      "args": [
        "--directory", 
        "/Users/jdhiman/mcp_servers/safe-python-executor/",
        "run",
        "mcp_server.py"
      ],
      "env": {
        "UV_PYTHON": "3.12"
      }
    }
  }
}
```

### Testing the Installation

Run this to test if it works:

```bash
cd ~/mcp_servers/safe-python-executor
uv run mcp_server.py
```

You should see MCP server output. Press Ctrl+C to stop.

## Option 2: MCP Code Executor (More Features)

This provides more advanced features but requires Node.js setup:

### Installation

```bash
# Install Node.js if you don't have it
brew install node

# Clone and setup
git clone https://github.com/bazinga012/mcp_code_executor.git ~/mcp_servers/code-executor
cd ~/mcp_servers/code-executor
npm install
npm run build
```

### WARP Configuration

```json
{
  "mcpServers": {
    "mcp-code-executor": {
      "command": "node",
      "args": [
        "/Users/jdhiman/mcp_servers/code-executor/build/index.js"
      ],
      "env": {
        "CODE_STORAGE_DIR": "/Users/jdhiman/mcp_servers/code-executor/storage",
        "ENV_TYPE": "venv-uv",
        "UV_VENV_PATH": "/Users/jdhiman/Documents/ml-learning/.venv"
      }
    }
  }
}
```

## Recommended Setup for Your ML Projects

### For Your ml-learning Repository

1. **Create a dedicated ML environment**:
```bash
cd ~/Documents/ml-learning
uv venv .venv
uv pip install jax flax wandb librosa pandas numpy matplotlib seaborn jupyter
```

2. **Configure MCP to use this environment**:
Use the first option (safe-python-executor) with your project's .venv

3. **Test with your projects**:
Once configured, I can help you:
- Test your JAX/Flax models interactively
- Analyze your PercePiano dataset
- Debug audio processing pipelines
- Validate safety classifier preprocessing

## Available MCP Tools

Once configured, I'll have access to tools like:

- `execute_code` - Run Python snippets
- `install_dependencies` - Install packages dynamically  
- `check_installed_packages` - Verify what's installed
- `configure_environment` - Switch Python environments
- `initialize_code_file` - Create persistent Python files
- `read_code_file` - Read existing code files

## Benefits for Your ML Workflow

### JAX/Flax Development
```python
# I can test your model components
import jax.numpy as jnp
from flax import linen as nn

# Validate tensor operations
dummy_input = jnp.ones((32, 128, 128, 1))  # Batch of spectrograms
print(f"Input shape: {dummy_input.shape}")
```

### Data Analysis
```python
# Explore your PercePiano dataset
import json
import pandas as pd

data = json.load(open('piano-analysis-model/data/percepio_labels.json'))
df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print(f"Perceptual dimensions: {df.columns.tolist()}")
```

### Audio Processing
```python
# Test librosa workflows
import librosa
import matplotlib.pyplot as plt

# Quick spectrogram analysis
y, sr = librosa.load('sample.wav', duration=30)
S = librosa.feature.melspectrogram(y=y, sr=sr)
plt.figure(figsize=(10, 4))
librosa.display.specshow(S, x_axis='time', y_axis='mel')
```

## Next Steps

1. **Choose your preferred option** (I recommend Option 1)
2. **Add the configuration** to your WARP settings
3. **Restart WARP** to load the MCP server
4. **Test by asking me** to run Python code!

## Troubleshooting

### Common Issues

**"Command not found: uv"**
- Solution: Make sure `export PATH="$HOME/.local/bin:$PATH"` is in your ~/.zshrc

**"MCP server connection failed"**  
- Check the file paths in your configuration
- Ensure the server runs manually first
- Verify environment variables

**"Package not found"**
- The MCP server can install packages dynamically
- Or pre-install in your project's .venv

### Testing Commands

```bash
# Test uv installation
uv --version

# Test MCP server manually
cd ~/mcp_servers/safe-python-executor
uv run mcp_server.py

# Test your ML environment
cd ~/Documents/ml-learning
uv run python -c "import jax; print(jax.devices())"
```

## Configuration Files Location

WARP/Claude Desktop configuration is typically at:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Create the file if it doesn't exist, and add the `mcpServers` section.
