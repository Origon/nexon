#!/bin/bash
# Nexon installer - adds 'nexon' command to your PATH

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="${HOME}/.local/bin"

echo "Nexon Installer"
echo "==============="
echo ""

# Check for Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: Nexon is optimized for Apple Silicon (M1/M2/M3/M4)."
    echo "Performance on Intel Macs will be limited."
    echo ""
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    echo "Download from: https://www.python.org/downloads/"
    exit 1
fi

# Install mlx-lm if needed
echo "Checking dependencies..."
if ! python3 -c "import mlx_lm" 2>/dev/null; then
    echo "Installing mlx-lm..."
    pip3 install mlx-lm
fi

# Create bin directory
mkdir -p "$INSTALL_DIR"

# Create nexon wrapper script
cat > "$INSTALL_DIR/nexon" << EOF
#!/bin/bash
exec "$SCRIPT_DIR/nexon.sh" "\$@"
EOF

chmod +x "$INSTALL_DIR/nexon"

# Add to PATH if needed
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    SHELL_RC=""
    if [[ -f "$HOME/.zshrc" ]]; then
        SHELL_RC="$HOME/.zshrc"
    elif [[ -f "$HOME/.bashrc" ]]; then
        SHELL_RC="$HOME/.bashrc"
    fi

    if [[ -n "$SHELL_RC" ]]; then
        echo "" >> "$SHELL_RC"
        echo "# Nexon" >> "$SHELL_RC"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$SHELL_RC"
        echo ""
        echo "Added ~/.local/bin to PATH in $SHELL_RC"
        echo "Run: source $SHELL_RC"
    else
        echo ""
        echo "Add this to your shell profile:"
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
fi

echo ""
echo "Installation complete!"
echo ""
echo "Usage:"
echo "  nexon -m <model> start     Start server with model"
echo "  nexon stop                 Stop server"
echo "  nexon -h                   Show help"
echo ""
echo "Quick start:"
echo "  nexon -m mlx-community/Qwen3-4B-4bit start"
echo "  open http://localhost:3000"
