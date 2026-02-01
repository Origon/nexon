#!/bin/bash
# Nexon - Process management for mlx_lm server + web UI

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check and install mlx_lm if needed
check_mlx_lm() {
    if ! python3 -c "import mlx_lm" 2>/dev/null; then
        echo "mlx_lm not found. Installing..."
        pip install mlx-lm
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install mlx-lm"
            exit 1
        fi
        echo "mlx_lm installed successfully"
    fi
}
PID_FILE="$SCRIPT_DIR/.nexon.pid"
LOG_FILE="$SCRIPT_DIR/.nexon.log"

# Defaults (can be overridden by env vars or flags)
MODEL="${NEXON_MODEL:-}"
ADAPTER="${NEXON_ADAPTER:-}"
MLX_PORT="${NEXON_PORT:-8080}"
WEB_PORT=3000

usage() {
    echo "Usage: $0 [OPTIONS] {start|stop|restart|status|logs}"
    echo ""
    echo "Commands:"
    echo "  start   - Start mlx_lm server and web UI"
    echo "  stop    - Stop all services"
    echo "  restart - Restart all services"
    echo "  status  - Show running status"
    echo "  logs    - Tail the log file"
    echo ""
    echo "Options:"
    echo "  -m, --model PATH     Model path or HuggingFace ID"
    echo "  -a, --adapter PATH   LoRA adapter path"
    echo "  -p, --port NUM       API port (default: 8080)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 -m ~/models/my-model start"
    echo "  $0 -m ~/models/base -a ./adapters/my-adapter start"
    echo "  $0 -m mlx-community/Qwen3-4B-4bit start"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)   MODEL="$2"; shift 2 ;;
        -a|--adapter) ADAPTER="$2"; shift 2 ;;
        -p|--port)    MLX_PORT="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        -*)           echo "Unknown option: $1"; usage; exit 1 ;;
        *)            COMMAND="$1"; shift; break ;;
    esac
done

start() {
    if [ -z "$MODEL" ]; then
        echo "Error: Model is required"
        echo "Usage: nexon -m <model> start"
        echo ""
        echo "Example: nexon -m mlx-community/Qwen3-8B-4bit start"
        exit 1
    fi

    # Validate local model path
    if [[ "$MODEL" == /* ]] || [[ "$MODEL" == ~* ]] || [[ "$MODEL" == ./* ]]; then
        # Expand ~ if present
        MODEL="${MODEL/#\~/$HOME}"

        if [ ! -d "$MODEL" ]; then
            echo "Error: Model directory not found: $MODEL"
            exit 1
        fi

        if [ ! -f "$MODEL/config.json" ]; then
            echo "Error: Invalid model directory (missing config.json)"
            echo "Path: $MODEL"
            echo ""
            echo "Make sure this is an MLX model directory with weights, not a metadata folder."
            exit 1
        fi
    fi

    check_mlx_lm

    if [ -f "$PID_FILE" ]; then
        if pgrep -F "$PID_FILE" > /dev/null 2>&1; then
            echo "Nexon is already running (PID: $(cat $PID_FILE))"
            return 1
        fi
        rm -f "$PID_FILE"
    fi

    echo "Starting Nexon..."
    echo "Model: $MODEL"
    [ -n "$ADAPTER" ] && echo "Adapter: $ADAPTER"
    echo "API: http://localhost:$MLX_PORT"
    echo "Web: http://localhost:$WEB_PORT"
    echo ""

    # Start mlx_lm server
    if [ -n "$ADAPTER" ]; then
        python3 -m mlx_lm.server --model "$MODEL" --adapter-path "$ADAPTER" --port $MLX_PORT >> "$LOG_FILE" 2>&1 &
    else
        python3 -m mlx_lm.server --model "$MODEL" --port $MLX_PORT >> "$LOG_FILE" 2>&1 &
    fi
    MLX_PID=$!

    # Wait for server to start
    echo -n "Waiting for model to load"
    for i in {1..60}; do
        if curl -s "http://localhost:$MLX_PORT/v1/models" > /dev/null 2>&1; then
            echo " ready!"
            break
        fi
        echo -n "."
        sleep 1
    done

    # Start web server (with proxy for token filtering)
    python3 "$SCRIPT_DIR/nexon-server.py" --mlx-port $MLX_PORT --web-port $WEB_PORT >> "$LOG_FILE" 2>&1 &
    WEB_PID=$!

    # Save PIDs
    echo "$MLX_PID $WEB_PID" > "$PID_FILE"

    echo ""
    echo "Nexon started!"
    echo "  MLX server PID: $MLX_PID"
    echo "  Web server PID: $WEB_PID"
    echo ""
    echo "Logs: nexon logs"
    echo "Stop: nexon stop"

    # Open browser
    sleep 1
    open "http://localhost:$WEB_PORT"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Nexon is not running"
        return 0
    fi

    echo "Stopping Nexon..."

    read MLX_PID WEB_PID < "$PID_FILE"

    [ -n "$MLX_PID" ] && kill $MLX_PID 2>/dev/null && echo "Stopped MLX server (PID: $MLX_PID)"
    [ -n "$WEB_PID" ] && kill $WEB_PID 2>/dev/null && echo "Stopped web server (PID: $WEB_PID)"

    rm -f "$PID_FILE"
    echo "Nexon stopped"
}

restart() {
    stop
    sleep 1
    start
}

status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Nexon is not running"
        return 1
    fi

    read MLX_PID WEB_PID < "$PID_FILE"

    echo "Nexon status:"

    if ps -p $MLX_PID > /dev/null 2>&1; then
        echo "  MLX server: running (PID: $MLX_PID)"
    else
        echo "  MLX server: stopped"
    fi

    if ps -p $WEB_PID > /dev/null 2>&1; then
        echo "  Web server: running (PID: $WEB_PID)"
    else
        echo "  Web server: stopped"
    fi

    echo ""
    echo "Model: $MODEL"
    [ -n "$ADAPTER" ] && echo "Adapter: $ADAPTER"
    echo "API: http://localhost:$MLX_PORT"
    echo "Web: http://localhost:$WEB_PORT"
}

logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "No log file found"
    fi
}

case "${COMMAND:-}" in
    start)   start ;;
    stop)    stop ;;
    restart) restart ;;
    status)  status ;;
    logs)    logs ;;
    *)       usage ;;
esac
