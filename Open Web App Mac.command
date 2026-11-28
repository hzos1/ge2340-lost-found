#!/bin/bash
cd "$(dirname "$0")"
URL="http://127.0.0.1:8765"

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Python 3.10 or newer is required. Install Python, then run this file again."
  read -r -p "Press Return to close..."
  exit 1
fi

VENV_PYTHON=".venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Creating the project's Python environment with $PYTHON..."
  if ! "$PYTHON" -m venv .venv; then
    echo "Could not create .venv. Install Python 3.10 or newer and try again."
    read -r -p "Press Return to close..."
    exit 1
  fi
fi

if ! "$VENV_PYTHON" -c 'import numpy, PIL, sklearn, torch, open_clip' >/dev/null 2>&1; then
  echo "Installing the project dependencies. The first setup may take several minutes..."
  if ! "$VENV_PYTHON" -m pip install -r requirements.txt; then
    echo "Dependency installation failed. Check your internet connection and Python version, then run this file again."
    read -r -p "Press Return to close..."
    exit 1
  fi
fi

"$VENV_PYTHON" app/server.py &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT INT TERM

for _ in {1..90}; do
  if curl --silent --fail "$URL" >/dev/null; then
    open "$URL"
    echo "GE2340 web app is running at $URL"
    echo "Keep this window open while using the app. Press Ctrl+C to stop the server."
    wait "$SERVER_PID"
    exit 0
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "The server stopped before it was ready. Check that dependencies are installed."
    read -r -p "Press Return to close..."
    exit 1
  fi
  sleep 1
done

echo "The server did not start within 90 seconds."
read -r -p "Press Return to close..."
exit 1
