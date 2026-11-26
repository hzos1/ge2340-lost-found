#!/bin/bash
cd "$(dirname "$0")"
URL="http://127.0.0.1:8765"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required. Install it, then run this file again."
  read -r -p "Press Return to close..."
  exit 1
fi

python3 app/server.py &
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
