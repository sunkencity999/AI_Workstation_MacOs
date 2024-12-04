#!/bin/bash

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$DIR"

# Activate virtual environment
source venv/bin/activate

# Make sure Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Launch the interface
echo "Starting Shell GPT Interface..."
streamlit run sgpt_interface.py --server.port 8527 --server.headless true

# Keep the terminal window open if there's an error
read -p "Press Enter to close..."