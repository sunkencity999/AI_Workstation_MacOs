#!/bin/bash
#
# AI Workstation Installation Script
# Author: Christopher Bradford (contact@christopherdanielbradford.com)
# 
# This script automates the installation and setup of the AI Workstation,
# including all necessary dependencies, environment configuration, and
# model downloads. It includes comprehensive error handling and logging
# to ensure a smooth installation process.
#
# Features:
# - Automated installation of system dependencies (Homebrew, Python, pip)
# - Installation and configuration of Ollama for LLM support
# - Setup of nmail for email integration
# - Python virtual environment creation and dependency management
# - Comprehensive error handling and logging
# - Graceful cleanup on interruption or failure
#
# Usage: ./install.sh
#

# Enable strict error handling
set -e  # Exit on error
set -u  # Exit on undefined variable

# Define global variables for state management
OLLAMA_PID=""          # Store Ollama server PID for cleanup
VENV_ACTIVATED=0       # Track virtual environment activation
LOG_FILE="install_log.txt"  # Log file for installation progress
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # Script directory

# Cleanup function to ensure graceful exit
# This runs on script exit (success or failure) and handles:
# - Stopping Ollama server if we started it
# - Deactivating virtual environment
# - Logging final status
cleanup() {
    local exit_code=$?
    echo "Performing cleanup..."
    
    # Kill Ollama server if we started it
    if [ -n "$OLLAMA_PID" ]; then
        echo "Stopping Ollama server..."
        kill $OLLAMA_PID 2>/dev/null || true
    fi
    
    # Deactivate virtual environment if we activated it
    if [ $VENV_ACTIVATED -eq 1 ]; then
        echo "Deactivating virtual environment..."
        deactivate 2>/dev/null || true
    fi
    
    # Log exit status
    if [ $exit_code -ne 0 ]; then
        echo "Installation failed with exit code $exit_code. Check $LOG_FILE for details."
        echo "You can resume the installation by running: ./install.sh"
    fi
    
    exit $exit_code
}

# Set up trap for cleanup on exit and errors
trap cleanup EXIT INT TERM

# Logging function for consistent output and logging
log() {
    echo "$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Error handling function with line number reporting
handle_error() {
    log "Error occurred in line $1"
    exit 1
}
trap 'handle_error ${LINENO}' ERR

# Initial setup and OS check
echo "=== AI Workstation Installation Script ===" | tee -a "$LOG_FILE"
echo "This script will guide you through the installation process." | tee -a "$LOG_FILE"
echo "Installation log will be saved to: $LOG_FILE" | tee -a "$LOG_FILE"

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    log "Error: This installation script is designed for macOS."
    exit 1
fi

# Utility Functions
###################

# Check if a command exists in PATH
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verify if a Homebrew package is installed
check_brew_package() {
    if brew list "$1" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Core Installation Functions
############################

# Install or update Homebrew
install_homebrew() {
    if ! command_exists brew; then
        log "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
            log "Failed to install Homebrew. Please install it manually from https://brew.sh"
            exit 1
        }
        # Add Homebrew to PATH if needed
        if [[ ":$PATH:" != *":/opt/homebrew/bin:"* ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    else
        log "Homebrew is already installed."
        # Update Homebrew
        log "Updating Homebrew..."
        brew update || log "Warning: Failed to update Homebrew. Continuing anyway..."
    fi
}

# Install Python if needed
install_python() {
    if ! command_exists python3; then
        log "Installing Python..."
        brew install python || {
            log "Failed to install Python"
            exit 1
        }
    else
        log "Python is already installed: $(python3 --version)"
    fi
}

# Ensure pip is installed and updated
install_pip() {
    if ! command_exists pip3; then
        log "Installing pip..."
        python3 -m ensurepip --upgrade || {
            log "Failed to install pip"
            exit 1
        }
    else
        log "pip is already installed: $(pip3 --version)"
        log "Upgrading pip..."
        python3 -m pip install --upgrade pip
    fi
}

# Set up Python virtual environment
setup_venv() {
    log "Setting up virtual environment..."
    if [ -d "venv" ]; then
        log "Found existing virtual environment. Backing up..."
        mv venv "venv_backup_$(date +%Y%m%d_%H%M%S)"
    fi
    
    python3 -m venv venv || {
        log "Failed to create virtual environment"
        exit 1
    }
    
    log "Activating virtual environment..."
    source venv/bin/activate || {
        log "Failed to activate virtual environment"
        exit 1
    }
    VENV_ACTIVATED=1
    
    # Verify activation
    if [[ "$VIRTUAL_ENV" != *"venv"* ]]; then
        log "Virtual environment activation verification failed"
        exit 1
    }
}

# Install Ollama LLM framework
install_ollama() {
    if ! check_brew_package ollama; then
        log "Installing Ollama..."
        brew install ollama || {
            log "Failed to install Ollama"
            exit 1
        }
    else
        log "Ollama is already installed. Checking for updates..."
        brew upgrade ollama || log "Warning: Failed to upgrade Ollama. Continuing anyway..."
    fi
}

# Install nmail email client
install_nmail() {
    if ! check_brew_package nmail; then
        log "Installing nmail..."
        brew install nmail || {
            log "Failed to install nmail"
            exit 1
        }
    else
        log "nmail is already installed. Checking for updates..."
        brew upgrade nmail || log "Warning: Failed to upgrade nmail. Continuing anyway..."
    fi
}

# Create or verify requirements.txt
create_requirements() {
    if [ ! -f "requirements.txt" ]; then
        log "Creating requirements.txt..."
        cat > requirements.txt << EOL || {
            log "Failed to create requirements.txt"
            exit 1
        }
streamlit==1.29.0
requests==2.31.0
python-dotenv==1.0.0
configparser==6.0.0
EOL
    else
        log "requirements.txt already exists"
    fi
}

# Download and verify Ollama model with retries
verify_ollama_model() {
    local model=$1
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log "Attempting to download $model (attempt $attempt of $max_attempts)..."
        if ollama pull $model; then
            log "Successfully downloaded $model"
            return 0
        fi
        attempt=$((attempt + 1))
        if [ $attempt -le $max_attempts ]; then
            log "Download failed. Retrying in 5 seconds..."
            sleep 5
        fi
    done
    
    log "Failed to download $model after $max_attempts attempts"
    return 1
}

# Main Installation Process
##########################

log "Starting installation process..."

# Step 1: Install system dependencies
log "Step 1: Installing system dependencies..."
install_homebrew
install_python
install_pip
install_ollama
install_nmail

# Step 2: Set up Python environment
log "Step 2: Setting up Python environment..."
setup_venv
create_requirements

# Step 3: Install Python dependencies
log "Step 3: Installing Python dependencies..."
pip3 install -r requirements.txt || {
    log "Failed to install Python dependencies"
    exit 1
}

# Step 4: Initialize Ollama and download required models
log "Step 4: Setting up Ollama..."
if ! pgrep -x "ollama" >/dev/null; then
    log "Starting Ollama server..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    log "Waiting for Ollama server to start..."
    sleep 5
    
    # Verify server is running
    if ! curl -s http://localhost:11434/api/version >/dev/null; then
        log "Failed to start Ollama server"
        exit 1
    fi
else
    log "Ollama server is already running"
fi

log "Downloading required model (this may take a while)..."
verify_ollama_model "llama3.2-vision:latest" || {
    log "Failed to download required model"
    exit 1
}

# Step 5: Create necessary directories
log "Step 5: Creating necessary directories..."
mkdir -p logs || {
    log "Failed to create logs directory"
    exit 1
}
mkdir -p static || {
    log "Failed to create static directory"
    exit 1
}

# Step 6: Set up environment variables
log "Step 6: Setting up environment variables..."
if [ ! -f ".env" ]; then
    cat > .env << EOL || {
        log "Failed to create .env file"
        exit 1
    }
OLLAMA_HOST=http://localhost:11434
LOG_LEVEL=INFO
EOL
    log "Created .env file"
else
    log ".env file already exists"
fi

# Step 7: Configure nmail
log "Step 7: Email integration setup instructions"
cat << EOL

=== Email Integration Setup ===
Please follow these steps to configure nmail:

1. Enable 2-Factor Authentication in your Google Account:
   - Go to your Google Account settings
   - Navigate to Security → 2-Step Verification
   - Follow the prompts to enable 2FA

2. Generate an App Password:
   - Go to your Google Account settings
   - Navigate to Security → 2-Step Verification
   - Scroll to "App passwords" at the bottom
   - Select "Mail" as the app and your device type
   - Copy the 16-character password generated

3. Configure nmail:
   - Run: nmail -s gmail
   - Enter your Gmail address when prompted
   - Enter the App Password (NOT your regular Gmail password)
   - Enter your display name

EOL

# Final Message
cat << EOL

Thank you for installing the AI Workstation!

This project was created with ❤️ by Christopher Bradford
For support or feedback, please contact:
Email: contact@christopherdanielbradford.com

Your installation log has been saved to: $LOG_FILE
EOL
