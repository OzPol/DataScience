#!/bin/bash

# Function to prompt the user for their Python path
prompt_python_path() {
    read -p "Enter the path to your Python 3.11 executable (or leave blank to use the default '/c/Users/odpol/AppData/Local/Programs/Python/Python311/python.exe'): " user_input
    if [ ! -z "$user_input" ]; then
        PYTHON_CMD="$user_input"
    else
        PYTHON_CMD="/c/Users/odpol/AppData/Local/Programs/Python/Python311/python.exe"
    fi
}

# Set default to python3 command
PYTHON_CMD="/c/Users/odpol/AppData/Local/Programs/Python/Python311/python.exe"

# Ensure the correct Python version is used
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
REQUIRED_PYTHON="Python 3.11"

if [[ "$PYTHON_VERSION" == "$REQUIRED_PYTHON"* ]]; then
    echo "Correct Python version detected: $PYTHON_VERSION"
else
    echo "Incorrect Python version: $PYTHON_VERSION"
    echo "This project requires Python 3.11."
    prompt_python_path
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    if [[ "$PYTHON_VERSION" != "$REQUIRED_PYTHON"* ]]; then
        echo "Failed to set the correct Python version. Exiting setup."
        exit 1
    fi
fi

# Upgrade pip and install requirements
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

# Start Jupyter Notebook
$PYTHON_CMD -m jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
