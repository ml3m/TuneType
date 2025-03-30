#!/bin/bash

# Color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  Music Genre Classification Setup     ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Check Python installation
echo -e "\n${YELLOW}Checking Python installation...${NC}"
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
    echo -e "${GREEN}Python 3 is installed.${NC}"
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
    echo -e "${GREEN}Python is installed.${NC}"
else
    echo -e "${RED}Python is not installed. Please install Python 3.x to continue.${NC}"
    exit 1
fi

# Check Python version
PY_VERSION=$($PYTHON_CMD -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
echo -e "Python version: ${GREEN}$PY_VERSION${NC}"

# Check for virtual environment
echo -e "\n${YELLOW}Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Please install the venv module.${NC}"
        echo -e "You can install it with: ${YELLOW}pip install virtualenv${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source venv/bin/activate
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    echo -e "${RED}Unsupported operating system. Please activate the virtual environment manually.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "\n${YELLOW}Creating necessary directories...${NC}"
mkdir -p tmp/uploads
mkdir -p app/models
mkdir -p samples
mkdir -p output

# Install required packages
echo -e "\n${YELLOW}Installing required packages...${NC}"
if [ -f "requirements.txt" ]; then
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install required packages. Please check requirements.txt.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Required packages installed successfully.${NC}"
else
    echo -e "${RED}requirements.txt not found. Please ensure the file exists.${NC}"
    exit 1
fi

# Initialize the model
echo -e "\n${YELLOW}Setting up the genre classification model...${NC}"
echo -e "This may take a few minutes depending on your internet connection and machine."

# Ask if user wants to build the model with downloaded samples
echo -e "\n${YELLOW}Do you want to download music samples and train the model? (y/n)${NC}"
echo -e "This is recommended for first-time setup and will improve model accuracy."
read -p "> " download_choice

if [[ $download_choice =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Building the model with downloaded samples...${NC}"
    python build_model.py 
    if [ $? -ne 0 ]; then
        echo -e "${RED}Model building failed. Please check the logs for details.${NC}"
        echo -e "You can retry later with: ${YELLOW}python build_model.py${NC}"
    else
        echo -e "${GREEN}Model built and trained successfully!${NC}"
    fi
else
    echo -e "\n${YELLOW}Skipping automatic sample download and training.${NC}"
    echo -e "You can manually train the model later with: ${YELLOW}python build_model.py${NC}"
fi

# Instructions for running the application
echo -e "\n${BLUE}=======================================${NC}"
echo -e "${BLUE}      Setup Complete!                  ${NC}"
echo -e "${BLUE}=======================================${NC}"

echo -e "\n${GREEN}To run the web application:${NC}"
echo -e "  ${YELLOW}python app.py${NC}"

echo -e "\n${GREEN}To test the model on a single file:${NC}"
echo -e "  ${YELLOW}python test_model.py path/to/your/song.mp3${NC}"

echo -e "\n${GREEN}To improve model accuracy:${NC}"
echo -e "  1. Organize more MP3 samples into genre folders under the 'samples' directory"
echo -e "  2. Run the training script: ${YELLOW}python train_with_samples.py${NC}"

echo -e "\n${GREEN}For more options:${NC}"
echo -e "  ${YELLOW}python build_model.py --help${NC}"

echo -e "\n${BLUE}Enjoy classifying music genres!${NC}" 