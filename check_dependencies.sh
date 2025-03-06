#!/bin/bash
#
# Dependency Checker Script for CNN-CIFAR10
# This script checks if all required Python dependencies are installed
# and offers to install missing ones.
#

# Color setup
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Banner
echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}     CNN-CIFAR10 Dependency Checker${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed. Please install pip first.${NC}"
    exit 1
fi

# Read requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found.${NC}"
    exit 1
fi

MISSING_PACKAGES=()
TOTAL_PACKAGES=0
INSTALLED_PACKAGES=0

echo -e "${YELLOW}Checking dependencies...${NC}"
echo ""

# Check each package
while IFS= read -r line; do
    # Skip comments and empty lines
    if [[ ${line:0:1} == "#" || -z "$line" ]]; then
        continue
    fi
    
    # Extract package name without version specifier
    package_name=$(echo "$line" | sed -E 's/([a-zA-Z0-9_\-\.]+).*/\1/')
    
    TOTAL_PACKAGES=$((TOTAL_PACKAGES+1))
    
    # Check if package is installed
    if python -c "import ${package_name//-/_}" 2>/dev/null; then
        echo -e "  [${GREEN}✓${NC}] ${package_name}"
        INSTALLED_PACKAGES=$((INSTALLED_PACKAGES+1))
    else
        echo -e "  [${RED}✗${NC}] ${package_name}"
        MISSING_PACKAGES+=("$line")
    fi
done < "requirements.txt"

echo ""
echo -e "${GREEN}${INSTALLED_PACKAGES}/${TOTAL_PACKAGES}${NC} dependencies installed."

# Handle missing packages
if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    echo -e "${GREEN}All dependencies are installed!${NC}"
    exit 0
else
    echo -e "${YELLOW}${#MISSING_PACKAGES[@]} dependencies are missing.${NC}"
    echo ""
    
    read -p "Would you like to install missing dependencies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Installing missing dependencies...${NC}"
        for package in "${MISSING_PACKAGES[@]}"; do
            echo -e "Installing ${package}..."
            pip install "$package"
        done
        echo -e "${GREEN}All dependencies installed successfully.${NC}"
    else
        echo -e "${YELLOW}Skipping installation. You can install dependencies later with:"
        echo -e "  pip install -e .${NC}"
    fi
fi 