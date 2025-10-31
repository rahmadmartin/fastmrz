#!/bin/bash
set -e

# =========================================
# FastMRZ Deployment Script (renewed)
# =========================================

# --- Color Codes ---
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# --- Helper Functions ---
print_status() { echo -e "${BLUE}$1${NC}"; }
print_success() { echo -e "${GREEN}$1${NC}"; }
print_warning() { echo -e "${YELLOW}$1${NC}"; }
print_error() { echo -e "${RED}$1${NC}"; }

check_status() {
    if [ $? -eq 0 ]; then
        print_success "✓ $1 completed successfully"
    else
        print_error "✗ $1 failed"
        exit 1
    fi
}

# --- Deployment Type Prompt ---
echo -e "${BLUE}Select deployment type:${NC}"
echo "1) Domain (production)"
echo "2) Localhost (development)"
read -p "Enter choice (1 or 2): " DEPLOY_TYPE

case $DEPLOY_TYPE in
    1)
        read -p "Enter your domain name (e.g. mrz.example.com): " DOMAIN_NAME
        if [ -z "$DOMAIN_NAME" ]; then
            print_error "Domain name cannot be empty"
            exit 1
        fi
        HOST_TYPE="domain"
        ;;
    2)
        DOMAIN_NAME="localhost"
        HOST_TYPE="local"
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

print_success "Starting FastMRZ deployment for: $DOMAIN_NAME"

# --- System Update ---
print_status "Updating system packages..."
sudo apt-get update -y && sudo apt-get upgrade -y
check_status "System update"

# --- Install Required OS Packages ---
print_status "Installing required packages..."
sudo apt-get install -y \
    python3-pip python3-venv \
    tesseract-ocr tesseract-ocr-ind \
    nginx certbot python3-certbot-nginx \
    libgl1-mesa-dev libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    wget git python3-pil python3-pil.imagetk
check_status "System package installation"

# --- Project Directory Setup ---
print_status "Setting up project directory..."
sudo mkdir -p /opt/projects
sudo chown -R $USER:$USER /opt/projects
cd /opt/projects
check_status "Project directory setup"

# --- Repository ---
print_status "Fetching FastMRZ repository..."
if [ -d ".git" ]; then
    print_warning "Repository already exists — pulling latest changes..."
    git pull origin main
else
    git clone https://github.com/rahmadmartin/fastmrz.git .
fi
check_status "Repository clone/update"

# --- Python Virtual Environment ---
print_status "Creating Python virtual environment..."
if [ -d "venv" ]; then
    print_warning "Removing old virtual environment..."
    rm -rf venv
fi
python3 -m venv venv
source venv/bin/activate
check_status "Virtual environment setup"

# --- Python Packages ---
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel
check_status "Pip upgrade"

print_status "Installing core Python dependencies..."
pip install opencv-python-headless>=4.9.0.80 pytesseract>=0.3.10 numpy>=2.0
check_status "Core Python dependencies"

print_status "Installing FastAPI stack..."
pip install fastapi uvicorn[standard] python-multipart
check_status "FastAPI + Uvicorn installation"

print_status "Installing FastMRZ package..."
pip install -e .
check_status "FastMRZ package installation"

# --- Tesseract MRZ data ---
print_status "Installing MRZ trained data..."
sudo mkdir -p /usr/share/tesseract-ocr/5/tessdata/
sudo wget -q https://github.com/rahmadmartin/fastmrz/raw/main/tessdata/mrz.traineddata \
    -O /usr/share/tesseract-ocr/5/tessdata/mrz.traineddata
check_status "MRZ trained data download"

# --- Test Import ---
print_status "Testing FastMRZ installation..."
python -c "import fastmrz; from fastmrz import mrz_api; print('FastMRZ OK')" || {
    print_error "FastMRZ import test failed"
    exit 1
}
check_status "FastMRZ import test"

# --- Clean Old Service & Nginx ---
print_status "Cleaning up old deployment..."
sudo systemctl stop fastmrz.service 2>/dev/null || true
sudo systemctl disable fastmrz.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/fastmrz.service
sudo systemctl daemon-reload
sudo systemctl reset-failed
sudo rm -f /etc/nginx/sites-{available,enabled}/fastmrz
check_status "Old service cleanup"

# --- Configure Nginx ---
print_status "Creating Nginx config..."
if [ "$HOST_TYPE" = "domain" ]; then
    sudo tee /etc/nginx/sites-available/fastmrz >/dev/null <<EOF
server {
    server_name ${DOMAIN_NAME};
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
    sudo ln -sf /etc/nginx/sites-available/fastmrz /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t && sudo systemctl reload nginx
    check_status "Nginx setup"

    print_status "Setting up SSL..."
    sudo certbot --nginx -d $DOMAIN_NAME --non-interactive --agree-tos --email admin@$DOMAIN_NAME || \
        print_warning "SSL certificate setup skipped (check DNS)"
    check_status "SSL setup"
else
    sudo tee /etc/nginx/sites-available/fastmrz >/dev/null <<EOF
server {
    listen 80;
    server_name localhost;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF
    sudo ln -sf /etc/nginx/sites-available/fastmrz /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t && sudo systemctl reload nginx
    check_status "Nginx setup"
fi

# --- Systemd Service ---
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/fastmrz.service >/dev/null <<EOF
[Unit]
Description=FastMRZ API Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/projects
Environment="PATH=/opt/projects/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=/opt/projects"
Environment="TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/"
ExecStart=/opt/projects/venv/bin/uvicorn fastmrz.mrz_api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
check_status "Systemd service creation"

# --- Start Service ---
print_status "Starting FastMRZ service..."
sudo systemctl daemon-reload
sudo systemctl enable fastmrz
sudo systemctl restart fastmrz
sleep 3
sudo systemctl is-active --quiet fastmrz && print_success "✓ Service running" || {
    print_error "✗ Service failed to start"
    sudo journalctl -u fastmrz -n 20 --no-pager
    exit 1
}

# --- API Test ---
print_status "Testing API..."
if curl -s http://127.0.0.1:8000/health > /dev/null; then
    print_success "✓ API health check passed"
else
    print_warning "⚠ API health check failed — service may still be initializing"
fi

# --- Done ---
print_success "=== Deployment completed successfully! ==="
if [ "$HOST_TYPE" = "domain" ]; then
    echo -e "${GREEN}Your API: https://${DOMAIN_NAME}${NC}"
    echo -e "${GREEN}Docs:    https://${DOMAIN_NAME}/docs${NC}"
else
    echo -e "${GREEN}Your API: http://localhost:8000${NC}"
    echo -e "${GREEN}Docs:    http://localhost:8000/docs${NC}"
fi

print_status "Useful commands:"
echo -e "${BLUE}sudo systemctl status fastmrz${NC}"
echo -e "${BLUE}sudo journalctl -u fastmrz -f${NC}"
echo -e "${BLUE}sudo systemctl restart fastmrz${NC}"
echo -e "${BLUE}sudo systemctl stop fastmrz${NC}"
