#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print colored output
print_status() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Function to check if command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        print_success "✓ $1 completed successfully"
    else
        print_error "✗ $1 failed"
        exit 1
    fi
}

# Prompt for deployment type
echo -e "${BLUE}Select deployment type:${NC}"
echo "1) Domain (for production)"
echo "2) Localhost (for development)"
read -p "Enter choice (1 or 2): " DEPLOY_TYPE

case $DEPLOY_TYPE in
    1)
        echo -e "${BLUE}Enter your domain name (e.g. mrz.example.com):${NC}"
        read DOMAIN_NAME
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

# Update system
print_status "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y
check_status "System update"

# Install required packages
print_status "Installing required packages..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    tesseract-ocr \
    nginx \
    certbot \
    python3-certbot-nginx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git
check_status "Package installation"

# Create project directory and set proper ownership
print_status "Setting up project directory..."
sudo mkdir -p /opt/projects
sudo chown -R $USER:$USER /opt/projects
cd /opt/projects
check_status "Project directory setup"

# Clone the repository
print_status "Cloning FastMRZ repository..."
if [ -d ".git" ]; then
    print_warning "Repository already exists, pulling latest changes..."
    git pull origin main
else
    git clone https://github.com/rahmadmartin/fastmrz.git .
fi
check_status "Repository clone/update"

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    print_warning "Removing existing virtual environment..."
    rm -rf venv
fi

# Set up Python virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
check_status "Virtual environment creation"

# Install Python requirements
print_status "Installing Python packages..."
pip install --upgrade pip
check_status "Pip upgrade"

# Install requirements with proper opencv version for headless servers
print_status "Installing OpenCV (headless version for servers)..."
pip install opencv-python-headless>=4.9.0.80
check_status "OpenCV installation"

print_status "Installing other requirements..."
pip install pytesseract>=0.3.10
pip install numpy>=2.0
check_status "Basic requirements installation"

print_status "Installing FastAPI and Uvicorn..."
pip install fastapi uvicorn[standard]
check_status "FastAPI and Uvicorn installation"

print_status "Installing FastMRZ package..."
pip install -e .
check_status "FastMRZ package installation"

# Download MRZ trained data
print_status "Downloading MRZ trained data..."
sudo mkdir -p /usr/share/tesseract-ocr/5/tessdata/
sudo wget https://github.com/rahmadmartin/fastmrz/raw/refs/heads/main/tessdata/mrz.traineddata \
    -O /usr/share/tesseract-ocr/5/tessdata/mrz.traineddata
check_status "MRZ trained data download"

# Test the installation before proceeding
print_status "Testing FastMRZ installation..."
python -c "import fastmrz; print('FastMRZ imported successfully')" || {
    print_error "FastMRZ import test failed"
    exit 1
}

python -c "from fastmrz import mrz_api; print('API module imported successfully')" || {
    print_error "API module import test failed"
    exit 1
}
check_status "FastMRZ installation test"

# Create Nginx configuration
print_status "Configuring Nginx..."
if [ "$HOST_TYPE" = "domain" ]; then
    sudo tee /etc/nginx/sites-available/fastmrz << EOF
server {
    server_name ${DOMAIN_NAME};
    
    location / {
        proxy_pass http://localhost:8000;
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
    check_status "Nginx configuration"
    
    print_status "Setting up SSL certificate..."
    sudo certbot --nginx -d $DOMAIN_NAME --non-interactive --agree-tos --email admin@$DOMAIN_NAME
    check_status "SSL certificate setup"
else
    sudo tee /etc/nginx/sites-available/fastmrz << EOF
server {
    listen 80;
    server_name localhost;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF
    sudo ln -sf /etc/nginx/sites-available/fastmrz /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t && sudo systemctl reload nginx
    check_status "Nginx configuration"
fi

# Create systemd service with correct paths
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/fastmrz.service << EOF
[Unit]
Description=FastMRZ API Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/projects
Environment="PATH=/opt/projects/venv/bin"
Environment="PYTHONPATH=/opt/projects"
Environment="TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/"
ExecStart=/opt/projects/venv/bin/uvicorn fastmrz.mrz_api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
check_status "Systemd service creation"

# Start and enable service
print_status "Starting FastMRZ service..."
sudo systemctl daemon-reload
sudo systemctl enable fastmrz
sudo systemctl start fastmrz
check_status "Service start"

# Wait a moment for service to start
sleep 3

# Check service status
print_status "Checking service status..."
if sudo systemctl is-active --quiet fastmrz; then
    print_success "✓ FastMRZ service is running"
else
    print_error "✗ FastMRZ service failed to start"
    print_warning "Checking service logs:"
    sudo journalctl -u fastmrz -n 10 --no-pager
    exit 1
fi

# Test the API
print_status "Testing API..."
sleep 2
if curl -s http://localhost:8000/health > /dev/null; then
    print_success "✓ API health check passed"
else
    print_warning "⚠ API health check failed, but service might still be starting"
fi

# Final steps
print_success "=== Deployment completed successfully! ==="
if [ "$HOST_TYPE" = "domain" ]; then
    print_success "Your FastMRZ API is now available at: https://$DOMAIN_NAME"
    print_success "API Documentation: https://$DOMAIN_NAME/docs"
    print_success "Health Check: https://$DOMAIN_NAME/health"
else
    print_success "Your FastMRZ API is now available at: http://localhost"
    print_success "API Documentation: http://localhost/docs"  
    print_success "Health Check: http://localhost/health"
    print_success "Direct API: http://localhost:8000"
fi

echo ""
print_status "=== Management Commands ==="
echo -e "${BLUE}To check status:${NC} sudo systemctl status fastmrz"
echo -e "${BLUE}To view logs:${NC} sudo journalctl -u fastmrz -f"
echo -e "${BLUE}To restart service:${NC} sudo systemctl restart fastmrz"
echo -e "${BLUE}To stop service:${NC} sudo systemctl stop fastmrz"
echo -e "${BLUE}To test the API:${NC} curl http://localhost:8000/health"

print_success "FastMRZ deployment completed successfully!"