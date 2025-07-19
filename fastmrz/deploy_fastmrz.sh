#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

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
            echo -e "${RED}Domain name cannot be empty${NC}"
            exit 1
        fi
        HOST_TYPE="domain"
        ;;
    2)
        DOMAIN_NAME="localhost"
        HOST_TYPE="local"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}Starting FastMRZ deployment for: $DOMAIN_NAME${NC}"

# Update system
echo -e "${BLUE}Updating system packages...${NC}"
sudo apt-get update && sudo apt-get upgrade -y

# Install required packages
echo -e "${BLUE}Installing required packages...${NC}"
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    tesseract-ocr \
    nginx \
    certbot \
    python3-certbot-nginx \
    libgl1-mesa-glx \
    wget

# Create project directory
echo -e "${BLUE}Setting up project directory...${NC}"
sudo mkdir -p /opt/fastmrz
cd /opt/fastmrz

# Set up Python virtual environment
echo -e "${BLUE}Creating Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Install Python requirements
echo -e "${BLUE}Installing Python packages...${NC}"
pip install fastmrz

# Download MRZ trained data
echo -e "${BLUE}Downloading MRZ trained data...${NC}"
sudo mkdir -p /usr/share/tesseract-ocr/4.00/tessdata/
sudo wget https://github.com/rahmadmartin/fastmrz/raw/main/tessdata/mrz.traineddata \
    -O /usr/share/tesseract-ocr/4.00/tessdata/mrz.traineddata

# Create Nginx configuration
echo -e "${BLUE}Configuring Nginx...${NC}"
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
    # Enable site and set up SSL
    sudo ln -s /etc/nginx/sites-available/fastmrz /etc/nginx/sites-enabled/
    sudo nginx -t && sudo systemctl reload nginx
    sudo certbot --nginx -d $DOMAIN_NAME --non-interactive --agree-tos --email admin@$DOMAIN_NAME
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
    sudo ln -s /etc/nginx/sites-available/fastmrz /etc/nginx/sites-enabled/
    sudo nginx -t && sudo systemctl reload nginx
fi

# Create systemd service
echo -e "${BLUE}Creating systemd service...${NC}"
sudo tee /etc/systemd/system/fastmrz.service << EOF
[Unit]
Description=FastMRZ API Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/fastmrz
Environment="PATH=/opt/fastmrz/venv/bin"
ExecStart=/opt/fastmrz/venv/bin/uvicorn fastmrz.mrz_api:app --host 0.0.0.0 --port 8000

[Install]
WantedBy=multi-user.target
EOF

# Start and enable service
sudo systemctl daemon-reload
sudo systemctl enable fastmrz
sudo systemctl start fastmrz

# Set up SSL
echo -e "${BLUE}Setting up SSL certificate...${NC}"
sudo certbot --nginx -d $DOMAIN_NAME --non-interactive --agree-tos --email admin@$DOMAIN_NAME

# Final steps
echo -e "${GREEN}Deployment completed!${NC}"
f [ "$HOST_TYPE" = "domain" ]; then
    echo -e "${GREEN}Your FastMRZ API is now available at: https://$DOMAIN_NAME${NC}"
else
    echo -e "${GREEN}Your FastMRZ API is now available at: http://localhost${NC}"
fi

echo -e "${BLUE}To check status:${NC} sudo systemctl status fastmrz"
echo -e "${BLUE}To view logs:${NC} sudo journalctl -u fastmrz -f"