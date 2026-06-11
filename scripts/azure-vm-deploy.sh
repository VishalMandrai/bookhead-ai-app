#!/usr/bin/env bash
set -euo pipefail

# Shell Local helper Script: Connect to Azure VM and deploy BookHead App 
# To Run this script:
# 1. Open Bash Terminal
# 2. Move to file directory and run these commands:
#       set -a
#       source scripts/deploy.env
#       set +a
#       bash scripts/azure-vm-deploy.sh

# NOTE: deploy.env must have all the secrets

VM_HOST="${VM_HOST:?Please set VM_HOST}"
VM_USER="${VM_USER:?Please set VM_USER}"
SSH_KEY_PATH="${SSH_KEY_PATH: ?Please set SSH Key Path}"
ACR_LOGIN_SERVER="${ACR_LOGIN_SERVER:?Please set ACR_LOGIN_SERVER}"
ACR_USERNAME="${ACR_USERNAME:?Please set ACR_USERNAME}"
ACR_PASSWORD="${ACR_PASSWORD:?Please set ACR_PASSWORD}"
REMOTE_DIR="${REMOTE_DIR:-/home/$VM_USER/bookhead}"

# Copy necessary files and folders to VM Host like:
# 1. models folder        - carrying all serialised models
# 2. docker-compose file  - edit it to use image from docker instead of building it
# 3. .env.prod            - fill in with necessary variables like Google API key etc.
echo "Copying necessary folders and files to Azure VM: $VM_HOST"
scp -i "$SSH_KEY_PATH" -r ./models "$VM_USER@$VM_HOST:$REMOTE_DIR"
scp -i "$SSH_KEY_PATH" docker-compose.yml "$VM_USER@$VM_HOST:$REMOTE_DIR"
scp -i "$SSH_KEY_PATH" .env.example "$VM_USER@$VM_HOST:$REMOTE_DIR"


echo "Running deployment commands on Azure VM"
ssh -i "$SSH_KEY_PATH" "$VM_USER@$VM_HOST" bash <<EOF
  set -euo pipefail
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose-plugin
  sudo systemctl enable --now docker
  cd "$REMOTE_DIR"

  mkdir uploads
  mkdir redis_data
  mkdir qdrant_storage

  echo "$ACR_PASSWORD" | sudo docker login $ACR_LOGIN_SERVER --username $ACR_USERNAME --password-stdin
  sudo docker compose pull
  sudo docker compose up -d --build
EOF

echo "Deployment complete."


