#!/usr/bin/env bash
set -euo pipefail

VM_HOST="${VM_HOST:?Please set VM_HOST}"
VM_USER="${VM_USER:?Please set VM_USER}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
ACR_LOGIN_SERVER="${ACR_LOGIN_SERVER:?Please set ACR_LOGIN_SERVER}"
ACR_USERNAME="${ACR_USERNAME:?Please set ACR_USERNAME}"
ACR_PASSWORD="${ACR_PASSWORD:?Please set ACR_PASSWORD}"
REMOTE_DIR="${REMOTE_DIR:-/home/$VM_USER/bookhead-app}"

echo "Copying repository to Azure VM: $VM_HOST"
scp -i "$SSH_KEY_PATH" -r . "$VM_USER@$VM_HOST:$REMOTE_DIR"

echo "Running deployment commands on Azure VM"
ssh -i "$SSH_KEY_PATH" "$VM_USER@$VM_HOST" bash <<EOF
  set -euo pipefail
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose-plugin
  sudo systemctl enable --now docker
  mkdir -p "$REMOTE_DIR"
  cd "$REMOTE_DIR"
  echo "$ACR_PASSWORD" | sudo docker login $ACR_LOGIN_SERVER --username $ACR_USERNAME --password-stdin
  sudo docker compose pull
  sudo docker compose up -d --build
EOF

echo "Deployment complete."


# Shell Local helper Script: Connect to Azure VM and deploy BookHead App 
# To Run this script:
# 1. Open Git Bash
# 2. Move to file directory and run:
#       set +a
#       
#       set -a
#       bash azure-vm-deploy.sh