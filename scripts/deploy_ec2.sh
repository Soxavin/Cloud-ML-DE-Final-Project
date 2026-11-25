#!/usr/bin/env bash
# Sets up our two services on a brand-new Ubuntu 24.04 EC2 instance in one go.
#
# Why this exists: the AWS Academy Data Engineering sandbox deletes every instance when the
# session ends, so we redeploy from scratch each time. Run it right after you SSH in:
#
#   curl -fsSL https://raw.githubusercontent.com/Soxavin/Cloud-ML-DE-Final-Project/main/scripts/deploy_ec2.sh | bash
#
# Nothing is built on the server. Both images are pulled ready-made from Docker Hub.

set -euo pipefail

REPO_RAW="https://raw.githubusercontent.com/Soxavin/Cloud-ML-DE-Final-Project/main"
APP_DIR="$HOME/bank-app"

echo "==> Installing Docker (takes a minute or two)"
if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sudo sh
fi
sudo systemctl enable --now docker

echo "==> Fetching docker-compose.yml"
mkdir -p "$APP_DIR"
cd "$APP_DIR"
curl -fsSL "$REPO_RAW/docker-compose.yml" -o docker-compose.yml

# sudo here so we don't have to log out and back in after joining the docker group
echo "==> Pulling the images and starting both services"
sudo docker compose pull
sudo docker compose up -d

# The model service needs a few seconds to load the model, and the backend waits for it,
# so give them up to two minutes before calling it a failure.
echo "==> Waiting for the backend to report healthy"
for _ in $(seq 1 24); do
    if curl -fsS http://localhost:8000/health; then
        echo
        public_ip=$(curl -fsS https://checkip.amazonaws.com || echo "<public-ip>")
        echo "==> All good. Put this in Postman's base_url: http://${public_ip}:8000"
        exit 0
    fi
    sleep 5
done

echo "==> The backend never came up healthy. Here's what Docker says:"
sudo docker compose ps
sudo docker compose logs --tail 40
exit 1
