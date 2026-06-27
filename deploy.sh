#!/bin/bash
set -e

# Configuration
DROPLET="root@168.144.95.207"
DEST_DIR="/var/www/enginewatch/"

echo "🚀 Starting deployment to EngineWatch droplet..."

# Step 1: Build the frontend locally
echo "📦 Building the React frontend..."
cd frontend
npm run build

# Step 2: Push the built files to the droplet using rsync
echo "📤 Transferring files to $DROPLET..."
# Using rsync is safer and faster than pulling and building on a 2GB droplet
rsync -avz --delete dist/ $DROPLET:$DEST_DIR

# Step 3: Reload Caddy on the remote server
echo "🔄 Reloading Caddy web server..."
ssh $DROPLET "systemctl reload caddy"

echo "✅ Deployment complete! Check your live service at https://enginewatch.tech"
