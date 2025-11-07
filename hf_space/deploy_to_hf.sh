#!/bin/bash
#############################################
# Deploy Maya1 Gradio App to HF Spaces
# Usage: ./deploy_to_hf.sh
#############################################

set -e

echo "======================================================"
echo "Maya1 - Deploy to Hugging Face Spaces"
echo "======================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found!"
    echo "Please run this script from the hf_space directory"
    exit 1
fi

# Clone or update the HF Space
SPACE_DIR="../maya1-hf-space"

if [ -d "$SPACE_DIR" ]; then
    echo "📁 Space directory exists, pulling latest..."
    cd "$SPACE_DIR"
    git pull
    cd -
else
    echo "📥 Cloning HF Space..."
    echo ""
    echo "When prompted for password, use your HF access token:"
    echo "Generate one here: https://huggingface.co/settings/tokens"
    echo ""
    git clone https://huggingface.co/spaces/maya-research/maya1 "$SPACE_DIR"
fi

# Copy files
echo ""
echo "📋 Copying files to space..."
cp app.py "$SPACE_DIR/"
cp requirements.txt "$SPACE_DIR/"
cp .gitignore "$SPACE_DIR/" 2>/dev/null || true

echo "✅ Files copied"

# Commit and push
echo ""
echo "📤 Committing and pushing to HF Spaces..."
cd "$SPACE_DIR"
git add .
git commit -m "Update Maya1 Gradio app with preset characters" || echo "No changes to commit"
git push

echo ""
echo "======================================================"
echo "✅ Deployment complete!"
echo "======================================================"
echo ""
echo "Your space should be live at:"
echo "https://huggingface.co/spaces/maya-research/maya1"
echo ""
echo "It may take a few minutes to build and deploy."
echo ""

