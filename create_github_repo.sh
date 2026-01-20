#!/bin/bash
# Script to create private GitHub repository and push code

# Check if GitHub token is available
if [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️  GITHUB_TOKEN environment variable not set."
    echo ""
    echo "To create the repository automatically, you need a GitHub Personal Access Token."
    echo "1. Go to: https://github.com/settings/tokens"
    echo "2. Generate a new token with 'repo' scope"
    echo "3. Run: export GITHUB_TOKEN=your_token_here"
    echo "4. Then run this script again"
    echo ""
    echo "Alternatively, create the repository manually:"
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: prismt"
    echo "3. Description: Unified Pipeline for Widefield and CDKL5 Data"
    echo "4. Select: Private"
    echo "5. DO NOT initialize with README (we already have one)"
    echo "6. Click 'Create repository'"
    echo "7. Then run: git push -u origin main"
    exit 1
fi

# Create private repository via GitHub API
echo "Creating private repository 'prismt' on GitHub..."
response=$(curl -s -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/user/repos \
    -d '{
        "name": "prismt",
        "description": "Unified Pipeline for Widefield and CDKL5 Data",
        "private": true
    }')

# Check if repository was created
if echo "$response" | grep -q '"name":"prismt"'; then
    echo "✅ Repository created successfully!"
    echo ""
    echo "Pushing code to GitHub..."
    git push -u origin main
else
    # Check if repository already exists
    if echo "$response" | grep -q "already exists"; then
        echo "⚠️  Repository already exists. Pushing code..."
        git push -u origin main
    else
        echo "❌ Failed to create repository:"
        echo "$response" | head -20
        echo ""
        echo "Please create the repository manually at: https://github.com/new"
    fi
fi
