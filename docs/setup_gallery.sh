#!/bin/bash

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

# STARFlow Video Gallery Setup Script
# This script helps you set up the video gallery with your generated content

set -e

echo "🎬 STARFlow Video Gallery Setup"
echo "================================"

# Check if we're in the right directory
if [ ! -f "transformer_flow.py" ]; then
    echo "❌ Error: Please run this script from the ml-starflow root directory"
    exit 1
fi

# Create video directories if they don't exist
echo "📁 Creating video directories..."
mkdir -p docs/videos/{text-to-video,image-to-video,long-videos,comparisons,thumbnails}

# Function to generate a video and thumbnail
generate_video_example() {
    local prompt="$1"
    local category="$2"
    local filename="$3"
    local target_length="${4:-81}"

    echo "🎥 Generating video: $prompt"

    # Generate video using STARFlow-V
    if [ "$category" = "text-to-video" ]; then
        bash scripts/test_sample_video.sh "$prompt" "none" "$target_length"
    else
        echo "⚠️  Manual generation required for category: $category"
        return
    fi

    # Find the generated video (this depends on your output structure)
    # You may need to adjust this path based on where sample.py saves files
    local generated_video=$(find . -name "*.mp4" -newer docs/videos/STRUCTURE.md | head -1)

    if [ -n "$generated_video" ] && [ -f "$generated_video" ]; then
        # Copy to gallery
        cp "$generated_video" "docs/videos/$category/$filename.mp4"

        # Generate thumbnail
        if command -v ffmpeg &> /dev/null; then
            ffmpeg -i "$generated_video" -ss 00:00:01 -vframes 1 -q:v 2 \
                   "docs/videos/thumbnails/${filename}_thumbnail.jpg" -y
            echo "✅ Created: $filename.mp4 and thumbnail"
        else
            echo "⚠️  FFmpeg not found. Please install FFmpeg to generate thumbnails."
        fi
    else
        echo "❌ Could not find generated video. Please check your output directory."
    fi
}

# Check if user wants to generate example videos
read -p "📥 Generate example videos? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Generating example videos..."

    # Check if checkpoints exist
    if [ ! -f "ckpts/starflow-v_7B_t2v_caus_480p_v3.pth" ]; then
        echo "⚠️  Checkpoint not found. Please ensure you have the required model checkpoints."
        echo "   Expected: ckpts/starflow-v_7B_t2v_caus_480p_v3.pth"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Generate example videos
    generate_video_example "A corgi dog looks at the camera" "text-to-video" "corgi_dog_camera"
    generate_video_example "A film still of a cat playing piano" "text-to-video" "cat_playing_piano"
    generate_video_example "Ocean waves crashing against rocks at sunset" "text-to-video" "ocean_waves_sunset"

    # Generate a longer video example
    generate_video_example "Walking through a peaceful forest path" "long-videos" "forest_walk_15s" 241

    echo "✅ Example videos generated!"
else
    echo "⏩ Skipping video generation. You can add videos manually later."
fi

# Set up GitHub Pages
echo "🌐 Setting up GitHub Pages..."

# Check if GitHub CLI is available
if command -v gh &> /dev/null; then
    echo "📡 GitHub CLI detected. Checking repository settings..."

    # Enable GitHub Pages (requires repo admin permissions)
    gh api repos/{owner}/{repo}/pages \
        --method POST \
        --field source[branch]=main \
        --field source[path]=/docs \
        2>/dev/null && echo "✅ GitHub Pages enabled" || echo "⚠️  Could not enable GitHub Pages automatically"
else
    echo "⚠️  GitHub CLI not found. Please enable GitHub Pages manually:"
    echo "   1. Go to your repository settings"
    echo "   2. Navigate to 'Pages' section"
    echo "   3. Select source: 'Deploy from a branch'"
    echo "   4. Choose branch: 'main' and folder: '/docs'"
fi

# Final instructions
echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Add your video files to docs/videos/ directories"
echo "2. Update docs/index.html with your video information"
echo "3. Commit and push your changes to GitHub"
echo "4. Your gallery will be available at: https://apple.github.io/ml-starflow"
echo ""
echo "📖 See docs/README.md for detailed instructions"
echo ""
echo "🔧 Useful commands:"
echo "   # Generate more videos:"
echo "   bash scripts/test_sample_video.sh \"your prompt\" \"none\" 81"
echo ""
echo "   # Create thumbnails:"
echo "   ffmpeg -i video.mp4 -ss 00:00:01 -vframes 1 thumbnail.jpg"
echo ""
echo "   # Test locally:"
echo "   cd docs && python -m http.server 8000"
echo "   # Then visit: http://localhost:8000"