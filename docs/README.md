# STARFlow Video Gallery Setup Guide

This guide explains how to set up and maintain the STARFlow video results gallery.

## 📁 Directory Structure

```
docs/
├── index.html              # Main gallery page
├── _config.yml            # Jekyll configuration
├── assets/
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── script.js      # Interactive functionality
└── videos/                # Video files and thumbnails
    ├── thumbnails/        # Video thumbnail images
    ├── text-to-video/     # T2V generation results
    ├── image-to-video/    # I2V generation results
    ├── long-videos/       # Extended length videos
    └── comparisons/       # Method comparisons
```

## 🚀 Getting Started

### 1. Enable GitHub Pages

1. Go to your repository settings
2. Navigate to "Pages" section
3. Select source: "Deploy from a branch"
4. Choose branch: `main`
5. Choose folder: `/docs`
6. Save the settings

Your site will be available at: `https://apple.github.io/ml-starflow`

### 2. Add Your Videos

1. **Generate videos** using STARFlow-V:
   ```bash
   # Text-to-video
   bash scripts/test_sample_video.sh "your prompt here"

   # Longer videos
   bash scripts/test_sample_video.sh "your prompt here" "none" 241
   ```

2. **Create thumbnails** for each video:
   ```bash
   # Extract thumbnail at 1 second mark
   ffmpeg -i your_video.mp4 -ss 00:00:01 -vframes 1 -q:v 2 thumbnail.jpg
   ```

3. **Organize files**:
   - Videos: `docs/videos/[category]/video_name.mp4`
   - Thumbnails: `docs/videos/thumbnails/video_name_thumbnail.jpg`

### 3. Update Content

Edit `docs/index.html` to add your videos:

```html
<div class="video-item">
    <div class="video-container">
        <video controls poster="videos/thumbnails/your_video_thumbnail.jpg">
            <source src="videos/text-to-video/your_video.mp4" type="video/mp4">
        </video>
    </div>
    <div class="video-info">
        <h3>"Your prompt text here"</h3>
        <p class="video-details">480p • 16fps • 5s • CFG 3.5</p>
    </div>
</div>
```

## 📊 Content Categories

### Text-to-Video (`#text-to-video`)
- Pure text-to-video generation
- Showcase variety of prompts
- Standard 5-second clips

### Image-to-Video (`#image-to-video`)
- Image conditioning examples
- Show input image + generated video
- Demonstrate temporal consistency

### Long Videos (`#long-videos`)
- Extended length generations (15s, 30s+)
- Show `--target_length` capability
- Use Jacobi sampling

### Comparisons (`#comparisons`)
- Side-by-side with other methods
- Same prompt, different models
- Highlight STARFlow-V advantages

## 🎨 Video Guidelines

### Technical Specifications
- **Resolution**: 640×480 (480p) or higher
- **Format**: MP4 (H.264)
- **Frame Rate**: 16 FPS (standard) or 8-24 FPS
- **Duration**: 5s (standard), up to 30s+ for long videos
- **File Size**: Aim for <10MB per video for web performance

### Content Guidelines
- **Diverse Prompts**: Show variety in scenes, objects, actions
- **Quality Focus**: Only include high-quality generations
- **Representative**: Show typical results, not cherry-picked
- **Ethical**: Avoid harmful, biased, or inappropriate content

### Thumbnail Creation
```bash
# Create consistent thumbnails
ffmpeg -i input.mp4 -ss 00:00:01 -vframes 1 -vf "scale=640:360" thumbnail.jpg
```

## 🔧 Advanced Features

### Analytics Integration

Add Google Analytics to `_config.yml`:
```yaml
google_analytics: G-XXXXXXXXXX
```

### Video Quality Options

To add multiple quality options, create additional video files:
- `video_name.mp4` (480p)
- `video_name_720p.mp4` (720p)
- `video_name_1080p.mp4` (1080p)

### Custom Sections

Add new sections by:
1. Adding a new tab button in HTML
2. Creating corresponding tab content
3. Updating JavaScript tab handling

## 📈 Performance Optimization

### Video Compression
```bash
# Optimize for web
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k output.mp4

# Further compression if needed
ffmpeg -i input.mp4 -c:v libx264 -crf 28 -preset slow -c:a aac -b:a 96k output.mp4
```

### Lazy Loading
The gallery automatically implements lazy loading for better performance.

### CDN Integration
For large video collections, consider using a CDN:
1. Upload videos to AWS S3, Google Cloud, or similar
2. Update video `src` attributes to CDN URLs
3. Enable CORS for cross-origin video access

## 🐛 Troubleshooting

### Common Issues

**Videos not loading:**
- Check file paths are correct
- Ensure videos are in MP4 format
- Verify file permissions

**Thumbnails not showing:**
- Check thumbnail file exists
- Verify image format (JPG/PNG)
- Check file path in `poster` attribute

**GitHub Pages not updating:**
- Check GitHub Actions for build errors
- Ensure files are committed to main branch
- Wait 5-10 minutes for deployment

**Mobile responsiveness issues:**
- Test on various screen sizes
- Check CSS media queries
- Verify video aspect ratios

### Debug Mode

Enable debug logging by adding to your browser console:
```javascript
localStorage.setItem('debug', 'true');
```

## 📝 Content Management

### Automated Updates

Create a script to auto-generate gallery content:

```python
# generate_gallery.py
import os
import json
from pathlib import Path

def scan_videos():
    videos = []
    video_dir = Path("docs/videos")

    for category in ["text-to-video", "image-to-video", "long-videos"]:
        category_path = video_dir / category
        if category_path.exists():
            for video_file in category_path.glob("*.mp4"):
                videos.append({
                    "category": category,
                    "filename": video_file.name,
                    "path": f"videos/{category}/{video_file.name}",
                    "thumbnail": f"videos/thumbnails/{video_file.stem}_thumbnail.jpg"
                })

    return videos

# Generate video list
videos = scan_videos()
with open("docs/videos.json", "w") as f:
    json.dump(videos, f, indent=2)
```

### Batch Processing

Process multiple videos at once:
```bash
#!/bin/bash
# batch_process.sh

for video in raw_videos/*.mp4; do
    filename=$(basename "$video" .mp4)

    # Optimize video
    ffmpeg -i "$video" -c:v libx264 -crf 23 -preset medium \
           "docs/videos/text-to-video/${filename}.mp4"

    # Create thumbnail
    ffmpeg -i "$video" -ss 00:00:01 -vframes 1 -vf "scale=640:360" \
           "docs/videos/thumbnails/${filename}_thumbnail.jpg"
done
```

## 🔒 Security & Privacy

- **No personal data**: Avoid videos with identifiable people
- **Copyright**: Only use content you have rights to
- **Content policy**: Follow GitHub's community guidelines
- **Access control**: Repository must be public for GitHub Pages

## 📞 Support

For issues with the gallery:
1. Check GitHub Actions logs for build errors
2. Validate HTML/CSS/JS syntax
3. Test locally before pushing changes
4. Create issues in the repository for bugs

---

**Pro Tip**: Start with a few high-quality videos and gradually expand your gallery. Focus on showcasing the unique capabilities of STARFlow-V!