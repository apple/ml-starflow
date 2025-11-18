# Example video files structure - replace with actual videos

# Text-to-Video Examples
docs/videos/text-to-video/
├── corgi_dog_camera.mp4              # "A corgi dog looks at the camera"
├── cat_playing_piano.mp4             # "A film still of a cat playing piano"
├── ocean_waves_sunset.mp4            # "Ocean waves crashing against rocks at sunset"
└── city_rain_night.mp4               # "A bustling city street with rain at night, neon lights reflecting"

# Image-to-Video Examples
docs/videos/image-to-video/
├── portrait_wind.mp4                 # Portrait + "Hair blowing in the wind"
└── landscape_clouds.mp4              # Landscape + "Clouds moving across the sky"

# Long Video Examples
docs/videos/long-videos/
├── forest_walk_15s.mp4               # "Walking through a peaceful forest path" (241 frames)
└── space_journey_30s.mp4             # "A journey through space with stars and nebulas" (481 frames)

# Method Comparisons
docs/videos/comparisons/
├── dancer_starflow.mp4               # STARFlow-V result
└── dancer_baseline.mp4               # Baseline method result

# Thumbnails (JPEG images)
docs/videos/thumbnails/
├── corgi_thumbnail.jpg
├── cat_piano_thumbnail.jpg
├── ocean_waves_thumbnail.jpg
├── city_rain_thumbnail.jpg
├── portrait_wind_thumbnail.jpg
├── landscape_clouds_thumbnail.jpg
├── forest_walk_long_thumbnail.jpg
├── space_journey_long_thumbnail.jpg
├── dancer_ours_thumbnail.jpg
└── dancer_baseline_thumbnail.jpg

# To add your own videos:
# 1. Generate videos using STARFlow-V scripts
# 2. Create thumbnails: ffmpeg -i video.mp4 -ss 00:00:01 -vframes 1 thumbnail.jpg
# 3. Place files in appropriate category folders
# 4. Update docs/index.html with new video entries