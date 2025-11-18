// Copyright (C) 2025 Apple Inc. All Rights Reserved.
// SPDX-License-Identifier: Apple Sample Code License

// STARFlow Video Gallery JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the gallery
    initializeTabs();
    initializeVideoHandling();
    initializeLazyLoading();
    initializeAnalytics();
});

// Tab functionality
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');

            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            document.getElementById(targetTab).classList.add('active');

            // Update URL hash for bookmarking
            history.pushState(null, null, `#${targetTab}`);

            // Pause all videos when switching tabs
            pauseAllVideos();
        });
    });

    // Handle initial tab from URL hash
    const hash = window.location.hash.slice(1);
    if (hash) {
        const targetButton = document.querySelector(`[data-tab="${hash}"]`);
        if (targetButton) {
            targetButton.click();
        }
    }

    // Handle browser back/forward
    window.addEventListener('popstate', () => {
        const hash = window.location.hash.slice(1);
        if (hash) {
            const targetButton = document.querySelector(`[data-tab="${hash}"]`);
            if (targetbutton) {
                targetButton.click();
            }
        }
    });
}

// Video handling functionality
function initializeVideoHandling() {
    const videos = document.querySelectorAll('video');

    videos.forEach(video => {
        // Add play/pause toggle on click
        video.addEventListener('click', function(e) {
            if (this.paused) {
                pauseAllVideos(); // Pause other videos
                this.play();
            } else {
                this.pause();
            }
        });

        // Add loading error handling
        video.addEventListener('error', function() {
            const container = this.parentElement;
            const errorDiv = document.createElement('div');
            errorDiv.className = 'video-error';
            errorDiv.innerHTML = `
                <div class="error-content">
                    <p>Video failed to load</p>
                    <button onclick="this.parentElement.parentElement.previousElementSibling.load()">
                        Retry
                    </button>
                </div>
            `;
            container.appendChild(errorDiv);
        });

        // Add loading state management
        video.addEventListener('loadstart', function() {
            this.parentElement.classList.add('loading');
        });

        video.addEventListener('canplay', function() {
            this.parentElement.classList.remove('loading');
        });

        // Track video interactions for analytics
        video.addEventListener('play', function() {
            trackEvent('video_play', {
                video_src: this.src,
                video_title: this.closest('.video-item')?.querySelector('h3')?.textContent || 'Unknown'
            });
        });

        video.addEventListener('ended', function() {
            trackEvent('video_complete', {
                video_src: this.src,
                video_title: this.closest('.video-item')?.querySelector('h3')?.textContent || 'Unknown'
            });
        });
    });

    // Add keyboard controls
    document.addEventListener('keydown', function(e) {
        const activeVideo = document.querySelector('video:not([paused])');
        if (activeVideo) {
            switch(e.code) {
                case 'Space':
                    e.preventDefault();
                    if (activeVideo.paused) {
                        activeVideo.play();
                    } else {
                        activeVideo.pause();
                    }
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    activeVideo.currentTime = Math.max(0, activeVideo.currentTime - 5);
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    activeVideo.currentTime = Math.min(activeVideo.duration, activeVideo.currentTime + 5);
                    break;
                case 'KeyM':
                    e.preventDefault();
                    activeVideo.muted = !activeVideo.muted;
                    break;
            }
        }
    });
}

// Lazy loading for videos
function initializeLazyLoading() {
    if ('IntersectionObserver' in window) {
        const videoObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const video = entry.target;
                    const poster = video.getAttribute('poster');

                    // Load video source if not already loaded
                    if (!video.hasAttribute('data-loaded')) {
                        video.setAttribute('data-loaded', 'true');
                        // Preload metadata for faster playback
                        video.preload = 'metadata';
                    }

                    videoObserver.unobserve(video);
                }
            });
        }, {
            rootMargin: '50px'
        });

        document.querySelectorAll('video').forEach(video => {
            videoObserver.observe(video);
        });
    }
}

// Analytics and tracking
function initializeAnalytics() {
    // Track page views
    trackEvent('page_view', {
        page: 'video_gallery',
        user_agent: navigator.userAgent,
        screen_resolution: `${screen.width}x${screen.height}`
    });

    // Track tab switches
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            trackEvent('tab_switch', {
                tab: button.getAttribute('data-tab')
            });
        });
    });

    // Track external link clicks
    document.querySelectorAll('a[href^="http"]').forEach(link => {
        link.addEventListener('click', () => {
            trackEvent('external_link_click', {
                url: link.href,
                text: link.textContent
            });
        });
    });
}

// Utility functions
function pauseAllVideos() {
    document.querySelectorAll('video').forEach(video => {
        if (!video.paused) {
            video.pause();
        }
    });
}

function trackEvent(eventName, properties = {}) {
    // Basic console logging - replace with your analytics service
    console.log('Analytics Event:', eventName, properties);

    // Example integration with Google Analytics
    if (typeof gtag !== 'undefined') {
        gtag('event', eventName, properties);
    }

    // Example integration with custom analytics
    if (typeof analytics !== 'undefined') {
        analytics.track(eventName, properties);
    }
}

// Video quality selector (optional enhancement)
function addQualitySelector(video) {
    const qualities = [
        { label: '480p', src: video.src },
        { label: '720p', src: video.src.replace('.mp4', '_720p.mp4') },
        { label: '1080p', src: video.src.replace('.mp4', '_1080p.mp4') }
    ];

    const container = video.parentElement;
    const qualitySelect = document.createElement('select');
    qualitySelect.className = 'quality-selector';

    qualities.forEach(quality => {
        const option = document.createElement('option');
        option.value = quality.src;
        option.textContent = quality.label;
        if (quality.src === video.src) {
            option.selected = true;
        }
        qualitySelect.appendChild(option);
    });

    qualitySelect.addEventListener('change', function() {
        const currentTime = video.currentTime;
        const isPlaying = !video.paused;

        video.src = this.value;
        video.load();

        video.addEventListener('loadedmetadata', function() {
            this.currentTime = currentTime;
            if (isPlaying) {
                this.play();
            }
        }, { once: true });
    });

    container.appendChild(qualitySelect);
}

// Performance monitoring
function monitorPerformance() {
    // Monitor video loading performance
    document.querySelectorAll('video').forEach(video => {
        const startTime = performance.now();

        video.addEventListener('loadeddata', function() {
            const loadTime = performance.now() - startTime;
            trackEvent('video_load_time', {
                video_src: this.src,
                load_time_ms: Math.round(loadTime)
            });
        }, { once: true });
    });

    // Monitor page performance
    window.addEventListener('load', function() {
        setTimeout(() => {
            const navigation = performance.getEntriesByType('navigation')[0];
            trackEvent('page_performance', {
                load_time_ms: Math.round(navigation.loadEventEnd - navigation.fetchStart),
                dom_content_loaded_ms: Math.round(navigation.domContentLoadedEventEnd - navigation.fetchStart)
            });
        }, 0);
    });
}

// Initialize performance monitoring
monitorPerformance();

// Accessibility enhancements
function enhanceAccessibility() {
    // Add ARIA labels to videos
    document.querySelectorAll('video').forEach(video => {
        const videoTitle = video.closest('.video-item')?.querySelector('h3')?.textContent;
        if (videoTitle) {
            video.setAttribute('aria-label', `Video: ${videoTitle}`);
        }
    });

    // Add keyboard navigation hints
    const helpText = document.createElement('div');
    helpText.className = 'keyboard-help';
    helpText.innerHTML = `
        <p><strong>Keyboard shortcuts:</strong></p>
        <ul>
            <li>Space: Play/Pause active video</li>
            <li>← →: Skip 5 seconds backward/forward</li>
            <li>M: Toggle mute</li>
        </ul>
    `;
    helpText.style.display = 'none';
    document.body.appendChild(helpText);

    // Toggle help with '?' key
    document.addEventListener('keydown', function(e) {
        if (e.key === '?' || e.key === 'h') {
            helpText.style.display = helpText.style.display === 'none' ? 'block' : 'none';
        }
        if (e.key === 'Escape') {
            helpText.style.display = 'none';
        }
    });
}

// Initialize accessibility enhancements
enhanceAccessibility();