"""
Shared constants for Pi-Whispr system
Based on performance optimization research from docs
"""

# Audio Configuration (optimized for Whisper)
SAMPLE_RATE = 16000  # Whisper's native sample rate
CHANNELS = 1         # Mono recording
CHUNK_SIZE = 320     # 20ms chunks for VAD (16000 * 0.02)
AUDIO_FORMAT = "int16"

# WebSocket Configuration
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765
WEBSOCKET_TIMEOUT = 30

# Speech Recognition Configuration
DEFAULT_MODEL = "tiny.en"    # For development
PRODUCTION_MODEL = "small.en" # For Pi deployment
VAD_AGGRESSIVENESS = 2       # WebRTC VAD level (0-3)

# Performance Settings
MAX_RECORDING_SECONDS = 30   # Safety limit
SILENCE_THRESHOLD_MS = 500   # End recording after silence
PROCESSING_TIMEOUT = 10      # Max processing time

# Client Configuration - Push-to-talk with lock
HOTKEY_SPACE = "space"           # Primary recording key
HOTKEY_LOCK_COMBO = "fn+space"   # Lock recording on/off
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0

# Text Insertion
TEXT_INSERTION_METHOD = "accessibility_api"  # Primary method
TEXT_INSERTION_FALLBACK = "applescript"      # Fallback method

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL_DEV = "DEBUG"
LOG_LEVEL_PROD = "INFO"

# File Paths
MODELS_DIR = "models"
CONFIG_DIR = "config"
LOGS_DIR = "logs" 