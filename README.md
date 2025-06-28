# Pi-Whispr: Local Speech-to-Text System

A fast, local speech-to-text system using Raspberry Pi 5 and macOS integration.

## Overview

Pi-Whispr provides sub-2 second speech transcription with:

- **Raspberry Pi 5**: Server running faster-whisper with WebSocket API
- **macOS Client**: Python script with global hotkeys and direct text insertion
- **Docker Development**: Local testing before Pi deployment
- **Zero Ongoing Costs**: Complete local processing

## Current Architecture (MVP)

### Recording Control

- **SPACE**: Hold to record (push-to-talk)
- **Fn + SPACE**: Toggle recording lock on/off
- Clear visual feedback for recording states

### Technology Stack

- **Network**: WebSocket (simple, reliable, adequate performance)
- **Client**: Python script (rapid development, easy debugging)
- **Text Insertion**: Accessibility API with AppleScript fallback
- **Models**: `tiny.en` (dev) → `small.en` (Pi production)

## Quick Start

### Development (Local Testing)

```bash
# 1. Setup environment
./scripts/setup.sh

# 2. Start mock server
docker-compose up dev-server

# 3. Run macOS client
python client/speech_client.py

# 4. Press SPACE to record, Fn+SPACE to lock
```

### Production (Pi Deployment)

```bash
# 1. Deploy to Pi
docker-compose up pi-server

# 2. Update client config with Pi IP
# 3. Same hotkey workflow
```

## Features

- **Fast**: Sub-2 second latency with optimized models
- **Local**: No cloud dependencies or ongoing costs
- **Integrated**: Direct text insertion in Cursor, browsers, and apps
- **Reliable**: WebSocket communication with automatic reconnection
- **Configurable**: Easy model switching via environment variables
- **Lock Mode**: Continuous recording without holding keys

## System Requirements

- **Raspberry Pi 5** 8GB with active cooling
- **macOS** 12+ with microphone and accessibility permissions
- **Docker** for development and deployment
- **2GB storage** for speech models

## Future Roadmap

### Phase 2: Performance Optimization

- **WebRTC Integration**: ~100-150ms latency improvement
- **Opus Audio Codec**: Real-time streaming optimization
- **Advanced VAD**: Improved voice activity detection

### Phase 3: Native Experience

- **Swift macOS App**: Native performance and background operation
- **Menu Bar Integration**: System tray controls and status
- **App Store Distribution**: Easy installation and updates

### Phase 4: Advanced Features

- **Multiple Models**: Specialized models for different use cases
- **Custom Vocabulary**: Domain-specific transcription accuracy
- **Multi-language Support**: Dynamic language switching

## Development Status

🚧 **Phase 1 (MVP)** - In Development

- [ ] Mock server implementation
- [ ] Python client with hotkey controls
- [ ] WebSocket communication
- [ ] Accessibility API text insertion
- [ ] Docker development environment
- [ ] Pi deployment and testing

## Documentation

See `docs/` directory for detailed setup guides and technical documentation.
