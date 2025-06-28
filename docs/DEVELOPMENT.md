# Development Guide

## Architecture Overview

Pi-Whispr uses a **WebSocket-based architecture** for simplicity and reliability in the MVP phase. Future enhancements may include WebRTC for lower latency and Swift native app for better performance.

### Current Technology Stack

- **Server**: Python WebSocket server with faster-whisper
- **Client**: Python script with PyAudio and macOS integration
- **Protocol**: WebSocket with structured JSON messages
- **Audio**: 16kHz mono, 20ms chunks with WebRTC VAD
- **Text Insertion**: Accessibility API (primary) + AppleScript (fallback)

## Getting Started

### 1. Environment Setup

```bash
# Quick setup with provided script
./scripts/setup.sh

# Manual setup
pip install -r requirements/client.txt
cp config/development.env .env
```

### 2. Development Workflow (Recommended Order)

#### Phase 1: Mock Server Testing

**Goal**: Test core workflow without model complexity

```bash
# Terminal 1: Start Mock Server
docker-compose up dev-server

# Terminal 2: Test the Server
python scripts/test_mock_server.py
```

Expected output:

```
🧪 Testing Mock Whisper Server
✅ Connected successfully!
🎵 Testing audio processing...
✨ Transcribed text: 'Mock transcription result'
⏱️  Processing time: 0.8s
```

#### Phase 2: Python Client Development

**Goal**: Implement hotkey controls and text insertion

1. **Basic WebSocket Client** (`client/speech_client.py`)

   - Connect to mock server
   - Send/receive structured messages
   - Handle connection errors and reconnection

2. **Audio Recording** (`client/audio_recorder.py`)

   - PyAudio microphone capture
   - 16kHz mono recording
   - VAD-based silence detection

3. **Hotkey Management** (`client/hotkey_manager.py`)

   - Global Fn key monitoring (push-to-talk)
   - Fn + Fn combination (toggle lock)
   - Visual feedback for recording states

4. **Text Insertion** (`client/text_inserter.py`)
   - Accessibility API implementation
   - AppleScript fallback
   - Error handling and permission checks

#### Phase 3: Docker Integration

**Goal**: Containerized development environment

```bash
# Build and test development container
docker-compose build dev-server
docker-compose up dev-server

# Test client against containerized mock server
python client/speech_client.py
```

#### Phase 4: Real Whisper Implementation

**Goal**: Replace mock with actual transcription

1. **Whisper Server** (`server/whisper_server.py`)

   - faster-whisper integration
   - Model loading and optimization
   - Audio processing pipeline

2. **Pi Deployment**
   - ARM64 container build
   - Model optimization for Pi 5
   - Performance monitoring

### 3. Project Structure

```
pi-whispr/
├── server/                 # Server components
│   ├── mock_server.py     # Development mock (IMPLEMENT FIRST)
│   ├── whisper_server.py  # Production server (Phase 4)
│   ├── audio_processor.py # VAD and audio handling
│   └── transcription.py   # Whisper model wrapper
├── client/                # macOS client (IMPLEMENT SECOND)
│   ├── speech_client.py   # Main client application
│   ├── audio_recorder.py  # Audio capture with PyAudio
│   ├── hotkey_manager.py  # Global hotkey detection
│   └── text_inserter.py   # macOS text insertion
├── shared/                # Common utilities ✅
│   ├── constants.py       # Configuration constants
│   ├── exceptions.py      # Custom exceptions
│   └── protocol.py        # WebSocket protocol
└── tests/                 # Test suite
```

### 4. Configuration

The project uses environment-based configuration:

#### Development Settings

```bash
# .env (created from config/development.env)
WHISPER_MODEL=tiny.en
USE_MOCK_SERVER=true
PI_SERVER_URL=ws://localhost:8765
HOTKEY_PRIMARY=fn
HOTKEY_LOCK=fn+space
```

#### Production Settings

```bash
# config/production.env
WHISPER_MODEL=small.en
USE_MOCK_SERVER=false
PI_SERVER_URL=ws://192.168.1.100:8765
OPTIMIZE_FOR_PI=true
```

### 5. Testing Strategy

#### Mock Server Testing

```python
# Test WebSocket communication
async def test_mock_server():
    client = SpeechClient("ws://localhost:8765")
    await client.connect()

    # Test audio processing
    audio_data = generate_test_audio()
    result = await client.send_audio(audio_data)

    assert result.text is not None
    assert result.processing_time < 2.0
```

#### Integration Testing

```python
# Test full recording workflow
def test_recording_workflow():
    recorder = AudioRecorder()
    hotkey_manager = HotkeyManager()
    text_inserter = TextInserter()

    # Simulate: press fn -> record -> transcribe -> insert
    # This tests the complete user workflow
```

### 6. Development Tools

- **Logging**: Use `structlog` for structured logging with context
- **Testing**: Run mock server tests frequently during development
- **Performance**: Mock server simulates realistic processing times (0.2-1.5s)
- **Debugging**: Use `--debug` flag for verbose logging

### 7. Performance Targets (Current Implementation)

| Component         | Target  | Current Implementation            |
| ----------------- | ------- | --------------------------------- |
| Network Latency   | <100ms  | WebSocket (~50ms local)           |
| Audio Processing  | <500ms  | PyAudio capture                   |
| Transcription     | <1.5s   | tiny.en model                     |
| Text Insertion    | <50ms   | Accessibility API                 |
| **Total Latency** | **<2s** | **Achievable with current stack** |

### 8. Future Enhancement Paths

#### WebRTC Migration (Phase 2)

- Replace WebSocket with WebRTC data channels
- Implement Opus codec for audio streaming
- Add STUN/TURN server support for NAT traversal

#### Swift Native App (Phase 3)

- Port Python client to Swift
- Use AVAudioEngine for lower-latency audio capture
- Implement native macOS menu bar integration

### 9. Troubleshooting

**Mock server won't start:**

```bash
# Check if port is in use
lsof -i :8765
docker-compose down  # Clean up containers
```

**Python client import errors:**

```bash
# Install dependencies
pip install -r requirements/client.txt

# Check Python path
export PYTHONPATH=$PWD:$PYTHONPATH
```

**macOS permission issues:**

- **Microphone**: System Preferences → Privacy → Microphone
- **Accessibility**: System Preferences → Privacy → Accessibility
- Add Terminal or Python to both lists

**WebSocket connection issues:**

```bash
# Test server connectivity
curl -i -N -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Key: test" \
  -H "Sec-WebSocket-Version: 13" \
  http://localhost:8765/
```

## Next Steps

1. **Implement mock server** following `shared/protocol.py` patterns
2. **Build basic Python client** with WebSocket communication
3. **Add hotkey functionality** with recording state management
4. **Integrate text insertion** with permission handling
5. **Test complete workflow** before moving to real Whisper implementation

Focus on getting the **core workflow working end-to-end** with the mock server before adding complexity.
