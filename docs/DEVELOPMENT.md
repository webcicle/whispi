# Development Guide

## Getting Started

### 1. Environment Setup

```bash
# Clone or navigate to the project
cd whisper-server

# Install dependencies
pip install -r requirements.txt
```

### 2. Testing the Mock Server

The mock server allows you to develop and test the client without requiring a Raspberry Pi or the actual Whisper models.

**Terminal 1: Start Mock Server**

```bash
python server/mock_server.py --debug
```

**Terminal 2: Test the Server**

```bash
python scripts/test_mock_server.py
```

You should see output like:

```
🧪 Testing Mock Whisper Server
========================================
Connecting to mock server at ws://localhost:8765...
✅ Connected successfully!
📥 Initial message: {"type":"status","status":"connected",...}
🏓 Testing ping...
📥 Pong response: {"type":"pong","client_id":"test_client",...}
🎵 Testing audio processing...
📥 Transcription response: {"type":"transcription_result","text":"Hello world",...}
✨ Transcribed text: 'Hello world'
⏱️  Processing time: 0.523s
🎯 Confidence: 0.85
✅ All tests passed! Mock server is working correctly.
```

### 3. Project Structure

```
whisper-server/
├── server/                 # Raspberry Pi server components
│   ├── mock_server.py     # Development mock server ✅
│   └── whisper_server.py  # Production Whisper server (TODO)
├── client/                # macOS client application
│   └── speech_client.py   # Main client app (TODO)
├── shared/                # Common utilities ✅
│   ├── constants.py       # Configuration constants
│   ├── exceptions.py      # Custom exceptions
│   └── protocol.py        # WebSocket protocol
├── tests/                 # Test suite
├── scripts/               # Utility scripts
│   └── test_mock_server.py ✅
└── docs/                  # Documentation
```

### 4. Development Workflow

1. **Start with Mock Server**: Use `server/mock_server.py` for initial development
2. **Build Client Components**: Develop `client/speech_client.py` against mock server
3. **Test Locally**: Verify everything works with the mock server
4. **Deploy to Pi**: Set up actual `server/whisper_server.py` on Raspberry Pi
5. **Integration Testing**: Test client against real Pi server

### 5. Next Steps

Based on the project requirements, here's what needs to be implemented:

#### Immediate (Local Development)

- [ ] `client/speech_client.py` - Basic WebSocket client
- [ ] `client/audio_recorder.py` - Audio capture with PyAudio
- [ ] `client/hotkey_manager.py` - Global hotkey detection

#### Next Phase (Pi Integration)

- [ ] `server/whisper_server.py` - Real Whisper server with faster-whisper
- [ ] `server/audio_processor.py` - Audio processing and VAD
- [ ] `client/text_inserter.py` - macOS text insertion

#### Testing & Polish

- [ ] Comprehensive test suite
- [ ] Performance optimization
- [ ] Error handling and recovery
- [ ] Documentation and setup scripts

### 6. Development Tools

- **Logging**: Use structured logging with context (see Python standards rule)
- **Testing**: Run `python scripts/test_mock_server.py` frequently
- **Performance**: Mock server simulates 0.2-1.5s processing times
- **Debugging**: Use `--debug` flag for verbose logging

### 7. Configuration

The project uses centralized configuration in `shared/constants.py`:

- **Audio**: 16kHz, mono, 20ms chunks
- **Network**: WebSocket on port 8765
- **Performance**: Optimized for Pi 5 with tiny.en model
- **Client**: Space key for recording, Accessibility API for text insertion

### 8. Troubleshooting

**Mock server won't start:**

```bash
# Check if port is in use
lsof -i :8765

# Use different port
python server/mock_server.py --port 8766
```

**Import errors:**

```bash
# Make sure you're in the project root
pwd  # Should show .../whisper-server

# Install missing dependencies
pip install -r requirements.txt
```

**WebSocket connection issues:**

- Ensure mock server is running
- Check firewall settings
- Verify correct host/port in client

## Architecture Decisions

### Why Mock Server First?

- Faster development iteration
- No dependency on Pi hardware
- Simulates realistic latency and responses
- Easier debugging and testing

### Why WebSocket Protocol?

- Low latency compared to HTTP
- Bidirectional communication
- Built-in ping/pong for connection health
- Efficient binary data transmission

### Why Modular Structure?

- Clear separation of concerns
- Easier testing and maintenance
- Reusable components
- Platform-specific optimizations

## Performance Targets

| Component         | Target  | Mock Server               |
| ----------------- | ------- | ------------------------- |
| Network Latency   | <50ms   | ~10ms                     |
| Audio Processing  | <500ms  | 200-1500ms (configurable) |
| Transcription     | <1.5s   | Simulated                 |
| **Total Latency** | **<2s** | **Realistic simulation**  |
