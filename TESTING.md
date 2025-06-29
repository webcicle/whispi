# Pi-Whispr Testing Guide

This guide explains how to test your Pi-Whispr system in a mock environment before deploying to the Raspberry Pi.

## Overview

The testing suite includes:

- **Environment Validation**: Automated checks for Python, Docker, and dependencies
- **Mock Servers**: Multiple Docker containers simulating different Pi environments
- **Connection Tests**: Basic WebSocket connectivity validation
- **Client Tests**: Full client functionality testing with real audio
- **Performance Tests**: Pi5 simulation with realistic performance metrics

## Quick Start

### Option 1: Automated Testing (Recommended)

Run the complete testing workflow:

```bash
./run_tests.sh
```

This will:

1. Validate your environment (Python, Docker, dependencies)
2. Install/update any missing dependencies
3. Run comprehensive tests
4. Generate test reports
5. Clean up containers automatically

### Option 2: Validate Environment First

Check if your system is ready for testing:

```bash
python3 validate_environment.py
```

This validates:

- Python 3.8+ is available
- Required modules (websockets, docker, etc.)
- Optional modules (pyaudio, numpy for client testing)
- Docker and Docker Compose installation
- Project files are present

### Option 3: Manual Testing Steps

#### Step 1: Install Dependencies

```bash
pip3 install -r requirements_test.txt
```

#### Step 2: Start Mock Servers

```bash
# Start all mock servers
docker-compose up -d

# Or start specific servers
docker-compose up -d mock-server    # Fast development (port 8765)
docker-compose up -d mock-pi5       # Pi5 simulation (port 8766)
docker-compose up -d mock-stress    # Stress testing (port 8767)
```

#### Step 3: Test Connection

```bash
# Test single server
python3 test_connection.py --server-url ws://localhost:8765

# Test all servers
python3 test_connection.py --all-servers

# Test with audio data
python3 test_connection.py --server-url ws://localhost:8765 --with-audio
```

#### Step 4: Test Client Functionality

```bash
# Mock mode (no real audio needed)
python3 client_runner.py --mode mock --duration 30

# Test mode (uses real audio - requires microphone)
python3 client_runner.py --mode test --duration 30

# Connect to specific server
python3 client_runner.py --server-url ws://localhost:8766 --mode mock
```

## Testing Modes

### Quick Tests (`./run_tests.sh --quick`)

- Basic connection validation
- Fast execution (~30 seconds)
- Good for CI/CD pipelines

### Client Tests (`./run_tests.sh --client`)

- Full client functionality testing
- Audio recording simulation
- Hotkey detection testing
- WebSocket communication validation

### Full Test Suite (`./run_tests.sh --full`)

- Complete system validation
- All mock server scenarios
- Performance testing
- Stress testing
- Comprehensive reporting

## Mock Server Scenarios

Your project includes three pre-configured mock scenarios:

### 1. Fast Development (`mock-server` - port 8765)

- Quick responses for rapid development
- Minimal latency simulation
- Best for basic functionality testing

### 2. Pi5 Simulation (`mock-pi5` - port 8766)

- Realistic Pi5 performance characteristics
- Actual processing delays
- Memory and CPU constraints simulation
- Best for production readiness testing

### 3. Stress Testing (`mock-stress` - port 8767)

- High latency scenarios
- Network interruption simulation
- Resource limitation testing
- Error condition testing

## Test Files Explained

| File                      | Purpose                              |
| ------------------------- | ------------------------------------ |
| `validate_environment.py` | Pre-flight environment checks        |
| `test_connection.py`      | Basic WebSocket connectivity tests   |
| `client_runner.py`        | Full client functionality simulation |
| `test_environment.py`     | Comprehensive system validation      |
| `run_tests.sh`            | Orchestrated testing workflow        |

## Troubleshooting

### Common Issues

**"pip: command not found"**

- The script now auto-detects pip3, python -m pip, etc.
- Run `python3 validate_environment.py` to check your setup

**"Docker daemon not running"**

- Start Docker Desktop (macOS/Windows) or Docker service (Linux)
- Verify with: `docker info`

**"No audio device found"**

- Use `--mode mock` for testing without microphone
- Install PyAudio: `pip3 install pyaudio`

**"Container not ready"**

- Check Docker logs: `docker-compose logs mock-server`
- Increase timeout: modify `wait_for_container()` in run_tests.sh

### Debug Mode

Enable verbose logging:

```bash
# Export debug flag
export TASKMASTER_LOG_LEVEL=DEBUG

# Run tests with detailed output
./run_tests.sh --full
```

### Manual Container Management

```bash
# Start specific containers
docker-compose up -d mock-server mock-pi5

# Check container status
docker-compose ps

# View container logs
docker-compose logs -f mock-server

# Stop and clean up
docker-compose down -v
```

## Integration Testing

### Testing with Real Pi Hardware

Once mock tests pass, test with actual Pi:

```bash
# On your development machine
./run_tests.sh --full

# Copy code to Pi
scp -r . pi@your-pi-ip:/home/pi/pi-whispr/

# On the Pi
docker-compose up pi-server

# Test client from development machine
python3 client_runner.py --server-url ws://your-pi-ip:8765 --mode test
```

### Continuous Integration

Add to your CI pipeline:

```yaml
# Example GitHub Actions
- name: Run Pi-Whispr Tests
  run: |
    chmod +x run_tests.sh
    ./run_tests.sh --quick
```

## Performance Validation

The test suite validates:

- **Latency**: End-to-end audio processing time
- **Accuracy**: Speech recognition quality
- **Resource Usage**: Memory and CPU consumption
- **Reliability**: Connection stability and error recovery

## Next Steps

After successful testing:

1. ✅ **Mock Environment Validated** - All tests pass locally
2. 🚀 **Deploy to Pi** - Copy code to Raspberry Pi
3. 🔧 **Pi Integration Test** - Test with actual Pi hardware
4. 🎯 **Production Ready** - Deploy your speech recognition system

For deployment instructions, see [pi_speech_server_setup.md](docs/pi_speech_server_setup.md).
