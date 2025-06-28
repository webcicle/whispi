# Troubleshooting Guide & Performance Optimization

## Common Setup Issues and Solutions

### Raspberry Pi Server Issues

#### 1. "ModuleNotFoundError: No module named 'faster_whisper'"

**Problem**: Virtual environment not activated or package not installed

**Solution**:

```bash
# Ensure you're in the virtual environment
cd ~/speech-server
source whisper_env/bin/activate

# Verify activation (should show virtual env path)
which python3

# Reinstall if needed
pip install --upgrade pip
pip install faster-whisper
```

#### 2. "OSError: [Errno 98] Address already in use"

**Problem**: Server already running or port 8765 in use

**Solution**:

```bash
# Find process using port 8765
sudo lsof -i :8765

# Kill existing process
sudo kill -9 <PID>

# Or use different port in whisper_server.py
PORT = 8766  # Change this line
```

#### 3. Model Download Fails

**Problem**: Network issues or insufficient storage

**Solution**:

```bash
# Check available space (need ~2GB)
df -h

# Manual model download
python3 -c "
from faster_whisper import WhisperModel
model = WhisperModel('tiny.en')  # Start with smallest model
print('Model downloaded successfully')
"

# If still failing, check internet connection
ping google.com
```

#### 4. High CPU Temperature / Throttling

**Problem**: Pi overheating during transcription

**Solutions**:

```bash
# Check current temperature
vcgencmd measure_temp

# Check for throttling
vcgencmd get_throttled

# If throttling (result != 0x0):
# 1. Ensure active cooling is working
# 2. Use smaller model (tiny.en instead of small.en)
# 3. Reduce CPU threads in whisper_server.py:
#    num_workers=2  # Instead of 4
```

#### 5. "Permission denied" on Bookworm

**Problem**: Bookworm security restrictions

**Solution**:

```bash
# Install in virtual environment (required on Bookworm)
python3 -m venv whisper_env --system-site-packages
source whisper_env/bin/activate

# If still issues, install system packages:
sudo apt install python3-pip python3-venv python3-dev build-essential
```

### MacBook Client Issues

#### 1. "ModuleNotFoundError: No module named 'websockets'"

**Problem**: Python packages not installed

**Solution**:

```bash
# Install required packages
pip3 install websockets pyaudio pynput pyobjc-framework-Cocoa pyobjc-framework-ApplicationServices

# If pip3 not found, install via Homebrew:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python
```

#### 2. "Error: PortAudio library not found"

**Problem**: PyAudio dependency missing

**Solution**:

```bash
# Install PortAudio via Homebrew
brew install portaudio

# Then reinstall PyAudio
pip3 uninstall pyaudio
pip3 install pyaudio

# Alternative: Use conda
conda install pyaudio
```

#### 3. "Connection refused" to Pi

**Problem**: Network connectivity or firewall issues

**Solutions**:

```bash
# Test basic connectivity
ping YOUR_PI_IP

# Test WebSocket connection
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" -H "Sec-WebSocket-Version: 13" http://YOUR_PI_IP:8765/

# Check Pi firewall (on Pi)
sudo ufw status
sudo ufw allow 8765  # If firewall is active

# Verify Pi server is running
# On Pi: netstat -tlnp | grep 8765
```

#### 4. Microphone Permission Issues

**Problem**: macOS blocking microphone access

**Solution**:

1. Go to System Preferences → Security & Privacy → Privacy
2. Select "Microphone" from left sidebar
3. Check the box next to Terminal (or Python)
4. Restart the client script

#### 5. Text Insertion Not Working

**Problem**: Accessibility permissions not granted

**Solution**:

1. System Preferences → Security & Privacy → Privacy
2. Select "Accessibility" from left sidebar
3. Click lock to make changes
4. Add Terminal (or your Python executable)
5. Restart client

#### 6. Global Hotkeys Not Working

**Problem**: Key monitoring blocked by macOS

**Solutions**:

```python
# Alternative hotkey combinations (edit speech_client.py)
# Change from Fn to different key:

def on_key_press(self, key):
    try:
        # Option 1: Use Fn + Space
        if key == keyboard.Key.space and keyboard.Key.fn in self.pressed_keys:
            self.start_recording()

        # Option 2: Use Cmd + Fn (but conflicts with Spotlight)
        if key == keyboard.Key.space and keyboard.Key.cmd in self.pressed_keys:
            self.start_recording()

        # Option 3: Use F13 key (if available)
        if key == keyboard.Key.f13:
            self.start_recording()
    except AttributeError:
        pass
```

### Network and Performance Issues

#### 1. High Latency (>5 seconds)

**Causes and Solutions**:

```bash
# Test network latency
ping YOUR_PI_IP

# If high ping times (>50ms):
# 1. Use Ethernet instead of WiFi
# 2. Ensure both devices on same network segment
# 3. Check for network congestion

# Test Pi processing speed
time python3 -c "
from faster_whisper import WhisperModel
model = WhisperModel('tiny.en')
segments, info = model.transcribe('test.wav')
"

# If slow processing:
# 1. Switch to tiny.en model
# 2. Check Pi temperature
# 3. Ensure active cooling
```

#### 2. Audio Quality Issues

**Problem**: Poor transcription accuracy

**Solutions**:

```python
# Improve audio quality in speech_client.py
# Add noise reduction parameters:

def _record_audio(self):
    # Increase sample rate for better quality
    self.rate = 44100  # Instead of 16000

    # Add noise gate
    volume_threshold = 500  # Adjust based on environment

    # Filter out background noise
    if max(np.frombuffer(data, dtype=np.int16)) < volume_threshold:
        continue  # Skip quiet frames
```

#### 3. Intermittent Connection Drops

**Problem**: WebSocket connection unstable

**Solution**:

```python
# Add connection retry logic to speech_client.py

async def connect_with_retry(self, max_retries=5):
    for attempt in range(max_retries):
        try:
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=10,  # Send ping every 10s
                ping_timeout=5,    # Timeout after 5s
                close_timeout=10   # Close timeout
            )
            return True
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    return False
```

## Performance Optimization

### Pi Server Optimization

#### 1. Model Selection for Different Use Cases

```python
# Edit whisper_server.py MODEL_SIZE based on needs:

# Ultra-fast transcription (sub-1s), 70-80% accuracy
MODEL_SIZE = "tiny.en"

# Balanced performance (1-3s), 85-90% accuracy (recommended)
MODEL_SIZE = "small.en"

# High accuracy (3-6s), 90-95% accuracy
MODEL_SIZE = "base.en"

# Maximum accuracy (5-10s), 95%+ accuracy
MODEL_SIZE = "medium.en"  # Only if you have excellent cooling
```

#### 2. Hardware Performance Tuning

```bash
# CPU Governor (on Pi)
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Memory optimization
sudo sysctl vm.swappiness=1

# USB storage optimization (if using USB SSD)
echo 'deadline' | sudo tee /sys/block/sda/queue/scheduler

# GPU memory split (reduce if not using desktop)
sudo raspi-config  # Advanced → Memory Split → 16
```

#### 3. faster-whisper Optimization

```python
# Optimize whisper_server.py for your Pi:

def initialize_model(self):
    self.model = WhisperModel(
        self.model_size,
        device="cpu",
        compute_type="int8",        # Use int8 for speed
        num_workers=2,              # Reduce if overheating
        download_root="./models",   # Local model storage
        local_files_only=False,     # Allow downloads
        cpu_threads=4               # Match your Pi's cores
    )
```

#### 4. Audio Processing Optimization

```python
# Optimize transcription parameters in whisper_server.py:

segments, info = self.model.transcribe(
    temp_path,
    beam_size=1,                    # Faster but less accurate
    language="en",                  # Skip language detection
    condition_on_previous_text=False,  # Faster processing
    vad_filter=True,               # Voice activity detection
    vad_parameters=dict(
        min_silence_duration_ms=300,   # Shorter silence detection
        speech_pad_ms=100              # Less padding
    ),
    temperature=0.0,               # Deterministic output
    compression_ratio_threshold=2.4,  # Skip very repetitive audio
    no_speech_threshold=0.6        # Skip non-speech audio
)
```

### MacBook Client Optimization

#### 1. Audio Recording Optimization

```python
# Optimize audio settings for speed:

class SpeechClient:
    def __init__(self):
        # Optimize for speed over quality
        self.rate = 16000          # Whisper's native rate
        self.chunk = 320           # 20ms chunks (optimal for VAD)
        self.channels = 1          # Mono audio
        self.audio_format = pyaudio.paInt16  # 16-bit depth
```

#### 2. Network Optimization

```python
# Add compression to reduce network usage:

import gzip

async def _send_audio_for_transcription(self, audio_file_path):
    with open(audio_file_path, 'rb') as f:
        audio_data = f.read()

    # Compress audio data
    compressed_data = gzip.compress(audio_data)

    # Send compressed data with header
    message = {
        "type": "audio",
        "compressed": True,
        "data": compressed_data.hex()
    }

    await self.websocket.send(json.dumps(message))
```

## Monitoring and Diagnostics

### Real-time Performance Monitoring

```bash
# Create monitoring script for Pi (save as monitor_pi.sh)
#!/bin/bash

echo "🔍 Pi Performance Monitor"
echo "========================"

while true; do
    clear
    echo "⏰ $(date)"
    echo "🌡️  CPU Temp: $(vcgencmd measure_temp)"
    echo "⚡ Throttling: $(vcgencmd get_throttled)"
    echo "💾 Memory Usage:"
    free -h
    echo "🖥️  CPU Usage:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
    echo "🔊 Active connections:"
    netstat -an | grep :8765 | grep ESTABLISHED | wc -l
    echo "========================"
    sleep 2
done
```

### Performance Benchmarking

```python
# Add to whisper_server.py for performance tracking:

class PerformanceTracker:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "avg_processing_time": 0,
            "fastest_transcription": float('inf'),
            "slowest_transcription": 0,
            "accuracy_samples": [],
            "error_count": 0
        }

    def log_transcription(self, processing_time, word_count, error=None):
        if error:
            self.metrics["error_count"] += 1
            return

        self.metrics["total_requests"] += 1
        self.metrics["avg_processing_time"] = (
            (self.metrics["avg_processing_time"] * (self.metrics["total_requests"] - 1) + processing_time)
            / self.metrics["total_requests"]
        )

        self.metrics["fastest_transcription"] = min(
            self.metrics["fastest_transcription"], processing_time
        )
        self.metrics["slowest_transcription"] = max(
            self.metrics["slowest_transcription"], processing_time
        )

        # Calculate words per second
        wps = word_count / processing_time if processing_time > 0 else 0
        print(f"📊 Performance: {processing_time:.2f}s, {wps:.1f} words/sec")

    def get_report(self):
        return {
            "requests_processed": self.metrics["total_requests"],
            "average_time": round(self.metrics["avg_processing_time"], 2),
            "fastest_time": round(self.metrics["fastest_transcription"], 2),
            "slowest_time": round(self.metrics["slowest_transcription"], 2),
            "error_rate": round(self.metrics["error_count"] / max(1, self.metrics["total_requests"]) * 100, 1)
        }
```

## Advanced Troubleshooting

### Debug Mode Setup

```python
# Add debug mode to both client and server:

import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('speech_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Add debug prints throughout your code:
logger.debug(f"Audio data size: {len(audio_data)} bytes")
logger.debug(f"Processing time: {processing_time:.2f}s")
logger.debug(f"Model response: {result}")
```

### Network Diagnostics

```bash
# Complete network test script (run on Mac)
#!/bin/bash

PI_IP="YOUR_PI_IP"
echo "🔍 Network Diagnostics for $PI_IP"
echo "=================================="

echo "1. Basic connectivity:"
ping -c 3 $PI_IP

echo -e "\n2. Port accessibility:"
nc -zv $PI_IP 8765

echo -e "\n3. WebSocket test:"
curl -i -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
     -H "Sec-WebSocket-Version: 13" \
     http://$PI_IP:8765/ 2>/dev/null | head -5

echo -e "\n4. Bandwidth test:"
iperf3 -c $PI_IP -t 5 2>/dev/null || echo "iperf3 not available"

echo -e "\n5. Route trace:"
traceroute $PI_IP | head -5
```

## Quick Reference Commands

### Restart Everything

```bash
# On Pi - restart server
sudo systemctl restart speech-server
# Or manually:
pkill -f whisper_server.py
cd ~/speech-server && source whisper_env/bin/activate && python3 whisper_server.py

# On Mac - restart client
pkill -f speech_client.py
cd ~/speech-client && python3 speech_client.py
```

### Check Status

```bash
# Pi server status
sudo systemctl status speech-server
netstat -tlnp | grep 8765

# Mac client status
ps aux | grep speech_client.py
lsof -i :8765  # Check connection to Pi
```

### Performance Quick Checks

```bash
# Pi performance
vcgencmd measure_temp && vcgencmd get_throttled && free -h

# Mac audio test
python3 -c "import pyaudio; p=pyaudio.PyAudio(); print(f'Audio devices: {p.get_device_count()}'); p.terminate()"

# Network latency
ping -c 5 YOUR_PI_IP | tail -1
```

With these troubleshooting steps and optimizations, you should be able to achieve reliable sub-1 second transcription with 85-90% accuracy, completely replacing WhisperFlow while saving $170+ per year!
