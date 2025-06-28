# MacBook Client - Alternative Implementation Options

## Option 1: Simple Python Script (Recommended for Quick Start)

This is the easiest implementation using the code from the main guide. Pros: 15-minute setup. Cons: Requires Terminal to be running.

## Option 2: Swift Native App (Professional Solution)

For a more polished experience, here's a complete Swift macOS app that runs in the background:

### Swift App Setup

1. **Create new Xcode project**:
   - Open Xcode → Create New Project
   - macOS → App
   - Name: "SpeechTranscriber"
   - Language: Swift
   - Interface: SwiftUI

2. **Configure Info.plist** - Add these keys:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>Required for voice transcription</string>
<key>LSUIElement</key>
<true/>
```

3. **Main App Code** - Replace `ContentView.swift`:

```swift
import SwiftUI
import AVFoundation
import Network
import ApplicationServices

struct ContentView: View {
    @StateObject private var speechClient = SpeechClient()
    
    var body: some View {
        VStack(spacing: 20) {
            HStack {
                Circle()
                    .fill(speechClient.isConnected ? Color.green : Color.red)
                    .frame(width: 12, height: 12)
                Text(speechClient.isConnected ? "Connected to Pi" : "Disconnected")
                    .foregroundColor(.secondary)
            }
            
            VStack(alignment: .leading) {
                Text("Status: \(speechClient.status)")
                    .font(.headline)
                
                if speechClient.isRecording {
                    HStack {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 8, height: 8)
                            .scaleEffect(speechClient.isRecording ? 1.2 : 1.0)
                            .animation(.easeInOut(duration: 0.5).repeatForever(), value: speechClient.isRecording)
                        Text("Recording...")
                            .foregroundColor(.red)
                    }
                }
                
                Text("Last transcription:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(speechClient.lastTranscription)
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
            }
            
            VStack {
                Text("Hold SPACE to record")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Button("Test Connection") {
                    speechClient.testConnection()
                }
            }
        }
        .padding()
        .frame(width: 300, height: 250)
        .onAppear {
            speechClient.setup()
        }
    }
}

class SpeechClient: ObservableObject {
    @Published var isConnected = false
    @Published var isRecording = false
    @Published var status = "Initializing..."
    @Published var lastTranscription = "No transcription yet"
    
    private var audioEngine = AVAudioEngine()
    private var webSocketTask: URLSessionWebSocketTask?
    private var recordingTimer: Timer?
    private var eventMonitor: Any?
    
    private let piServerURL = "ws://192.168.1.100:8765" // UPDATE THIS
    
    func setup() {
        setupAudioEngine()
        connectToServer()
        setupGlobalHotkeys()
    }
    
    private func setupAudioEngine() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
            status = "Audio engine ready"
        } catch {
            status = "Audio setup failed: \(error.localizedDescription)"
        }
    }
    
    private func connectToServer() {
        guard let url = URL(string: piServerURL) else {
            status = "Invalid server URL"
            return
        }
        
        webSocketTask = URLSession.shared.webSocketTask(with: url)
        webSocketTask?.resume()
        
        // Listen for messages
        receiveMessage()
        
        // Test connection
        testConnection()
    }
    
    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                DispatchQueue.main.async {
                    self?.handleReceivedMessage(message)
                }
                // Continue listening
                self?.receiveMessage()
            case .failure(let error):
                DispatchQueue.main.async {
                    self?.status = "Connection error: \(error.localizedDescription)"
                    self?.isConnected = false
                }
            }
        }
    }
    
    private func handleReceivedMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
}

## Option 3: Enhanced Python Script with Better Audio Handling

Here's an improved Python version with more robust audio processing:

```python
#!/usr/bin/env python3
"""
Enhanced MacBook Speech Client with Improved Audio Processing
Better noise handling and audio quality optimization
"""

import asyncio
import websockets
import json
import pyaudio
import wave
import tempfile
import threading
import time
import numpy as np
from pynput import keyboard
import subprocess
import os
import webrtcvad
from scipy.signal import butter, lfilter

class EnhancedSpeechClient:
    def __init__(self, server_url="ws://192.168.1.100:8765"):
        self.server_url = server_url
        self.websocket = None
        self.is_recording = False
        self.recording_thread = None
        self.hotkey_listener = None
        
        # Enhanced audio configuration
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 320  # 20ms chunks for VAD
        self.record_seconds = 30
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        
        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.audio_buffer = []
        
        print("🎤 Enhanced MacBook Speech Client")
        print(f"📡 Server: {self.server_url}")
        print("🔥 Press and hold SPACE to record (with noise reduction)")
        
    def preprocess_audio(self, audio_data):
        """Apply noise reduction and filtering"""
        # Convert to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Apply bandpass filter (300-3400 Hz for speech)
        nyquist = self.rate * 0.5
        low = 300 / nyquist
        high = 3400 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_audio = lfilter(b, a, audio_np)
        
        # Normalize audio
        filtered_audio = filtered_audio / np.max(np.abs(filtered_audio))
        
        # Convert back to bytes
        return (filtered_audio * 32767).astype(np.int16).tobytes()
    
    def has_speech(self, audio_chunk):
        """Check if audio chunk contains speech using VAD"""
        try:
            return self.vad.is_speech(audio_chunk, self.rate)
        except:
            return True  # Default to assuming speech if VAD fails
    
    def _record_audio_enhanced(self):
        """Enhanced recording with VAD and noise reduction"""
        frames = []
        speech_frames = []
        silence_count = 0
        speech_detected = False
        
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("🔴 Recording with noise reduction...")
        
        while self.is_recording:
            try:
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
                
                # Check for speech activity
                if self.has_speech(data):
                    speech_detected = True
                    silence_count = 0
                    speech_frames.extend(frames[-10:])  # Include some context
                elif speech_detected:
                    silence_count += 1
                    speech_frames.append(data)
                    
                    # Stop if we have 1 second of silence after speech
                    if silence_count > 50:  # 50 * 20ms = 1 second
                        break
                        
            except Exception as e:
                print(f"Recording error: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        
        if speech_frames:
            # Process the speech audio
            audio_data = b''.join(speech_frames)
            processed_audio = self.preprocess_audio(audio_data)
            
            # Create WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                wf = wave.open(temp_file.name, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(processed_audio)
                wf.close()
                
                asyncio.create_task(self._send_audio_for_transcription(temp_file.name))
        else:
            print("🔇 No speech detected")
    
    # ... rest of the methods remain the same as the basic version ...
```

## Option 4: Automation Script (Zero Configuration)

For the ultimate in simplicity, here's a one-command setup script:

```bash
#!/bin/bash
# Complete automated setup script

# Create setup script
cat > ~/setup_speech_client.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Setting up MacBook Speech Client..."

# Install dependencies
echo "📦 Installing Python dependencies..."
pip3 install websockets pyaudio pynput pyobjc-framework-Cocoa pyobjc-framework-ApplicationServices numpy scipy webrtcvad

# Get Pi IP address
echo "🔍 Please enter your Raspberry Pi IP address:"
read -p "Pi IP (e.g., 192.168.1.100): " PI_IP

# Download and configure client
echo "📥 Setting up speech client..."
mkdir -p ~/speech-client
cd ~/speech-client

# Create the client script with user's Pi IP
cat > speech_client.py << PYTHON_EOF
# [Insert the complete Python client code here with PI_IP variable substituted]
PYTHON_EOF

# Replace PI_IP placeholder
sed -i '' "s/192.168.1.100/$PI_IP/g" speech_client.py

# Create launch script
cat > start_speech_client.sh << 'LAUNCH_EOF'
#!/bin/bash
cd ~/speech-client
python3 speech_client.py
LAUNCH_EOF

chmod +x start_speech_client.sh

echo "✅ Setup complete!"
echo "🎤 Run: ~/speech-client/start_speech_client.sh"
echo "📋 Don't forget to enable microphone and accessibility permissions!"

EOF

chmod +x ~/setup_speech_client.sh
~/setup_speech_client.sh
```

## Comparison of Options

| Option | Setup Time | Features | Background Operation | Maintenance |
|--------|------------|----------|---------------------|-------------|
| **Python Script** | 15 min | Basic | Requires Terminal | Low |
| **Swift App** | 2-3 hours | Professional UI | Yes | Medium |
| **Enhanced Python** | 30 min | Noise reduction, VAD | Requires Terminal | Low |
| **Automation Script** | 5 min | Zero config | Requires Terminal | None |

## Recommended Approach

**Start with Option 1** (basic Python script) to get working quickly, then optionally upgrade to the Swift app for a more polished experience.

The Python script will give you immediate functionality to test the Pi server, and you can always enhance it later with the advanced features from Option 3.

## Permissions Required (All Options)

1. **Microphone**: Automatically prompted
2. **Accessibility**: System Preferences → Security & Privacy → Accessibility
3. **For Swift app**: Additional Xcode developer account for code signing

## Next Steps After Setup

1. Test basic functionality with Python script
2. Optimize Pi server performance for your voice
3. Consider upgrading to Swift app for production use
4. Add custom hotkey combinations
5. Implement auto-startup for both Pi and Mac components