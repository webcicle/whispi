"""
PyAudio-based Audio Recording System for macOS Client

This module implements the core audio recording functionality for pi-whispr,
providing high-quality audio capture suitable for speech-to-text transcription.

Features:
- 16kHz mono audio recording in 20ms chunks
- macOS microphone permission handling
- Audio device selection and management  
- Real-time audio level monitoring
- Robust error handling and device disconnection recovery
- Buffer management for streaming
"""

import pyaudio
import numpy as np
import threading
import time
import logging
import queue
from typing import Optional, Callable, List, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import struct

# macOS specific imports for permissions
try:
    import Foundation
    import AVFoundation
    from Cocoa import NSWorkspace, NSBundle
except ImportError:
    Foundation = None
    AVFoundation = None
    NSWorkspace = None
    NSBundle = None

from shared.constants import SAMPLE_RATE, CHANNELS, CHUNK_SIZE, AUDIO_FORMAT

logger = logging.getLogger(__name__)


class RecordingState(Enum):
    """Audio recording state enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RECORDING = "recording"
    STOPPING = "stopping"
    ERROR = "error"


class PermissionStatus(Enum):
    """Microphone permission status"""
    UNKNOWN = "unknown"
    GRANTED = "granted"
    DENIED = "denied"
    RESTRICTED = "restricted"


@dataclass
class AudioDeviceInfo:
    """Information about an audio input device"""
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float
    is_default: bool = False


@dataclass
class AudioChunk:
    """Container for audio data with metadata"""
    data: bytes
    timestamp: float
    chunk_index: int
    sample_rate: int
    channels: int
    level_db: float
    is_clipped: bool = False


class AudioLevelMonitor:
    """Real-time audio level monitoring and visualization"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.level_history: List[float] = []
        self.peak_level = -float('inf')
        self.rms_level = -float('inf')
        self.is_clipping = False
        self.clip_threshold_db = -1.0  # dBFS
        
    def update(self, audio_data: bytes, sample_rate: int) -> float:
        """Update audio levels with new data and return current RMS level in dB"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return -float('inf')
            
            # Calculate RMS level
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            
            # Convert to dB (avoiding log(0))
            if rms > 0:
                rms_db = 20 * np.log10(rms / 32768.0)  # 32768 = max int16 value
            else:
                rms_db = -float('inf')
            
            # Calculate peak level
            peak = np.max(np.abs(audio_array))
            peak_db = 20 * np.log10(peak / 32768.0) if peak > 0 else -float('inf')
            
            # Check for clipping
            self.is_clipping = peak_db > self.clip_threshold_db
            
            # Update levels
            self.rms_level = rms_db
            self.peak_level = peak_db
            
            # Maintain level history
            self.level_history.append(rms_db)
            if len(self.level_history) > self.window_size:
                self.level_history.pop(0)
            
            return rms_db
            
        except Exception as e:
            logger.error(f"Error calculating audio levels: {e}")
            return -float('inf')
    
    def get_average_level(self) -> float:
        """Get average RMS level over the window"""
        if not self.level_history:
            return -float('inf')
        
        valid_levels = [level for level in self.level_history if level > -float('inf')]
        if not valid_levels:
            return -float('inf')
        
        return sum(valid_levels) / len(valid_levels)
    
    def get_visual_level(self, max_bars: int = 20) -> str:
        """Get a visual representation of the current audio level"""
        if self.rms_level == -float('inf'):
            return '|' + ' ' * (max_bars - 1) + '|'
        
        # Map dB range (-60 to 0) to bar count
        db_range = 60.0
        normalized_level = max(0, min(1, (self.rms_level + db_range) / db_range))
        bar_count = int(normalized_level * (max_bars - 2))
        
        # Create visual representation
        bars = '█' * bar_count + ' ' * ((max_bars - 2) - bar_count)
        color_indicator = '!' if self.is_clipping else ':'
        
        return f'|{bars}|{color_indicator}'


class AudioPermissionHandler:
    """Handle macOS microphone permissions using pyobjc"""
    
    @staticmethod
    def check_permission_status() -> PermissionStatus:
        """Check current microphone permission status"""
        if not AVFoundation:
            logger.warning("AVFoundation not available, cannot check permissions")
            return PermissionStatus.UNKNOWN
        
        try:
            # Check authorization status
            status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
                AVFoundation.AVMediaTypeAudio
            )
            
            if status == AVFoundation.AVAuthorizationStatusAuthorized:
                return PermissionStatus.GRANTED
            elif status == AVFoundation.AVAuthorizationStatusDenied:
                return PermissionStatus.DENIED
            elif status == AVFoundation.AVAuthorizationStatusRestricted:
                return PermissionStatus.RESTRICTED
            else:  # AVAuthorizationStatusNotDetermined
                return PermissionStatus.UNKNOWN
                
        except Exception as e:
            logger.error(f"Error checking microphone permissions: {e}")
            return PermissionStatus.UNKNOWN
    
    @staticmethod
    async def request_permission() -> bool:
        """Request microphone permission from user"""
        if not AVFoundation:
            logger.warning("AVFoundation not available, cannot request permissions")
            return False
        
        try:
            # First check if already granted
            current_status = AudioPermissionHandler.check_permission_status()
            if current_status == PermissionStatus.GRANTED:
                return True
            
            # Request permission
            result = await AudioPermissionHandler._request_permission_async()
            return result
            
        except Exception as e:
            logger.error(f"Error requesting microphone permission: {e}")
            return False
    
    @staticmethod
    async def _request_permission_async() -> bool:
        """Async wrapper for permission request"""
        import asyncio
        
        def request_callback(granted):
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    future.set_result(granted), loop
                )
        
        future = asyncio.Future()
        
        try:
            AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVFoundation.AVMediaTypeAudio,
                request_callback
            )
            
            # Wait for result with timeout
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
            
        except asyncio.TimeoutError:
            logger.error("Permission request timed out")
            return False
        except Exception as e:
            logger.error(f"Permission request failed: {e}")
            return False


class AudioDeviceManager:
    """Manage audio input devices and selection"""
    
    def __init__(self, audio_instance: pyaudio.PyAudio):
        self.audio = audio_instance
        self._devices_cache: List[AudioDeviceInfo] = []
        self._cache_time = 0
        self._cache_duration = 10.0  # seconds
    
    def get_input_devices(self, refresh: bool = False) -> List[AudioDeviceInfo]:
        """Get list of available audio input devices"""
        current_time = time.time()
        
        # Use cache if still valid and not forced refresh
        if (not refresh and 
            self._devices_cache and 
            current_time - self._cache_time < self._cache_duration):
            return self._devices_cache
        
        devices = []
        
        try:
            default_input_index = self.audio.get_default_input_device_info()['index']
            device_count = self.audio.get_device_count()
            
            for i in range(device_count):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    
                    # Only include input devices
                    if device_info['maxInputChannels'] > 0:
                        audio_device = AudioDeviceInfo(
                            index=i,
                            name=device_info['name'],
                            max_input_channels=device_info['maxInputChannels'],
                            default_sample_rate=device_info['defaultSampleRate'],
                            is_default=(i == default_input_index)
                        )
                        devices.append(audio_device)
                        
                except Exception as e:
                    logger.warning(f"Error getting device info for index {i}: {e}")
                    continue
            
            # Update cache
            self._devices_cache = devices
            self._cache_time = current_time
            
            logger.info(f"Found {len(devices)} audio input devices")
            return devices
            
        except Exception as e:
            logger.error(f"Error enumerating audio devices: {e}")
            return []
    
    def get_default_device(self) -> Optional[AudioDeviceInfo]:
        """Get the default input device"""
        devices = self.get_input_devices()
        for device in devices:
            if device.is_default:
                return device
        
        # Return first device if no default found
        return devices[0] if devices else None
    
    def test_device_compatibility(self, device: AudioDeviceInfo) -> bool:
        """Test if device supports required audio format"""
        try:
            # Test if device supports our required format
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device.index,
                frames_per_buffer=CHUNK_SIZE,
                start=False
            )
            stream.close()
            return True
            
        except Exception as e:
            logger.warning(f"Device {device.name} not compatible: {e}")
            return False


class AudioRecorder:
    """Core PyAudio-based audio recording system"""
    
    def __init__(self, 
                 sample_rate: int = SAMPLE_RATE,
                 channels: int = CHANNELS,
                 chunk_size: int = CHUNK_SIZE,
                 buffer_size: int = 100):
        
        # Audio configuration
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio_format = pyaudio.paInt16
        
        # PyAudio instance
        self.audio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        
        # Device management
        self.device_manager: Optional[AudioDeviceManager] = None
        self.current_device: Optional[AudioDeviceInfo] = None
        
        # Recording state
        self.state = RecordingState.STOPPED
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        
        # Audio monitoring
        self.level_monitor = AudioLevelMonitor()
        
        # Buffer management
        self.audio_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.chunk_counter = 0
        self.start_time = 0.0
        
        # Callbacks
        self.chunk_callback: Optional[Callable[[AudioChunk], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None
        self.level_callback: Optional[Callable[[float, str], None]] = None
        
        # Error handling
        self.last_error = ""
        self.error_count = 0
        
    async def initialize(self) -> bool:
        """Initialize audio system and check permissions"""
        try:
            # Check microphone permissions first
            permission_status = AudioPermissionHandler.check_permission_status()
            
            if permission_status == PermissionStatus.DENIED:
                self.last_error = "Microphone permission denied"
                logger.error(self.last_error)
                return False
            elif permission_status == PermissionStatus.UNKNOWN:
                logger.info("Requesting microphone permission...")
                granted = await AudioPermissionHandler.request_permission()
                if not granted:
                    self.last_error = "Microphone permission not granted"
                    logger.error(self.last_error)
                    return False
            
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            self.device_manager = AudioDeviceManager(self.audio)
            
            # Get default device
            self.current_device = self.device_manager.get_default_device()
            if not self.current_device:
                self.last_error = "No audio input devices found"
                logger.error(self.last_error)
                return False
            
            # Test device compatibility
            if not self.device_manager.test_device_compatibility(self.current_device):
                self.last_error = f"Device {self.current_device.name} not compatible"
                logger.error(self.last_error)
                return False
            
            logger.info(f"Audio system initialized with device: {self.current_device.name}")
            return True
            
        except Exception as e:
            self.last_error = f"Audio initialization failed: {e}"
            logger.error(self.last_error)
            return False
    
    def set_callbacks(self,
                     chunk_callback: Optional[Callable[[AudioChunk], None]] = None,
                     error_callback: Optional[Callable[[str], None]] = None,
                     level_callback: Optional[Callable[[float, str], None]] = None):
        """Set callback functions for audio events"""
        self.chunk_callback = chunk_callback
        self.error_callback = error_callback
        self.level_callback = level_callback
    
    def get_available_devices(self) -> List[AudioDeviceInfo]:
        """Get list of available input devices"""
        if not self.device_manager:
            return []
        return self.device_manager.get_input_devices()
    
    def set_device(self, device: AudioDeviceInfo) -> bool:
        """Set the audio input device"""
        if not self.device_manager:
            return False
        
        # Test compatibility first
        if not self.device_manager.test_device_compatibility(device):
            self.last_error = f"Device {device.name} not compatible"
            return False
        
        # Stop recording if active
        was_recording = self.is_recording
        if was_recording:
            self.stop_recording()
        
        self.current_device = device
        logger.info(f"Audio device set to: {device.name}")
        
        # Resume recording if it was active
        if was_recording:
            self.start_recording()
        
        return True
    
    def start_recording(self) -> bool:
        """Start audio recording"""
        if self.is_recording or not self.current_device:
            return False
        
        try:
            self.state = RecordingState.STARTING
            
            # Create audio stream
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.current_device.index,
                frames_per_buffer=self.chunk_size,
                start=False
            )
            
            # Reset counters and buffers
            self.chunk_counter = 0
            self.start_time = time.time()
            while not self.audio_buffer.empty():
                try:
                    self.audio_buffer.get_nowait()
                except queue.Empty:
                    break
            
            # Start recording thread
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
            self.recording_thread.start()
            
            # Start the stream
            self.stream.start_stream()
            self.state = RecordingState.RECORDING
            
            logger.info("Audio recording started")
            return True
            
        except Exception as e:
            self.last_error = f"Failed to start recording: {e}"
            self.state = RecordingState.ERROR
            if self.error_callback:
                self.error_callback(self.last_error)
            logger.error(self.last_error)
            return False
    
    def stop_recording(self) -> bool:
        """Stop audio recording"""
        if not self.is_recording:
            return True
        
        try:
            self.state = RecordingState.STOPPING
            self.is_recording = False
            
            # Stop and close stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=1.0)
            
            self.state = RecordingState.STOPPED
            logger.info("Audio recording stopped")
            return True
            
        except Exception as e:
            self.last_error = f"Error stopping recording: {e}"
            self.state = RecordingState.ERROR
            if self.error_callback:
                self.error_callback(self.last_error)
            logger.error(self.last_error)
            return False
    
    def _recording_loop(self):
        """Main recording loop (runs in separate thread)"""
        while self.is_recording and self.stream:
            try:
                # Read audio data
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                timestamp = time.time()
                
                # Calculate audio levels
                level_db = self.level_monitor.update(audio_data, self.sample_rate)
                
                # Create audio chunk
                chunk = AudioChunk(
                    data=audio_data,
                    timestamp=timestamp,
                    chunk_index=self.chunk_counter,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    level_db=level_db,
                    is_clipped=self.level_monitor.is_clipping
                )
                
                # Add to buffer (non-blocking)
                try:
                    self.audio_buffer.put_nowait(chunk)
                except queue.Full:
                    # Remove oldest chunk if buffer is full
                    try:
                        self.audio_buffer.get_nowait()
                        self.audio_buffer.put_nowait(chunk)
                    except queue.Empty:
                        pass
                
                # Call callbacks
                if self.chunk_callback:
                    self.chunk_callback(chunk)
                
                if self.level_callback:
                    visual_level = self.level_monitor.get_visual_level()
                    self.level_callback(level_db, visual_level)
                
                self.chunk_counter += 1
                
            except Exception as e:
                if self.is_recording:  # Only log if not intentionally stopped
                    self.last_error = f"Recording error: {e}"
                    self.error_count += 1
                    if self.error_callback:
                        self.error_callback(self.last_error)
                    logger.error(self.last_error)
                break
    
    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[AudioChunk]:
        """Get next audio chunk from buffer"""
        try:
            return self.audio_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get comprehensive status information"""
        device_info = {
            'name': self.current_device.name if self.current_device else 'None',
            'index': self.current_device.index if self.current_device else -1,
        } if self.current_device else {}
        
        return {
            'state': self.state.value,
            'is_recording': self.is_recording,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'chunk_size': self.chunk_size,
            'device': device_info,
            'chunk_count': self.chunk_counter,
            'buffer_size': self.audio_buffer.qsize(),
            'audio_level_db': self.level_monitor.rms_level,
            'peak_level_db': self.level_monitor.peak_level,
            'is_clipping': self.level_monitor.is_clipping,
            'last_error': self.last_error,
            'error_count': self.error_count,
            'uptime': time.time() - self.start_time if self.is_recording else 0
        }
    
    def cleanup(self):
        """Clean up audio resources"""
        logger.info("Cleaning up audio recorder...")
        
        # Stop recording
        self.stop_recording()
        
        # Terminate PyAudio
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.warning(f"Error terminating audio: {e}")
            finally:
                self.audio = None
        
        logger.info("Audio recorder cleanup completed") 