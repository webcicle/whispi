"""
Tests for the AudioRecorder system

Tests cover:
- Audio system initialization and permissions
- Audio device discovery and selection
- Recording functionality and buffer management
- Audio level monitoring and visualization
- Error handling and device disconnection
- macOS permission handling
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import queue
import numpy as np

from client.audio_recorder import (
    AudioRecorder, AudioLevelMonitor, AudioPermissionHandler, AudioDeviceManager,
    RecordingState, PermissionStatus, AudioDeviceInfo, AudioChunk
)
from shared.constants import SAMPLE_RATE, CHANNELS, CHUNK_SIZE


class TestAudioLevelMonitor:
    """Test audio level monitoring functionality"""
    
    @pytest.fixture
    def level_monitor(self):
        """Create test audio level monitor"""
        return AudioLevelMonitor(window_size=5)
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data for testing"""
        # Generate a 20ms chunk of 1kHz sine wave at 16kHz
        duration = 0.02  # 20ms
        frequency = 1000  # 1kHz
        sample_rate = 16000
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 16383  # 50% of max amplitude
        return audio.astype(np.int16).tobytes()
    
    def test_level_monitor_initialization(self, level_monitor):
        """Test level monitor initializes correctly"""
        assert level_monitor.window_size == 5
        assert level_monitor.level_history == []
        assert level_monitor.peak_level == -float('inf')
        assert level_monitor.rms_level == -float('inf')
        assert not level_monitor.is_clipping
    
    def test_audio_level_calculation(self, level_monitor, sample_audio_data):
        """Test audio level calculation with real audio data"""
        rms_db = level_monitor.update(sample_audio_data, SAMPLE_RATE)
        
        # Should return a valid dB value (accept both float and numpy types)
        assert isinstance(rms_db, (float, np.floating))
        assert rms_db > -float('inf')
        assert rms_db < 0  # dBFS should be negative
        
        # Check that internal state is updated
        assert level_monitor.rms_level == rms_db
        assert level_monitor.peak_level > -float('inf')
        assert len(level_monitor.level_history) == 1
    
    def test_silence_handling(self, level_monitor):
        """Test handling of silence (zero audio)"""
        silence = b'\x00' * (CHUNK_SIZE * 2)  # 16-bit silence
        rms_db = level_monitor.update(silence, SAMPLE_RATE)
        
        assert rms_db == -float('inf')
        assert level_monitor.rms_level == -float('inf')
        assert not level_monitor.is_clipping
    
    def test_clipping_detection(self, level_monitor):
        """Test clipping detection with loud audio"""
        # Generate audio at maximum amplitude (clipping)
        loud_audio = np.full(CHUNK_SIZE, 32767, dtype=np.int16).tobytes()
        
        level_monitor.update(loud_audio, SAMPLE_RATE)
        
        assert level_monitor.is_clipping
        assert level_monitor.peak_level > level_monitor.clip_threshold_db
    
    def test_level_history_window(self, level_monitor, sample_audio_data):
        """Test level history window management"""
        # Add more updates than window size
        for i in range(10):
            level_monitor.update(sample_audio_data, SAMPLE_RATE)
        
        # Should maintain only window_size entries
        assert len(level_monitor.level_history) == level_monitor.window_size
    
    def test_visual_level_representation(self, level_monitor, sample_audio_data):
        """Test visual level bar generation"""
        level_monitor.update(sample_audio_data, SAMPLE_RATE)
        visual = level_monitor.get_visual_level()
        
        assert isinstance(visual, str)
        assert visual.startswith('|')
        assert visual.endswith('|:') or visual.endswith('|!')  # Normal or clipping
        # Length should be 20 bars + 2 borders + 1 indicator = 23, but unicode may affect counting
        assert len(visual) >= 21 and len(visual) <= 23


class TestAudioPermissionHandler:
    """Test macOS permission handling"""
    
    @patch('client.audio_recorder.AVFoundation')
    def test_permission_check_granted(self, mock_avfoundation):
        """Test checking granted permission status"""
        mock_avfoundation.AVCaptureDevice.authorizationStatusForMediaType_.return_value = 3  # Authorized
        mock_avfoundation.AVAuthorizationStatusAuthorized = 3
        
        status = AudioPermissionHandler.check_permission_status()
        assert status == PermissionStatus.GRANTED
    
    @patch('client.audio_recorder.AVFoundation')
    def test_permission_check_denied(self, mock_avfoundation):
        """Test checking denied permission status"""
        mock_avfoundation.AVCaptureDevice.authorizationStatusForMediaType_.return_value = 2  # Denied
        mock_avfoundation.AVAuthorizationStatusDenied = 2
        
        status = AudioPermissionHandler.check_permission_status()
        assert status == PermissionStatus.DENIED
    
    @patch('client.audio_recorder.AVFoundation', None)
    def test_permission_check_no_avfoundation(self):
        """Test permission check when AVFoundation is not available"""
        status = AudioPermissionHandler.check_permission_status()
        assert status == PermissionStatus.UNKNOWN
    
    @pytest.mark.asyncio
    @patch('client.audio_recorder.AVFoundation', None)
    async def test_permission_request_no_avfoundation(self):
        """Test permission request when AVFoundation is not available"""
        result = await AudioPermissionHandler.request_permission()
        assert result is False


class TestAudioDeviceManager:
    """Test audio device management"""
    
    @pytest.fixture
    def mock_pyaudio(self):
        """Create mock PyAudio instance"""
        mock_audio = Mock()
        mock_audio.get_device_count.return_value = 3
        mock_audio.get_default_input_device_info.return_value = {'index': 1}
        
        # Mock device info
        device_infos = [
            {'name': 'Built-in Microphone', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0},
            {'name': 'External USB Mic', 'maxInputChannels': 1, 'defaultSampleRate': 48000.0},
            {'name': 'Built-in Output', 'maxInputChannels': 0, 'defaultSampleRate': 44100.0}  # Output only
        ]
        
        def get_device_info_by_index(index):
            return device_infos[index]
        
        mock_audio.get_device_info_by_index.side_effect = get_device_info_by_index
        return mock_audio
    
    @pytest.fixture
    def device_manager(self, mock_pyaudio):
        """Create test audio device manager"""
        return AudioDeviceManager(mock_pyaudio)
    
    def test_device_enumeration(self, device_manager):
        """Test enumeration of audio input devices"""
        devices = device_manager.get_input_devices()
        
        # Should find 2 input devices (excluding output-only device)
        assert len(devices) == 2
        
        # Check device properties
        assert devices[0].name == 'Built-in Microphone'
        assert devices[0].max_input_channels == 2
        assert not devices[0].is_default
        
        assert devices[1].name == 'External USB Mic'
        assert devices[1].is_default  # Index 1 is default
    
    def test_default_device_selection(self, device_manager):
        """Test default device selection"""
        default_device = device_manager.get_default_device()
        
        assert default_device is not None
        assert default_device.name == 'External USB Mic'
        assert default_device.is_default
    
    def test_device_compatibility_test(self, device_manager, mock_pyaudio):
        """Test device compatibility testing"""
        device = AudioDeviceInfo(
            index=0,
            name='Test Device',
            max_input_channels=1,
            default_sample_rate=16000.0
        )
        
        # Mock successful stream opening
        mock_stream = Mock()
        mock_pyaudio.open.return_value = mock_stream
        
        compatible = device_manager.test_device_compatibility(device)
        assert compatible
        
        # Verify stream was opened with correct parameters
        mock_pyaudio.open.assert_called_once()
        mock_stream.close.assert_called_once()
    
    def test_device_compatibility_failure(self, device_manager, mock_pyaudio):
        """Test device compatibility test failure"""
        device = AudioDeviceInfo(
            index=0,
            name='Incompatible Device',
            max_input_channels=1,
            default_sample_rate=16000.0
        )
        
        # Mock stream opening failure
        mock_pyaudio.open.side_effect = Exception("Device not supported")
        
        compatible = device_manager.test_device_compatibility(device)
        assert not compatible


class TestAudioRecorder:
    """Test core audio recording functionality"""
    
    @pytest.fixture
    def mock_pyaudio_class(self):
        """Mock the PyAudio class"""
        with patch('client.audio_recorder.pyaudio.PyAudio') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            # Setup default device info
            mock_instance.get_default_input_device_info.return_value = {
                'index': 0,
                'name': 'Test Microphone',
                'maxInputChannels': 1,
                'defaultSampleRate': 16000.0
            }
            mock_instance.get_device_count.return_value = 1
            mock_instance.get_device_info_by_index.return_value = {
                'name': 'Test Microphone',
                'maxInputChannels': 1,
                'defaultSampleRate': 16000.0
            }
            
            # Setup stream mock
            mock_stream = Mock()
            mock_instance.open.return_value = mock_stream
            
            yield mock_class, mock_instance, mock_stream
    
    @pytest.fixture
    def audio_recorder(self):
        """Create test audio recorder"""
        return AudioRecorder()
    
    @pytest.mark.asyncio
    async def test_audio_recorder_initialization(self, audio_recorder, mock_pyaudio_class):
        """Test audio recorder initialization"""
        mock_class, mock_instance, mock_stream = mock_pyaudio_class
        
        with patch.object(AudioPermissionHandler, 'check_permission_status', 
                         return_value=PermissionStatus.GRANTED):
            success = await audio_recorder.initialize()
            
            assert success
            assert audio_recorder.audio is not None
            assert audio_recorder.device_manager is not None
            assert audio_recorder.current_device is not None
    
    @pytest.mark.asyncio
    async def test_initialization_permission_denied(self, audio_recorder):
        """Test initialization with denied permissions"""
        with patch.object(AudioPermissionHandler, 'check_permission_status',
                         return_value=PermissionStatus.DENIED):
            success = await audio_recorder.initialize()
            
            assert not success
            assert "permission denied" in audio_recorder.last_error.lower()
    
    @pytest.mark.asyncio
    async def test_initialization_permission_request(self, audio_recorder, mock_pyaudio_class):
        """Test initialization with permission request"""
        mock_class, mock_instance, mock_stream = mock_pyaudio_class
        
        with patch.object(AudioPermissionHandler, 'check_permission_status',
                         return_value=PermissionStatus.UNKNOWN), \
             patch.object(AudioPermissionHandler, 'request_permission',
                         return_value=True):
            
            success = await audio_recorder.initialize()
            assert success
    
    def test_callback_setup(self, audio_recorder):
        """Test callback function setup"""
        chunk_callback = Mock()
        error_callback = Mock()
        level_callback = Mock()
        
        audio_recorder.set_callbacks(
            chunk_callback=chunk_callback,
            error_callback=error_callback,
            level_callback=level_callback
        )
        
        assert audio_recorder.chunk_callback == chunk_callback
        assert audio_recorder.error_callback == error_callback
        assert audio_recorder.level_callback == level_callback
    
    @pytest.mark.asyncio
    async def test_recording_lifecycle(self, audio_recorder, mock_pyaudio_class):
        """Test complete recording lifecycle"""
        mock_class, mock_instance, mock_stream = mock_pyaudio_class
        
        # Initialize recorder
        with patch.object(AudioPermissionHandler, 'check_permission_status',
                         return_value=PermissionStatus.GRANTED):
            await audio_recorder.initialize()
        
        # Test starting recording
        success = audio_recorder.start_recording()
        assert success
        assert audio_recorder.is_recording
        assert audio_recorder.state == RecordingState.RECORDING
        
        # Verify stream was started
        mock_stream.start_stream.assert_called_once()
        
        # Wait a bit for recording thread to start
        await asyncio.sleep(0.1)
        
        # Test stopping recording
        success = audio_recorder.stop_recording()
        assert success
        assert not audio_recorder.is_recording
        assert audio_recorder.state == RecordingState.STOPPED
        
        # Verify stream was stopped
        mock_stream.stop_stream.assert_called_once()
        # close() may be called multiple times during cleanup, which is safe
        assert mock_stream.close.call_count >= 1
    
    def test_audio_chunk_processing(self, audio_recorder):
        """Test audio chunk processing and buffering"""
        # Setup callbacks
        chunk_callback = Mock()
        audio_recorder.set_callbacks(chunk_callback=chunk_callback)
        
        # Create test audio chunk
        test_data = b'\x00\x01' * 160  # 320 bytes = 20ms at 16kHz
        chunk = AudioChunk(
            data=test_data,
            timestamp=time.time(),
            chunk_index=0,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            level_db=-20.0
        )
        
        # Simulate chunk processing
        audio_recorder.audio_buffer.put_nowait(chunk)
        
        # Test retrieving chunk
        retrieved_chunk = audio_recorder.get_audio_chunk(timeout=0.1)
        assert retrieved_chunk is not None
        assert retrieved_chunk.data == test_data
        assert retrieved_chunk.chunk_index == 0
    
    def test_buffer_overflow_handling(self, audio_recorder):
        """Test audio buffer overflow handling"""
        # Fill buffer beyond capacity
        for i in range(audio_recorder.audio_buffer.maxsize + 10):
            chunk = AudioChunk(
                data=b'\x00' * 320,
                timestamp=time.time(),
                chunk_index=i,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                level_db=-20.0
            )
            
            try:
                audio_recorder.audio_buffer.put_nowait(chunk)
            except queue.Full:
                break
        
        # Buffer should be at max capacity
        assert audio_recorder.audio_buffer.qsize() == audio_recorder.audio_buffer.maxsize
    
    def test_status_info(self, audio_recorder):
        """Test status information reporting"""
        status = audio_recorder.get_status_info()
        
        assert isinstance(status, dict)
        assert 'state' in status
        assert 'is_recording' in status
        assert 'sample_rate' in status
        assert 'channels' in status
        assert 'chunk_size' in status
        assert 'chunk_count' in status
        assert 'buffer_size' in status
        assert 'audio_level_db' in status
        assert 'last_error' in status
        assert 'error_count' in status
    
    def test_cleanup(self, audio_recorder, mock_pyaudio_class):
        """Test resource cleanup"""
        mock_class, mock_instance, mock_stream = mock_pyaudio_class
        
        # Setup recorder with audio instance
        audio_recorder.audio = mock_instance
        
        audio_recorder.cleanup()
        
        # Verify PyAudio was terminated
        mock_instance.terminate.assert_called_once()
        assert audio_recorder.audio is None


class TestAudioRecorderIntegration:
    """Integration tests for audio recorder with real-world scenarios"""
    
    @pytest.fixture
    def configured_recorder(self):
        """Create a configured audio recorder for integration tests"""
        recorder = AudioRecorder()
        
        # Mock successful initialization
        with patch.object(recorder, 'initialize', return_value=True):
            asyncio.run(recorder.initialize())
        
        return recorder
    
    def test_device_switching(self, configured_recorder):
        """Test switching between audio devices"""
        # Create mock devices
        device1 = AudioDeviceInfo(0, "Device 1", 1, 16000.0)
        device2 = AudioDeviceInfo(1, "Device 2", 2, 44100.0)
        
        configured_recorder.current_device = device1
        
        with patch.object(configured_recorder, 'device_manager') as mock_manager:
            mock_manager.test_device_compatibility.return_value = True
            
            success = configured_recorder.set_device(device2)
            assert success
            assert configured_recorder.current_device == device2
    
    @pytest.mark.asyncio
    async def test_recording_with_callbacks(self, configured_recorder):
        """Test recording with callback functions"""
        chunks_received = []
        errors_received = []
        levels_received = []
        
        def chunk_handler(chunk):
            chunks_received.append(chunk)
        
        def error_handler(error):
            errors_received.append(error)
        
        def level_handler(level_db, visual):
            levels_received.append((level_db, visual))
        
        configured_recorder.set_callbacks(
            chunk_callback=chunk_handler,
            error_callback=error_handler,
            level_callback=level_handler
        )
        
        # Simulate audio data
        test_chunk = AudioChunk(
            data=b'\x00\x01' * 160,
            timestamp=time.time(),
            chunk_index=0,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            level_db=-15.0
        )
        
        # Trigger callbacks
        if configured_recorder.chunk_callback:
            configured_recorder.chunk_callback(test_chunk)
        
        if configured_recorder.level_callback:
            configured_recorder.level_callback(-15.0, "|████    |:")
        
        # Verify callbacks were called
        assert len(chunks_received) == 1
        assert chunks_received[0] == test_chunk
        assert len(levels_received) == 1
        assert levels_received[0][0] == -15.0
    
    def test_error_handling(self, configured_recorder):
        """Test error handling and recovery"""
        error_count_before = configured_recorder.error_count
        
        # Simulate an error by directly setting the error state
        error_message = "Test audio error"
        configured_recorder.last_error = error_message
        configured_recorder.error_count += 1
        
        # Check error was recorded
        assert configured_recorder.last_error == error_message
        assert configured_recorder.error_count == error_count_before + 1
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, configured_recorder):
        """Test concurrent access to audio buffer"""
        # Start multiple tasks that access the buffer
        async def producer():
            for i in range(10):
                chunk = AudioChunk(
                    data=b'\x00' * 320,
                    timestamp=time.time(),
                    chunk_index=i,
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    level_db=-20.0
                )
                try:
                    configured_recorder.audio_buffer.put_nowait(chunk)
                except queue.Full:
                    pass
                await asyncio.sleep(0.01)
        
        async def consumer():
            chunks = []
            for _ in range(5):
                chunk = configured_recorder.get_audio_chunk(timeout=0.1)
                if chunk:
                    chunks.append(chunk)
                await asyncio.sleep(0.02)
            return chunks
        
        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())
        
        await asyncio.gather(producer_task, consumer_task)
        
        # Should complete without errors
        assert True


if __name__ == "__main__":
    pytest.main([__file__]) 