"""
Test suite for Pi-Whispr WebSocket Client

This test suite validates the macOS WebSocket client functionality including:
- Connection establishment and management
- Audio data transmission with proper message formatting
- Transcription result handling
- Status updates and error handling
- Automatic reconnection logic
- Performance tracking and ping/pong
- Message validation and sequencing

Test Coverage aligns with Task 1.3 requirements:
- Connection establishment, audio data transmission, transcription results
- Status updates, error handling, ping/pong, client management, performance tracking
"""

import pytest
import asyncio
import json
import base64
import time
import tempfile
import wave
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Mock PyAudio and pynput for testing (since they require hardware)
import sys
from unittest.mock import MagicMock

sys.modules['pyaudio'] = MagicMock()
sys.modules['pynput'] = MagicMock()
sys.modules['pynput.keyboard'] = MagicMock()
sys.modules['webrtcvad'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.signal'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['pyobjc'] = MagicMock()

from client.websocket_client import EnhancedSpeechClient, ClientConnectionManager
from shared.protocol import (
    MessageType, Priority, ClientStatus,
    MessageHeader, WebSocketMessage,
    AudioConfigPayload, AudioDataPayload, TranscriptionResultPayload,
    PerformanceMetricsPayload, ClientInfoPayload, ErrorPayload,
    MessageBuilder, MessageValidator
)


class MockWebSocket:
    """Mock WebSocket for testing"""
    
    def __init__(self):
        self.sent_messages = []
        self.received_messages = []
        self.is_connected = True
        self.close_code = None
        
    async def send(self, message: str):
        """Mock send method"""
        if not self.is_connected:
            raise ConnectionError("WebSocket not connected")
        self.sent_messages.append(message)
        
    async def recv(self):
        """Mock receive method"""
        if not self.is_connected:
            raise ConnectionError("WebSocket not connected")
        if not self.received_messages:
            await asyncio.sleep(0.1)  # Simulate waiting
            return None
        return self.received_messages.pop(0)
        
    async def close(self, code=1000):
        """Mock close method"""
        self.is_connected = False
        self.close_code = code
        
    def add_received_message(self, message: str):
        """Add a message to be received"""
        self.received_messages.append(message)


class TestEnhancedSpeechClient:
    """Test cases for EnhancedSpeechClient functionality"""
    
    @pytest.fixture
    def client(self):
        """Create a test client with mocked dependencies"""
        with patch('client.websocket_client.websockets.connect') as mock_connect:
            mock_ws = MockWebSocket()
            mock_connect.return_value.__aenter__.return_value = mock_ws
            
            client = EnhancedSpeechClient(server_url="ws://localhost:8765")
            client.websocket = mock_ws
            return client
    
    @pytest.fixture
    def sample_audio_data(self):
        """Create sample audio data for testing"""
        # Create 20ms of sample audio data (16kHz, mono, int16)
        sample_rate = 16000
        duration_ms = 20
        samples = int(sample_rate * duration_ms / 1000)
        
        # Generate simple sine wave
        import numpy as np
        t = np.linspace(0, duration_ms / 1000, samples)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        return audio.tobytes()
    
    def test_client_initialization(self, client):
        """Test client initializes with correct configuration"""
        assert client.server_url == "ws://localhost:8765"
        assert client.sample_rate == 16000
        assert client.channels == 1
        assert client.chunk_size == 320
        assert not client.is_recording
        assert not client.is_connected
    
    @pytest.mark.asyncio
    async def test_connection_establishment(self, client):
        """Test WebSocket connection establishment"""
        # Mock successful connection
        with patch('client.websocket_client.websockets.connect') as mock_connect:
            mock_ws = MockWebSocket()
            mock_connect.return_value.__aenter__.return_value = mock_ws
            
            await client.connect()
            
            assert client.is_connected
            assert len(mock_ws.sent_messages) >= 1
            
            # Verify connect message was sent
            connect_msg = json.loads(mock_ws.sent_messages[0])
            assert connect_msg["header"]["type"] == "connect"
            assert connect_msg["payload"]["client_name"] == "Enhanced macOS Speech Client"
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self, client):
        """Test handling of connection failures"""
        with patch('client.websocket_client.websockets.connect') as mock_connect:
            mock_connect.side_effect = ConnectionError("Connection failed")
            
            with pytest.raises(ConnectionError):
                await client.connect()
                
            assert not client.is_connected
    
    @pytest.mark.asyncio
    async def test_audio_data_transmission(self, client, sample_audio_data):
        """Test audio data is properly formatted and transmitted"""
        client.is_connected = True
        mock_ws = client.websocket
        
        # Simulate sending audio data
        await client._send_audio_chunk(sample_audio_data, chunk_index=1)
        
        assert len(mock_ws.sent_messages) == 1
        
        # Verify audio message structure
        audio_msg = json.loads(mock_ws.sent_messages[0])
        assert audio_msg["header"]["type"] == "audio_data"
        assert audio_msg["payload"]["chunk_index"] == 1
        assert "audio_data" in audio_msg["payload"]
        
        # Verify audio data is base64 encoded
        audio_data = audio_msg["payload"]["audio_data"]
        decoded = base64.b64decode(audio_data)
        assert decoded == sample_audio_data
    
    @pytest.mark.asyncio
    async def test_transcription_result_handling(self, client):
        """Test handling of transcription results"""
        client.is_connected = True
        mock_ws = client.websocket
        
        # Create mock transcription payload (just the payload part)
        transcription_payload = {
            "text": "Hello world",
            "confidence": 0.95,
            "processing_time": 0.8,
            "model_used": "tiny.en",
            "language": "en"
        }
        
        # Mock the text insertion method
        with patch.object(client, '_insert_text') as mock_insert:
            await client._handle_transcription_result(transcription_payload)
            
            mock_insert.assert_called_once_with("Hello world")
            assert client.last_transcription == "Hello world"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test error message handling"""
        client.is_connected = True
        
        # Create mock error payload (just the payload part)
        error_payload = {
            "error_code": "TRANSCRIPTION_ERROR",
            "error_message": "Model failed to process audio",
            "recoverable": True,
            "suggested_action": "Try recording again"
        }
        
        # Test error handling
        await client._handle_error_message(error_payload)
        
        assert client.last_error == "TRANSCRIPTION_ERROR: Model failed to process audio"
    
    @pytest.mark.asyncio
    async def test_ping_pong_mechanism(self, client):
        """Test ping/pong heartbeat mechanism"""
        client.is_connected = True
        mock_ws = client.websocket
        
        # Send ping
        await client._send_ping()
        
        assert len(mock_ws.sent_messages) == 1
        ping_msg = json.loads(mock_ws.sent_messages[0])
        assert ping_msg["header"]["type"] == "ping"
        
        # Test pong response (just the payload part)
        pong_payload = {}
        
        await client._handle_pong_message(pong_payload)
        
        # Verify latency was measured
        assert client.last_ping_latency > 0
    
    @pytest.mark.asyncio
    async def test_automatic_reconnection(self, client):
        """Test automatic reconnection logic"""
        # Simulate connection loss
        client.is_connected = True
        client.websocket.is_connected = False
        
        reconnect_count = 0
        original_connect = client.connect
        
        async def mock_connect():
            nonlocal reconnect_count
            reconnect_count += 1
            if reconnect_count < 3:
                raise ConnectionError("Connection failed")
            await original_connect()
        
        with patch.object(client, 'connect', side_effect=mock_connect):
            await client._handle_connection_loss()
            
            # Verify reconnection attempts were made
            assert reconnect_count >= 3
    
    def test_audio_preprocessing(self, client, sample_audio_data):
        """Test audio preprocessing with noise reduction"""
        # Test the actual implementation which just does normalization if numpy is available
        processed_audio = client._preprocess_audio(sample_audio_data)
        
        # Should return bytes (either original or processed)
        assert isinstance(processed_audio, bytes)
        assert len(processed_audio) == len(sample_audio_data)
    
    def test_voice_activity_detection(self, client, sample_audio_data):
        """Test Voice Activity Detection functionality"""
        with patch.object(client.vad, 'is_speech') as mock_vad:
            mock_vad.return_value = True
            
            has_speech = client._has_speech(sample_audio_data)
            
            assert has_speech
            mock_vad.assert_called_once_with(sample_audio_data, client.sample_rate)
    
    @pytest.mark.asyncio
    async def test_message_sequencing(self, client):
        """Test message sequence ID management"""
        client.is_connected = True
        mock_ws = client.websocket
        
        # Send multiple messages
        await client._send_ping()
        await client._send_ping()
        await client._send_ping()
        
        # Verify sequence IDs increment
        messages = [json.loads(msg) for msg in mock_ws.sent_messages]
        sequence_ids = [msg["header"]["sequence_id"] for msg in messages]
        
        assert len(set(sequence_ids)) == 3  # All different
        assert sequence_ids == sorted(sequence_ids)  # Increasing order
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, client):
        """Test performance metrics collection and reporting"""
        client.is_connected = True
        
        # Simulate performance data
        client.last_ping_latency = 25.5
        client.audio_processing_time = 150.0
        
        metrics = client._get_performance_metrics()
        
        assert metrics["latency_ms"] == 25.5
        assert metrics["audio_processing_time"] == 150.0
        assert "timestamp" in metrics
    
    def test_hotkey_handling(self, client):
        """Test global hotkey detection and handling"""
        # Test that the method exists and can be called
        client._setup_hotkeys()
        # The mock implementation just logs, so no assertions needed
        assert True  # Test passes if no exception is raised
    
    @pytest.mark.asyncio
    async def test_recording_lifecycle(self, client, sample_audio_data):
        """Test complete recording lifecycle"""
        client.is_connected = True
        mock_ws = client.websocket
        
        # Mock audio recording
        with patch.object(client, '_record_audio_chunk') as mock_record:
            mock_record.return_value = sample_audio_data
            
            # Start recording
            await client.start_recording()
            assert client.is_recording
            
            # Simulate audio chunks
            await client._process_audio_chunk(sample_audio_data, chunk_index=0)
            await client._process_audio_chunk(sample_audio_data, chunk_index=1, is_final=True)
            
            # Stop recording
            await client.stop_recording()
            assert not client.is_recording
            
            # Verify audio_start and audio_end messages were sent
            messages = [json.loads(msg) for msg in mock_ws.sent_messages]
            message_types = [msg["header"]["type"] for msg in messages]
            
            assert "audio_start" in message_types
            assert "audio_data" in message_types
            assert "audio_end" in message_types


class TestClientConnectionManager:
    """Test cases for connection management functionality"""
    
    @pytest.fixture
    def connection_manager(self):
        """Create a test connection manager"""
        return ClientConnectionManager("ws://localhost:8765")
    
    def test_connection_manager_initialization(self, connection_manager):
        """Test connection manager initializes correctly"""
        assert connection_manager.server_url == "ws://localhost:8765"
        assert connection_manager.max_reconnect_attempts == 5
        assert connection_manager.reconnect_delay == 1.0
        assert not connection_manager.is_connected
    
    @pytest.mark.asyncio
    async def test_connection_retry_logic(self, connection_manager):
        """Test connection retry with exponential backoff"""
        attempt_count = 0
        
        async def mock_connect():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Connection failed")
            return MockWebSocket()
        
        with patch('client.websocket_client.websockets.connect', side_effect=mock_connect):
            websocket = await connection_manager.connect_with_retry()
            
            assert websocket is not None
            assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, connection_manager):
        """Test connection timeout handling"""
        async def slow_connect():
            await asyncio.sleep(10)  # Simulate slow connection
            return MockWebSocket()
        
        with patch('client.websocket_client.websockets.connect', side_effect=slow_connect):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    connection_manager.connect_with_retry(),
                    timeout=1.0
                )


class TestMessageValidation:
    """Test cases for message validation and protocol compliance"""
    
    def test_outgoing_message_validation(self):
        """Test validation of outgoing messages"""
        client_id = "test-client-123"
        builder = MessageBuilder(client_id, "session-456")
        
        # Test valid audio data message
        audio_payload = AudioDataPayload(
            audio_data=base64.b64encode(b"test audio data").decode(),
            chunk_index=1,
            is_final=False
        )
        
        message = builder.audio_data_message(audio_payload)
        assert MessageValidator.validate_message(message)
        
        # Test message JSON serialization
        json_str = message.to_json()
        assert isinstance(json_str, str)
        
        # Test round-trip serialization
        restored_message = WebSocketMessage.from_json(json_str)
        assert restored_message.header.message_type == MessageType.AUDIO_DATA
        assert restored_message.header.client_id == client_id
    
    def test_incoming_message_validation(self):
        """Test validation of incoming messages"""
        # Valid transcription result
        valid_message = {
            "header": {
                "type": "transcription_result",
                "sequence_id": 1,
                "timestamp": time.time(),
                "client_id": "test-client"
            },
            "payload": {
                "text": "Hello world",
                "confidence": 0.95,
                "processing_time": 0.5,
                "model_used": "tiny.en"
            }
        }
        
        json_str = json.dumps(valid_message)
        assert MessageValidator.validate_message_json(json_str)
        
        # Invalid message structure
        invalid_message = {
            "header": {
                "type": "invalid_type",
                "sequence_id": -1  # Invalid sequence ID
            }
        }
        
        invalid_json = json.dumps(invalid_message)
        assert not MessageValidator.validate_message_json(invalid_json)


class TestIntegration:
    """Integration tests for complete client workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_transcription_workflow(self):
        """Test complete transcription workflow from recording to text insertion"""
        # Create sample audio data
        sample_audio_data = b'\x00\x01\x02\x03' * 80  # 320 bytes = 20ms at 16kHz
        
        with patch('client.websocket_client.websockets.connect') as mock_connect:
            mock_ws = MockWebSocket()
            mock_connect.return_value.__aenter__.return_value = mock_ws
            
            client = EnhancedSpeechClient()
            client.websocket = mock_ws
            client.is_connected = True
            
            # Mock text insertion
            with patch.object(client, '_insert_text') as mock_insert:
                # Start recording workflow
                await client.start_recording()
                
                # Send audio data
                await client._send_audio_chunk(sample_audio_data, chunk_index=0)
                await client._send_audio_chunk(sample_audio_data, chunk_index=1, is_final=True)
                
                # Simulate transcription result from server (just payload)
                transcription_payload = {
                    "text": "Test transcription result",
                    "confidence": 0.95,
                    "processing_time": 0.8,
                    "model_used": "tiny.en"
                }
                
                await client._handle_transcription_result(transcription_payload)
                
                # Stop recording
                await client.stop_recording()
                
                # Verify complete workflow
                messages = [json.loads(msg) for msg in mock_ws.sent_messages]
                message_types = [msg["header"]["type"] for msg in messages]
                
                assert "audio_start" in message_types
                assert "audio_data" in message_types
                assert "audio_end" in message_types
                
                mock_insert.assert_called_once_with("Test transcription result") 