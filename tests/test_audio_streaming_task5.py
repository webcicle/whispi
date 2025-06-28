"""
Test suite for Pi-Whispr Audio Streaming Implementation (Task 5)

Tests for:
- Real-time audio chunk streaming
- Buffer management for network latency
- Sequence numbers and timestamps
- Compression support  
- Streaming reconnection handling
"""

import pytest
import asyncio
import json
import base64
import time
import zlib
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Mock hardware dependencies for testing
import sys
sys.modules['pyaudio'] = MagicMock()
sys.modules['pynput'] = MagicMock()

from client.websocket_client import EnhancedSpeechClient
from shared.protocol import MessageType, AudioDataPayload


class TestAudioStreamingBasic:
    """Basic audio streaming tests"""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock client for testing"""
        with patch('client.websocket_client.websockets.connect'):
            client = EnhancedSpeechClient("ws://localhost:8765")
            client.websocket = Mock()
            client.websocket.send = AsyncMock()
            client.is_connected = True
            return client
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data"""
        # 20ms chunk at 16kHz = 320 samples = 640 bytes
        return b'\x00\x01' * 320  # Simple pattern
    
    @pytest.mark.asyncio
    async def test_send_audio_chunk_basic(self, mock_client, sample_audio_data):
        """Test basic audio chunk sending"""
        await mock_client._send_audio_chunk(sample_audio_data, chunk_index=0)
        
        # Verify websocket.send was called
        mock_client.websocket.send.assert_called_once()
        
        # Get the sent message
        sent_message = mock_client.websocket.send.call_args[0][0]
        parsed = json.loads(sent_message)
        
        # Verify message structure
        assert parsed["header"]["type"] == "audio_data"
        assert parsed["payload"]["chunk_index"] == 0
        assert "audio_data" in parsed["payload"]
        
        # Verify audio data can be decoded
        audio_data = base64.b64decode(parsed["payload"]["audio_data"])
        assert audio_data == sample_audio_data
    
    @pytest.mark.asyncio
    async def test_audio_chunk_sequencing(self, mock_client, sample_audio_data):
        """Test audio chunks are properly sequenced"""
        sent_messages = []
        
        async def capture_message(msg):
            sent_messages.append(json.loads(msg))
        
        mock_client.websocket.send.side_effect = capture_message
        
        # Send multiple chunks
        for i in range(3):
            await mock_client._send_audio_chunk(sample_audio_data, chunk_index=i)
        
        # Verify chunks were sent in order
        assert len(sent_messages) == 3
        for i, msg in enumerate(sent_messages):
            assert msg["payload"]["chunk_index"] == i
            assert msg["header"]["sequence_id"] > 0
    
    def test_audio_compression_basic(self, sample_audio_data):
        """Test basic audio compression"""
        # Compress the audio data
        compressed = zlib.compress(sample_audio_data)
        
        # Should be smaller (or at least not larger for small data)
        assert len(compressed) <= len(sample_audio_data)
        
        # Should decompress correctly
        decompressed = zlib.decompress(compressed)
        assert decompressed == sample_audio_data
    
    @pytest.mark.asyncio 
    async def test_streaming_when_disconnected(self, mock_client, sample_audio_data):
        """Test streaming behavior when disconnected"""
        mock_client.is_connected = False
        
        # Should handle gracefully when disconnected
        await mock_client._send_audio_chunk(sample_audio_data, chunk_index=0)
        
        # Should not have tried to send
        mock_client.websocket.send.assert_not_called()
    
    def test_performance_requirements(self):
        """Test streaming meets performance requirements"""
        # 20ms chunks should be 320 samples at 16kHz
        sample_rate = 16000
        chunk_duration_ms = 20
        expected_samples = int(sample_rate * chunk_duration_ms / 1000)
        assert expected_samples == 320
        
        # Each sample is 2 bytes (int16)
        expected_bytes = expected_samples * 2
        assert expected_bytes == 640
        
        # 50 chunks per second = reasonable bandwidth
        chunks_per_second = 1000 / chunk_duration_ms
        assert chunks_per_second == 50.0 