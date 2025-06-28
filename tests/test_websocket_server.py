"""
Test suite for Pi-Whispr WebSocket Server

This test suite validates the WebSocket server functionality including:
- Connection management and client registration
- Message routing and handling for all protocol message types
- Audio data processing and transcription
- Performance monitoring and metrics collection
- Error handling and recovery
- Concurrent client support
"""

import pytest
import asyncio
import websockets
import json
import base64
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from server.websocket_server import WhisperWebSocketServer
from shared.protocol import (
    MessageType, Priority, ClientStatus,
    MessageHeader, WebSocketMessage,
    AudioConfigPayload, AudioDataPayload, TranscriptionResultPayload,
    PerformanceMetricsPayload, ClientInfoPayload, ErrorPayload,
    MessageBuilder
)
from shared.constants import WEBSOCKET_HOST, WEBSOCKET_PORT, DEFAULT_MODEL


class TestWhisperWebSocketServer:
    """Test cases for WhisperWebSocketServer"""
    
    @pytest.fixture
    def server(self):
        """Create server instance for testing"""
        return WhisperWebSocketServer(host="localhost", port=8766, model_size="tiny.en")
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection"""
        websocket = Mock()
        websocket.remote_address = ("127.0.0.1", 12345)
        websocket.send = AsyncMock()
        websocket.close = AsyncMock()
        return websocket
    
    @pytest.fixture
    def message_builder(self):
        """Create message builder for test messages"""
        return MessageBuilder(client_id="test-client-123", session_id="session-456")
    
    def test_server_initialization(self, server):
        """Test server initialization with correct parameters"""
        assert server.host == "localhost"
        assert server.port == 8766
        assert server.model_size == "tiny.en"
        assert server.stats["total_connections"] == 0
        assert server.stats["active_connections"] == 0
        assert not server._model_loaded
        assert len(server.clients) == 0
    
    @pytest.mark.asyncio
    async def test_model_loading_mock(self, server):
        """Test model loading in mock mode (when faster-whisper not available)"""
        with patch('server.websocket_server.WhisperModel', None):
            await server._load_model()
            assert server._model_loaded
    
    @pytest.mark.asyncio
    async def test_model_loading_success(self, server):
        """Test successful model loading"""
        mock_model = Mock()
        with patch('server.websocket_server.WhisperModel', return_value=mock_model):
            await server._load_model()
            assert server._model_loaded
            assert server.model == mock_model
    
    @pytest.mark.asyncio
    async def test_client_registration(self, server, mock_websocket, message_builder):
        """Test client registration handling"""
        # Create client registration message
        client_info = ClientInfoPayload(
            client_name="Test Client",
            client_version="1.0.0",
            platform="macOS",
            capabilities=["audio_transcription", "real_time"],
            status=ClientStatus.CONNECTED
        )
        
        connect_message = message_builder.connect_message(client_info)
        
        # Handle registration
        await server._handle_client_registration(mock_websocket, connect_message)
        
        # Verify client was registered
        client_id = connect_message.header.client_id
        assert client_id in server.clients
        assert server.clients[client_id]["websocket"] == mock_websocket
        assert server.clients[client_id]["info"].client_name == "Test Client"
        assert server.clients[client_id]["status"] == ClientStatus.CONNECTED
        
        # Verify response was sent
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "connect_ack"
    
    @pytest.mark.asyncio
    async def test_audio_data_processing(self, server, mock_websocket, message_builder):
        """Test audio data message handling and transcription"""
        # First register a client with proper ClientInfoPayload
        server.clients["test-client-123"] = {
            "websocket": mock_websocket,
            "info": ClientInfoPayload(
                client_name="Test Client",
                client_version="1.0.0", 
                platform="macOS",
                capabilities=["audio_transcription"],
                status=ClientStatus.CONNECTED
            ),
            "status": ClientStatus.CONNECTED,
            "last_seen": time.time()
        }
        
        # Mock the model as loaded
        server._model_loaded = True
        
        # Create audio data message
        audio_data = base64.b64encode(b"fake_audio_data").decode('utf-8')
        audio_payload = AudioDataPayload(
            audio_data=audio_data,
            chunk_index=1,
            is_final=True
        )
        
        audio_message = message_builder.audio_data_message(audio_payload)
        
        # Mock the transcription result as async coroutine
        async def mock_transcribe(audio_bytes):
            return {
                "text": "Hello world",
                "confidence": 0.95,
                "language": "en",
                "duration": 1.0
            }
        
        with patch.object(server, '_transcribe_audio', side_effect=mock_transcribe):
            await server._handle_audio_data(mock_websocket, audio_message)
        
        # Verify transcription response was sent
        assert mock_websocket.send.call_count == 1
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "transcription_result"
        assert response_data["payload"]["text"] == "Hello world"
        assert response_data["payload"]["confidence"] == 0.95
    
    @pytest.mark.asyncio
    async def test_ping_handling(self, server, mock_websocket, message_builder):
        """Test ping message handling and pong response"""
        ping_message = message_builder.ping_message()
        
        await server._handle_ping(mock_websocket, ping_message)
        
        # Verify pong response
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "pong"
        assert response_data["header"]["correlation_id"] == str(ping_message.header.sequence_id)
    
    @pytest.mark.asyncio
    async def test_status_request_handling(self, server, mock_websocket, message_builder):
        """Test status request handling"""
        status_message = WebSocketMessage(
            header=MessageHeader(
                message_type=MessageType.STATUS_REQUEST,
                sequence_id=1,
                timestamp=time.time(),
                client_id="test-client-123"
            )
        )
        
        await server._handle_status_request(mock_websocket, status_message)
        
        # Verify status response
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "status_response"
        assert "model_loaded" in response_data["payload"]
        assert "active_connections" in response_data["payload"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, server, mock_websocket):
        """Test error message sending"""
        await server._send_error(mock_websocket, "TEST_ERROR", "Test error message")
        
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "error"
        assert response_data["payload"]["error_code"] == "TEST_ERROR"
        assert response_data["payload"]["error_message"] == "Test error message"
    
    @pytest.mark.asyncio
    async def test_client_list_handling(self, server, mock_websocket, message_builder):
        """Test client list request handling"""
        # Add some test clients with proper ClientInfoPayload objects
        server.clients["client1"] = {
            "websocket": Mock(),
            "info": ClientInfoPayload(
                client_name="Client 1",
                client_version="1.0.0",
                platform="macOS",
                capabilities=["audio_transcription"],
                status=ClientStatus.CONNECTED
            ),
            "status": ClientStatus.CONNECTED,
            "last_seen": time.time()
        }
        server.clients["client2"] = {
            "websocket": Mock(),
            "info": ClientInfoPayload(
                client_name="Client 2",
                client_version="1.0.0", 
                platform="Windows",
                capabilities=["audio_transcription"],
                status=ClientStatus.IDLE
            ),
            "status": ClientStatus.IDLE,
            "last_seen": time.time()
        }
        
        client_list_message = WebSocketMessage(
            header=MessageHeader(
                message_type=MessageType.CLIENT_LIST_REQUEST,
                sequence_id=1,
                timestamp=time.time(),
                client_id="test-client-123"
            )
        )
        
        await server._handle_client_list_request(mock_websocket, client_list_message)
        
        # Verify client list response
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "client_list_response"
        assert len(response_data["payload"]["clients"]) == 2
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, server, mock_websocket):
        """Test performance metrics collection"""
        metrics = server._get_performance_metrics()
        
        assert hasattr(metrics, 'latency_ms')
        assert hasattr(metrics, 'cpu_usage')
        assert hasattr(metrics, 'memory_usage')
        assert metrics.cpu_usage is not None
        assert metrics.memory_usage is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_clients(self, server):
        """Test handling multiple concurrent clients"""
        # Create multiple mock clients
        clients = []
        for i in range(3):
            mock_ws = Mock()
            mock_ws.remote_address = ("127.0.0.1", 12345 + i)
            mock_ws.send = AsyncMock()
            clients.append(mock_ws)
        
        # Register all clients with unique client IDs
        for i, client_ws in enumerate(clients):
            # Create a separate message builder for each client
            message_builder = MessageBuilder(client_id=f"test-client-{i}", session_id=f"session-{i}")
            
            client_info = ClientInfoPayload(
                client_name=f"Client {i}",
                client_version="1.0.0",
                platform="macOS",
                capabilities=["audio_transcription"],
                status=ClientStatus.CONNECTED
            )
            
            connect_message = message_builder.connect_message(client_info)
            
            await server._handle_client_registration(client_ws, connect_message)
        
        # Verify all clients are registered
        assert len(server.clients) == 3
        
        # Test cleanup
        for client_id in list(server.clients.keys()):
            await server._handle_client_disconnect(client_id)
        
        assert len(server.clients) == 0
    
    @pytest.mark.asyncio
    async def test_message_validation(self, server, mock_websocket):
        """Test message validation and error handling"""
        # Test invalid JSON
        await server._handle_message(mock_websocket, "invalid json")
        mock_websocket.send.assert_called()
        
        # Reset mock
        mock_websocket.send.reset_mock()
        
        # Test expired message
        expired_message = WebSocketMessage(
            header=MessageHeader(
                message_type=MessageType.PING,
                sequence_id=1,
                timestamp=time.time() - 100,  # 100 seconds ago
                client_id="test-client",
                ttl=10.0  # 10 second TTL
            )
        )
        
        await server._handle_message(mock_websocket, expired_message.to_json())
        # Should not send any response for expired messages
        mock_websocket.send.assert_not_called()
    
    def test_cpu_memory_monitoring(self, server):
        """Test system resource monitoring"""
        cpu_usage = server._get_cpu_usage()
        memory_usage = server._get_memory_usage()
        
        assert isinstance(cpu_usage, float)
        assert isinstance(memory_usage, float)
        assert 0 <= cpu_usage <= 100
        assert 0 <= memory_usage <= 100


class TestMessageValidatorExtended:
    """Test extended message validation functionality"""
    
    def test_validate_message_json_valid(self):
        """Test validation of valid JSON message"""
        from server.websocket_server import MessageValidator
        
        valid_message = {
            "header": {
                "type": "ping",
                "sequence_id": 1,
                "timestamp": time.time(),
                "client_id": "test-client"
            },
            "payload": {}
        }
        
        assert MessageValidator.validate_message_json(json.dumps(valid_message))
    
    def test_validate_message_json_invalid(self):
        """Test validation of invalid JSON"""
        from server.websocket_server import MessageValidator
        
        assert not MessageValidator.validate_message_json("invalid json")
        assert not MessageValidator.validate_message_json('{"incomplete": ')


@pytest.mark.asyncio
async def test_server_integration():
    """Integration test for basic server functionality"""
    server = WhisperWebSocketServer(host="localhost", port=8767, model_size="tiny.en")
    
    # Test model loading
    await server._load_model()
    assert server._model_loaded
    
    # Test stats initialization
    assert server.stats["total_connections"] == 0
    assert server.stats["active_connections"] == 0
    
    # Test performance metrics
    metrics = server._get_performance_metrics()
    assert metrics.cpu_usage is not None
    assert metrics.memory_usage is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 