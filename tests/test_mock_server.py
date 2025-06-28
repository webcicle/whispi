"""
Test suite for Mock WebSocket Server Core (Task 2.2)

This test suite validates the mock server implementation including:
- Complete WebSocket protocol implementation matching the real server
- All message types: connection, audio, transcription, status, error, ping/pong, client management, performance tracking
- Protocol validation and message handling
- Mock transcription responses with realistic delays
- Client registration and management
- Error handling and recovery
- Performance metrics and monitoring
- Concurrent client support
"""

import pytest
import asyncio
import websockets
import json
import base64
import time
import random
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from server.mock_server import MockWhisperServer
from shared.protocol import (
    MessageType, Priority, ClientStatus,
    MessageHeader, WebSocketMessage,
    AudioConfigPayload, AudioDataPayload, TranscriptionResultPayload,
    PerformanceMetricsPayload, ClientInfoPayload, ErrorPayload,
    MessageBuilder, MessageValidator
)
from shared.constants import WEBSOCKET_HOST, WEBSOCKET_PORT, DEFAULT_MODEL


class TestMockWhisperServer:
    """Test cases for MockWhisperServer implementation of task 2.2"""
    
    @pytest.fixture
    def mock_server(self):
        """Create mock server instance for testing"""
        return MockWhisperServer(host="localhost", port=8767, latency_ms=50, model_size="tiny.en")
    
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
        return MessageBuilder(client_id="test-mock-client", session_id="mock-session-123")
    
    def test_mock_server_initialization(self, mock_server):
        """Test mock server initialization with correct parameters"""
        assert mock_server.host == "localhost"
        assert mock_server.port == 8767
        assert mock_server.latency_ms == 50
        assert mock_server.model_size == "tiny.en"
        assert mock_server._model_loaded is True  # Mock server always has model loaded
        assert mock_server.stats["total_connections"] == 0
        assert mock_server.stats["active_connections"] == 0
        assert mock_server.stats["total_transcriptions"] == 0
        assert len(mock_server.clients) == 0
        assert len(mock_server.mock_transcriptions) > 0
    
    @pytest.mark.asyncio
    async def test_message_validation_integration(self, mock_server, mock_websocket):
        """Test message validation using proper protocol"""
        # Test valid message handling
        valid_message = {
            "header": {
                "type": "ping",
                "sequence_id": 1,
                "timestamp": time.time(),
                "client_id": "test-client",
                "priority": "normal"
            },
            "payload": {}
        }
        
        await mock_server._handle_message(mock_websocket, json.dumps(valid_message))
        
        # Should have sent a pong response
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_message_validation_invalid_json(self, mock_server, mock_websocket):
        """Test handling of invalid JSON messages"""
        invalid_json = '{"invalid": json syntax'
        
        await mock_server._handle_message(mock_websocket, invalid_json)
        
        # Should send error response
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "error"
        assert response_data["payload"]["error_code"] == "INVALID_JSON"
    
    @pytest.mark.asyncio
    async def test_message_validation_invalid_structure(self, mock_server, mock_websocket):
        """Test handling of messages with invalid structure"""
        invalid_message = {"some": "invalid structure"}
        
        await mock_server._handle_message(mock_websocket, json.dumps(invalid_message))
        
        # Should send validation error
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "error"
        assert response_data["payload"]["error_code"] == "INVALID_MESSAGE"
    
    @pytest.mark.asyncio
    async def test_client_registration_complete_flow(self, mock_server, mock_websocket, message_builder):
        """Test complete client registration flow using proper protocol"""
        # Create client registration message
        client_info = ClientInfoPayload(
            client_name="Mock Test Client",
            client_version="1.0.0-test",
            platform="pytest",
            capabilities=["audio_transcription", "real_time", "mock_testing"],
            status=ClientStatus.CONNECTED
        )
        
        connect_message = message_builder.connect_message(client_info)
        
        # Handle registration
        await mock_server._handle_client_registration(mock_websocket, connect_message)
        
        # Verify client was registered with correct structure
        client_id = connect_message.header.client_id
        assert client_id in mock_server.clients
        client_data = mock_server.clients[client_id]
        
        assert client_data["websocket"] == mock_websocket
        assert client_data["info"].client_name == "Mock Test Client"
        assert client_data["info"].platform == "pytest"
        assert client_data["status"] == ClientStatus.CONNECTED
        assert "connected_at" in client_data
        assert "last_seen" in client_data
        assert client_data["session_id"] == connect_message.header.session_id
        
        # Verify acknowledgment response was sent
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "connect_ack"
        assert response_data["header"]["correlation_id"] == connect_message.header.correlation_id
        assert response_data["payload"]["status"] == "connected"
        assert response_data["payload"]["client_id"] == client_id
        assert "mock-whisper-tiny" in response_data["payload"]["model_name"]
        assert "transcription" in response_data["payload"]["server_capabilities"]
    
    @pytest.mark.asyncio
    async def test_audio_data_processing_complete_flow(self, mock_server, mock_websocket, message_builder):
        """Test complete audio data processing with mock transcription"""
        # First register a client
        client_info = ClientInfoPayload(
            client_name="Audio Test Client",
            client_version="1.0.0",
            platform="pytest",
            capabilities=["audio_transcription"],
            status=ClientStatus.CONNECTED
        )
        
        client_id = message_builder.client_id
        mock_server.clients[client_id] = {
            "websocket": mock_websocket,
            "info": client_info,
            "status": ClientStatus.CONNECTED,
            "connected_at": time.time(),
            "last_seen": time.time(),
            "session_id": message_builder.session_id
        }
        
        # Create audio data message with final chunk
        audio_data = base64.b64encode(b"mock_audio_data_for_transcription").decode('utf-8')
        audio_payload = AudioDataPayload(
            audio_data=audio_data,
            chunk_index=5,
            is_final=True,
            timestamp_offset=2.5,
            energy_level=0.75
        )
        
        audio_message = message_builder.audio_data_message(audio_payload)
        
        # Track initial stats
        initial_transcriptions = mock_server.stats["total_transcriptions"]
        
        # Handle audio data
        start_time = time.time()
        await mock_server._handle_audio_data(mock_websocket, audio_message)
        processing_time = time.time() - start_time
        
        # Verify processing occurred with appropriate delay
        assert processing_time >= 0.1  # Should have some processing delay
        
        # Verify transcription was counted
        assert mock_server.stats["total_transcriptions"] == initial_transcriptions + 1
        
        # Verify response was sent
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        
        # Validate transcription result structure
        assert response_data["header"]["type"] == "transcription_result"
        assert response_data["header"]["correlation_id"] == audio_message.header.correlation_id
        
        payload = response_data["payload"]
        assert "text" in payload
        assert payload["text"] in mock_server.mock_transcriptions
        assert "confidence" in payload
        assert 0.85 <= payload["confidence"] <= 0.98
        assert "processing_time" in payload
        assert payload["model_used"] == "mock-whisper-tiny.en"
        assert payload["language"] == "en"
        assert "audio_duration" in payload
        assert payload["is_partial"] is False
        
        # Verify client status was updated correctly
        assert mock_server.clients[client_id]["status"] == ClientStatus.CONNECTED
    
    @pytest.mark.asyncio
    async def test_audio_data_non_final_chunks(self, mock_server, mock_websocket, message_builder):
        """Test that non-final audio chunks don't trigger transcription"""
        # Register client
        client_id = message_builder.client_id
        mock_server.clients[client_id] = {
            "websocket": mock_websocket,
            "info": ClientInfoPayload("Test", "1.0", "test", [], ClientStatus.CONNECTED),
            "status": ClientStatus.CONNECTED,
            "connected_at": time.time(),
            "last_seen": time.time(),
            "session_id": message_builder.session_id
        }
        
        # Send non-final chunk
        audio_payload = AudioDataPayload(
            audio_data=base64.b64encode(b"partial_audio").decode('utf-8'),
            chunk_index=1,
            is_final=False
        )
        
        audio_message = message_builder.audio_data_message(audio_payload)
        initial_transcriptions = mock_server.stats["total_transcriptions"]
        
        await mock_server._handle_audio_data(mock_websocket, audio_message)
        
        # No transcription should be triggered for non-final chunks
        assert mock_server.stats["total_transcriptions"] == initial_transcriptions
        # No response should be sent for non-final chunks
        mock_websocket.send.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_audio_data_unregistered_client(self, mock_server, mock_websocket, message_builder):
        """Test audio data from unregistered client triggers error"""
        audio_payload = AudioDataPayload(
            audio_data=base64.b64encode(b"test_audio").decode('utf-8'),
            chunk_index=1,
            is_final=True
        )
        
        audio_message = message_builder.audio_data_message(audio_payload)
        
        await mock_server._handle_audio_data(mock_websocket, audio_message)
        
        # Should send error response
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "error"
        assert response_data["payload"]["error_code"] == "CLIENT_NOT_REGISTERED"
    
    @pytest.mark.asyncio
    async def test_ping_pong_handling(self, mock_server, mock_websocket, message_builder):
        """Test ping-pong message handling"""
        ping_message = message_builder.ping_message()
        
        await mock_server._handle_ping(mock_websocket, ping_message)
        
        # Verify pong response
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "pong"
        assert response_data["header"]["correlation_id"] == ping_message.header.correlation_id
        assert "server_time" in response_data["payload"]
        assert "uptime" in response_data["payload"]
    
    @pytest.mark.asyncio
    async def test_status_request_handling(self, mock_server, mock_websocket, message_builder):
        """Test server status request handling"""
        status_message = WebSocketMessage(
            header=MessageHeader(
                message_type=MessageType.STATUS_REQUEST,
                sequence_id=10,
                timestamp=time.time(),
                client_id=message_builder.client_id,
                session_id=message_builder.session_id
            )
        )
        
        await mock_server._handle_status_request(mock_websocket, status_message)
        
        # Verify status response
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "status_response"
        
        payload = response_data["payload"]
        assert payload["status"] == "running"
        assert payload["server_type"] == "mock"
        assert payload["model_loaded"] is True
        assert payload["model_name"] == "mock-whisper-tiny.en"
        assert payload["version"] == "1.0.0-mock"
        assert "statistics" in payload
        assert "active_clients" in payload
        assert "supported_capabilities" in payload
        assert "transcription" in payload["supported_capabilities"]
    
    @pytest.mark.asyncio
    async def test_client_list_request_handling(self, mock_server, mock_websocket, message_builder):
        """Test client list request handling"""
        # Register a few mock clients
        test_clients = {
            "client1": {
                "websocket": mock_websocket,
                "info": ClientInfoPayload("Client 1", "1.0", "test", [], ClientStatus.CONNECTED),
                "status": ClientStatus.CONNECTED,
                "connected_at": time.time() - 100,
                "last_seen": time.time() - 10
            },
            "client2": {
                "websocket": Mock(),
                "info": ClientInfoPayload("Client 2", "2.0", "test", [], ClientStatus.RECORDING),
                "status": ClientStatus.RECORDING,
                "connected_at": time.time() - 50,
                "last_seen": time.time() - 5
            }
        }
        mock_server.clients.update(test_clients)
        
        list_message = WebSocketMessage(
            header=MessageHeader(
                message_type=MessageType.CLIENT_LIST_REQUEST,
                sequence_id=15,
                timestamp=time.time(),
                client_id=message_builder.client_id
            )
        )
        
        await mock_server._handle_client_list_request(mock_websocket, list_message)
        
        # Verify client list response
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "client_list_response"
        
        payload = response_data["payload"]
        assert payload["total_clients"] == 2
        assert len(payload["clients"]) == 2
        
        # Check client info structure
        client_info = payload["clients"][0]
        assert "client_id" in client_info
        assert "client_name" in client_info
        assert "status" in client_info
        assert "connected_at" in client_info
        assert "last_seen" in client_info
    
    @pytest.mark.asyncio
    async def test_performance_metrics_handling(self, mock_server, mock_websocket, message_builder):
        """Test performance metrics request handling"""
        perf_message = WebSocketMessage(
            header=MessageHeader(
                message_type=MessageType.PERFORMANCE_METRICS,
                sequence_id=20,
                timestamp=time.time(),
                client_id=message_builder.client_id
            )
        )
        
        await mock_server._handle_performance_request(mock_websocket, perf_message)
        
        # Verify performance metrics response
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "performance_metrics"
        
        payload = response_data["payload"]
        assert "latency_ms" in payload
        assert "throughput_mbps" in payload
        assert "cpu_usage" in payload
        assert "memory_usage" in payload
        assert "temperature" in payload
        assert "network_quality" in payload
        assert "processing_queue_size" in payload
        assert "error_count" in payload
        assert "uptime_seconds" in payload
    
    @pytest.mark.asyncio
    async def test_health_check_handling(self, mock_server, mock_websocket, message_builder):
        """Test health check request handling"""
        health_message = WebSocketMessage(
            header=MessageHeader(
                message_type=MessageType.HEALTH_CHECK,
                sequence_id=25,
                timestamp=time.time(),
                client_id=message_builder.client_id
            )
        )
        
        await mock_server._handle_health_check(mock_websocket, health_message)
        
        # Verify health response
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "health_response"
        
        payload = response_data["payload"]
        assert payload["status"] == "healthy"
        assert "uptime" in payload
        assert payload["model_loaded"] is True
        assert "active_connections" in payload
    
    @pytest.mark.asyncio
    async def test_audio_start_end_handling(self, mock_server, mock_websocket, message_builder):
        """Test audio start and end message handling"""
        # Register client first
        client_id = message_builder.client_id
        mock_server.clients[client_id] = {
            "websocket": mock_websocket,
            "info": ClientInfoPayload("Test", "1.0", "test", [], ClientStatus.CONNECTED),
            "status": ClientStatus.CONNECTED,
            "connected_at": time.time(),
            "last_seen": time.time(),
            "session_id": message_builder.session_id
        }
        
        # Test audio start
        audio_start_message = message_builder.audio_start_message(
            AudioConfigPayload(sample_rate=16000, channels=1)
        )
        
        await mock_server._handle_audio_start(mock_websocket, audio_start_message)
        assert mock_server.clients[client_id]["status"] == ClientStatus.RECORDING
        
        # Test audio end
        audio_end_message = message_builder.audio_end_message()
        
        await mock_server._handle_audio_end(mock_websocket, audio_end_message)
        assert mock_server.clients[client_id]["status"] == ClientStatus.CONNECTED
    
    @pytest.mark.asyncio
    async def test_client_disconnect_handling(self, mock_server, mock_websocket, message_builder):
        """Test client disconnect message handling"""
        # Register client
        client_id = message_builder.client_id
        mock_server.clients[client_id] = {
            "websocket": mock_websocket,
            "info": ClientInfoPayload("Test", "1.0", "test", [], ClientStatus.CONNECTED),
            "status": ClientStatus.CONNECTED,
            "connected_at": time.time(),
            "last_seen": time.time(),
            "session_id": message_builder.session_id
        }
        
        disconnect_message = message_builder.disconnect_message()
        
        await mock_server._handle_client_disconnect_msg(mock_websocket, disconnect_message)
        
        # Verify client was removed
        assert client_id not in mock_server.clients
        
        # Verify disconnect acknowledgment was sent
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "disconnect"
        assert response_data["payload"]["status"] == "disconnected"
    
    @pytest.mark.asyncio
    async def test_unknown_message_handling(self, mock_server, mock_websocket):
        """Test handling of unknown message types"""
        # Create a message with unknown type by manually crafting it
        unknown_message = WebSocketMessage(
            header=MessageHeader(
                message_type=MessageType.ACK,  # Use ACK as "unknown" to mock server
                sequence_id=1,
                timestamp=time.time(),
                client_id="test-client"
            )
        )
        
        await mock_server._handle_unknown(mock_websocket, unknown_message)
        
        # Should send error response
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        assert response_data["header"]["type"] == "error"
        assert response_data["payload"]["error_code"] == "UNKNOWN_MESSAGE_TYPE"
    
    @pytest.mark.asyncio
    async def test_error_sending(self, mock_server, mock_websocket):
        """Test error message sending with proper protocol"""
        correlation_id = "test-correlation-123"
        
        await mock_server._send_error(mock_websocket, "TEST_ERROR", "Test error message", correlation_id)
        
        mock_websocket.send.assert_called_once()
        response_data = json.loads(mock_websocket.send.call_args[0][0])
        
        assert response_data["header"]["type"] == "error"
        assert response_data["header"]["correlation_id"] == correlation_id
        assert response_data["payload"]["error_code"] == "TEST_ERROR"
        assert response_data["payload"]["error_message"] == "Test error message"
        assert response_data["payload"]["recoverable"] is True
        
        # Verify error count was incremented
        assert mock_server.stats["error_count"] == 1
    
    @pytest.mark.asyncio
    async def test_latency_simulation(self, mock_server, mock_websocket, message_builder):
        """Test that configured latency is applied to message processing"""
        # Set higher latency for this test
        mock_server.latency_ms = 200
        
        ping_message = message_builder.ping_message()
        
        start_time = time.time()
        await mock_server._handle_message(mock_websocket, ping_message.to_json())
        processing_time = time.time() - start_time
        
        # Should have taken at least the configured latency time
        assert processing_time >= 0.2
    
    def test_mock_processing_time_calculation(self, mock_server):
        """Test mock processing time calculation based on model size"""
        # Test different model sizes
        test_cases = [
            ("tiny.en", 0.1, 0.3),
            ("small.en", 0.4, 0.7),
            ("base.en", 0.9, 1.2),
            ("medium.en", 1.9, 2.2),
            ("large", 2.9, 3.2),
        ]
        
        for model_size, min_time, max_time in test_cases:
            mock_server.model_size = model_size
            processing_time = mock_server._get_mock_processing_time()
            assert min_time <= processing_time <= max_time
    
    def test_mock_transcription_generation(self, mock_server):
        """Test mock transcription text generation"""
        transcriptions = set()
        
        # Generate multiple transcriptions to test variety
        for _ in range(20):
            transcription = mock_server._generate_mock_transcription()
            assert transcription in mock_server.mock_transcriptions
            transcriptions.add(transcription)
        
        # Should get some variety in responses
        assert len(transcriptions) > 1
    
    def test_processing_time_average_calculation(self, mock_server):
        """Test processing time average calculation"""
        # Test first transcription
        mock_server._update_processing_time_avg(0.5)
        assert mock_server.stats["avg_processing_time"] == 0.5
        
        # Test second transcription
        mock_server.stats["total_transcriptions"] = 2
        mock_server._update_processing_time_avg(1.0)
        assert mock_server.stats["avg_processing_time"] == 0.75
        
        # Test third transcription
        mock_server.stats["total_transcriptions"] = 3
        mock_server._update_processing_time_avg(0.6)
        expected_avg = (0.75 * 2 + 0.6) / 3
        assert abs(mock_server.stats["avg_processing_time"] - expected_avg) < 0.001
    
    def test_mock_performance_metrics_generation(self, mock_server):
        """Test mock performance metrics generation"""
        metrics = mock_server._get_mock_performance_metrics()
        
        # Verify all required fields are present and within expected ranges
        assert isinstance(metrics.latency_ms, float)
        assert metrics.latency_ms > 0
        assert isinstance(metrics.throughput_mbps, float)
        assert 5.0 <= metrics.throughput_mbps <= 25.0
        assert isinstance(metrics.cpu_usage, float)
        assert 10.0 <= metrics.cpu_usage <= 60.0
        assert isinstance(metrics.memory_usage, float)
        assert 30.0 <= metrics.memory_usage <= 70.0
        assert isinstance(metrics.temperature, float)
        assert 35.0 <= metrics.temperature <= 55.0
        assert isinstance(metrics.network_quality, float)
        assert 0.8 <= metrics.network_quality <= 1.0
        assert isinstance(metrics.processing_queue_size, int)
        assert 0 <= metrics.processing_queue_size <= 5
        assert metrics.error_count == mock_server.stats["error_count"]
        assert metrics.uptime_seconds > 0
    
    @pytest.mark.asyncio
    async def test_connection_cleanup(self, mock_server, mock_websocket):
        """Test client cleanup when websocket disconnects"""
        # Register a client
        client_id = "test-cleanup-client"
        mock_server.clients[client_id] = {
            "websocket": mock_websocket,
            "info": ClientInfoPayload("Test", "1.0", "test", [], ClientStatus.CONNECTED),
            "status": ClientStatus.CONNECTED,
            "connected_at": time.time(),
            "last_seen": time.time(),
            "session_id": "test-session"
        }
        
        # Verify client is registered
        assert client_id in mock_server.clients
        
        # Cleanup disconnected client
        await mock_server._cleanup_disconnected_client(mock_websocket)
        
        # Verify client was removed
        assert client_id not in mock_server.clients
    
    @pytest.mark.asyncio
    async def test_expired_message_handling(self, mock_server, mock_websocket):
        """Test handling of expired messages"""
        # Create an expired message
        expired_message = WebSocketMessage(
            header=MessageHeader(
                message_type=MessageType.PING,
                sequence_id=1,
                timestamp=time.time() - 100,  # Old timestamp
                client_id="test-client",
                ttl=10  # Expired 90 seconds ago
            )
        )
        
        await mock_server._handle_message(mock_websocket, expired_message.to_json())
        
        # Should not process expired message, no response sent
        mock_websocket.send.assert_not_called()


class TestMockServerProtocolCompatibility:
    """Test mock server protocol compatibility with real server"""
    
    @pytest.fixture
    def mock_server(self):
        return MockWhisperServer(host="localhost", port=8768)
    
    @pytest.fixture
    def mock_websocket(self):
        websocket = Mock()
        websocket.remote_address = ("127.0.0.1", 12345)
        websocket.send = AsyncMock()
        return websocket
    
    @pytest.mark.asyncio
    async def test_message_routing_completeness(self, mock_server, mock_websocket):
        """Test that all expected message types are routed correctly"""
        # Test that all important message types have handlers
        expected_handlers = [
            MessageType.CONNECT,
            MessageType.DISCONNECT,
            MessageType.AUDIO_DATA,
            MessageType.AUDIO_START,
            MessageType.AUDIO_END,
            MessageType.PING,
            MessageType.STATUS_REQUEST,
            MessageType.CLIENT_LIST_REQUEST,
            MessageType.PERFORMANCE_METRICS,
            MessageType.HEALTH_CHECK,
        ]
        
        # Create a test message for each type
        for message_type in expected_handlers:
            test_message = WebSocketMessage(
                header=MessageHeader(
                    message_type=message_type,
                    sequence_id=1,
                    timestamp=time.time(),
                    client_id="test-protocol-client"
                )
            )
            
            # Should not raise exception for supported message types
            try:
                await mock_server._route_message(mock_websocket, test_message)
            except Exception as e:
                pytest.fail(f"Message type {message_type} routing failed: {e}")
    
    @pytest.mark.asyncio
    async def test_response_message_structure_compatibility(self, mock_server, mock_websocket):
        """Test that response messages follow the same structure as real server"""
        message_builder = MessageBuilder("test-compatibility-client", "test-session")
        
        # Test ping-pong response structure
        ping_message = message_builder.ping_message()
        await mock_server._handle_ping(mock_websocket, ping_message)
        
        mock_websocket.send.assert_called()
        response_json = mock_websocket.send.call_args[0][0]
        
        # Verify response can be parsed by protocol
        response_message = WebSocketMessage.from_json(response_json)
        assert response_message.header.message_type == MessageType.PONG
        assert response_message.header.correlation_id == ping_message.header.correlation_id
        
    def test_stats_structure_compatibility(self, mock_server):
        """Test that stats structure matches real server"""
        required_stats = [
            "total_connections",
            "active_connections", 
            "total_transcriptions",
            "avg_processing_time",
            "error_count",
            "uptime_seconds"
        ]
        
        for stat in required_stats:
            assert stat in mock_server.stats
            assert isinstance(mock_server.stats[stat], (int, float))


# Integration tests for task 2.2 validation
@pytest.mark.asyncio
async def test_mock_server_complete_workflow():
    """Integration test for complete mock server workflow matching task 2.2 requirements"""
    mock_server = MockWhisperServer(host="localhost", port=8769, latency_ms=10)
    
    # Mock websocket for testing
    mock_websocket = Mock()
    mock_websocket.remote_address = ("127.0.0.1", 12345)
    mock_websocket.send = AsyncMock()
    
    message_builder = MessageBuilder("integration-test-client", "integration-session")
    
    # 1. Test client registration
    client_info = ClientInfoPayload(
        client_name="Integration Test Client",
        client_version="1.0.0",
        platform="pytest-integration",
        capabilities=["audio_transcription", "real_time"],
        status=ClientStatus.CONNECTED
    )
    
    connect_message = message_builder.connect_message(client_info)
    await mock_server._handle_client_registration(mock_websocket, connect_message)
    
    # Verify client is registered
    assert message_builder.client_id in mock_server.clients
    
    # 2. Test audio processing workflow
    # Start audio stream
    audio_config = AudioConfigPayload(sample_rate=16000, channels=1)
    audio_start_message = message_builder.audio_start_message(audio_config)
    await mock_server._handle_audio_start(mock_websocket, audio_start_message)
    
    # Send audio data
    audio_data = base64.b64encode(b"integration_test_audio_data").decode('utf-8')
    audio_payload = AudioDataPayload(
        audio_data=audio_data,
        chunk_index=1,
        is_final=True
    )
    audio_message = message_builder.audio_data_message(audio_payload)
    await mock_server._handle_audio_data(mock_websocket, audio_message)
    
    # End audio stream
    audio_end_message = message_builder.audio_end_message()
    await mock_server._handle_audio_end(mock_websocket, audio_end_message)
    
    # 3. Test status monitoring
    status_message = WebSocketMessage(
        header=MessageHeader(
            message_type=MessageType.STATUS_REQUEST,
            sequence_id=10,
            timestamp=time.time(),
            client_id=message_builder.client_id
        )
    )
    await mock_server._handle_status_request(mock_websocket, status_message)
    
    # 4. Test disconnect
    disconnect_message = message_builder.disconnect_message()
    await mock_server._handle_client_disconnect_msg(mock_websocket, disconnect_message)
    
    # Verify client was cleaned up
    assert message_builder.client_id not in mock_server.clients
    
    # Verify stats were updated
    assert mock_server.stats["total_transcriptions"] >= 1
    
    # Verify multiple responses were sent (registration ack, transcription result, status response, disconnect ack)
    assert mock_websocket.send.call_count >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 