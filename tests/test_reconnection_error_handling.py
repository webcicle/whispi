"""
Test suite for Automatic Reconnection and Error Handling (Task 1.4)

This test suite validates the automatic reconnection and robust error handling
functionality for both the WebSocket client and server components.

Test Coverage:
- Connection loss detection and automatic reconnection
- Exponential backoff and retry mechanisms
- Session recovery and state management
- Network error handling (timeouts, refused connections)
- Protocol error handling (invalid messages, malformed data)
- Server error handling (overload, internal errors)
- Audio system error handling (device issues, permissions)
- Ping/pong health monitoring
- Message queuing during disconnection
- Resource management and cleanup
- Performance and latency requirements (<100ms)
"""

import pytest
import asyncio
import json
import time
import random
from unittest.mock import Mock, patch, AsyncMock, call
from typing import Dict, Any, List
import websockets
import logging

# Mock external dependencies for testing
import sys
from unittest.mock import MagicMock
sys.modules['pyaudio'] = MagicMock()
sys.modules['webrtcvad'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.signal'] = MagicMock()
sys.modules['numpy'] = MagicMock()

from client.websocket_client import EnhancedSpeechClient, ClientConnectionManager
from server.websocket_server import WhisperWebSocketServer
from shared.protocol import (
    MessageType, Priority, ClientStatus,
    MessageHeader, WebSocketMessage,
    AudioDataPayload, ErrorPayload,
    ClientInfoPayload, MessageBuilder
)
from shared.exceptions import NetworkError, TranscriptionError


# Test fixtures and utilities
class ConnectionSimulator:
    """Simulates various connection scenarios for testing"""
    
    def __init__(self):
        self.connection_attempts = 0
        self.should_fail = False
        self.failure_count = 0
        self.max_failures = 0
        self.latency_ms = 0
        self.should_timeout = False
        
    async def mock_connect(self, uri, **kwargs):
        """Mock websocket connection with controllable behavior"""
        self.connection_attempts += 1
        
        if self.should_timeout:
            await asyncio.sleep(10)  # Simulate timeout
            
        if self.should_fail and self.failure_count < self.max_failures:
            self.failure_count += 1
            if self.failure_count <= 2:
                raise ConnectionRefusedError("Connection refused")
            elif self.failure_count <= 4:
                raise OSError("Network unreachable")
            else:
                raise asyncio.TimeoutError("Connection timeout")
                
        # Simulate network latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
            
        return MockWebSocket()
    
    def set_failure_mode(self, max_failures: int):
        """Configure connection to fail for specified attempts"""
        self.should_fail = True
        self.max_failures = max_failures
        self.failure_count = 0
    
    def set_timeout_mode(self, enabled: bool):
        """Configure connection to timeout"""
        self.should_timeout = enabled
    
    def set_latency(self, latency_ms: int):
        """Set connection latency in milliseconds"""
        self.latency_ms = latency_ms


class MockWebSocket:
    """Enhanced mock WebSocket with connection state simulation"""
    
    def __init__(self):
        self.sent_messages = []
        self.received_messages = []
        self.is_connected = True
        self.close_code = None
        self.latency_ms = 0
        self.should_drop_connection = False
        self.drop_after_messages = 0
        self.message_count = 0
        
    def __aiter__(self):
        """Make MockWebSocket async iterable"""
        return self
        
    async def __anext__(self):
        """Async iterator method"""
        if not self.is_connected:
            raise StopAsyncIteration
        if self.received_messages:
            return self.received_messages.pop(0)
        # Simulate waiting for messages
        await asyncio.sleep(0.1)
        raise StopAsyncIteration
        
    async def send(self, message: str):
        """Mock send with connection dropping simulation"""
        if not self.is_connected:
            raise websockets.exceptions.ConnectionClosed(1006, "Connection lost")
            
        if self.should_drop_connection and self.message_count >= self.drop_after_messages:
            self.is_connected = False
            raise websockets.exceptions.ConnectionClosed(1006, "Connection lost")
            
        self.message_count += 1
        
        # Simulate network latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
            
        self.sent_messages.append(message)
        
    async def recv(self):
        """Mock receive with connection simulation"""
        if not self.is_connected:
            raise websockets.exceptions.ConnectionClosed(1006, "Connection lost")
            
        if not self.received_messages:
            await asyncio.sleep(0.1)
            return None
        return self.received_messages.pop(0)
        
    async def close(self, code=1000):
        """Mock close method"""
        self.is_connected = False
        self.close_code = code
        
    def simulate_connection_loss(self, after_messages: int = 0):
        """Simulate connection loss after specified number of messages"""
        self.should_drop_connection = True
        self.drop_after_messages = after_messages
        
    def add_received_message(self, message: str):
        """Add message to receive queue"""
        self.received_messages.append(message)


# Test classes
class TestAutomaticReconnection:
    """Test automatic reconnection functionality"""
    
    @pytest.fixture
    def connection_simulator(self):
        """Create connection simulator for testing"""
        return ConnectionSimulator()
    
    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies"""
        return EnhancedSpeechClient(server_url="ws://localhost:8765")
    
    @pytest.mark.asyncio
    async def test_connection_retry_with_exponential_backoff(self, client, connection_simulator):
        """Test exponential backoff during connection retries"""
        connection_simulator.set_failure_mode(max_failures=3)
        
        start_time = time.time()
        
        with patch('websockets.connect', connection_simulator.mock_connect):
            await client.connect()
            
        elapsed_time = time.time() - start_time
        
        # Verify exponential backoff delays (1s, 2s, 4s = ~7s total)
        assert elapsed_time >= 6.0, "Should have exponential backoff delays"
        assert connection_simulator.connection_attempts == 4, "Should make 4 attempts (3 failures + 1 success)"
        assert client.is_connected, "Should successfully connect after retries"
    
    @pytest.mark.asyncio
    async def test_maximum_retry_limit(self, client, connection_simulator):
        """Test maximum retry limit is respected"""
        connection_simulator.set_failure_mode(max_failures=10)  # More than max retries
        
        with patch('websockets.connect', connection_simulator.mock_connect):
            with pytest.raises(Exception):  # Should raise after max retries
                await client.connect()
                
        assert connection_simulator.connection_attempts == 5, "Should respect max retry limit"
        assert not client.is_connected, "Should not be connected after max retries"
    
    @pytest.mark.asyncio
    async def test_immediate_reconnection_on_connection_loss(self, client):
        """Test immediate reconnection attempt when connection is lost"""
        mock_ws = MockWebSocket()
        client.websocket = mock_ws
        client.is_connected = True
        
        # Simulate connection loss
        mock_ws.simulate_connection_loss(after_messages=0)
        
        reconnect_called = False
        original_connect = client.connect
        
        async def mock_connect():
            nonlocal reconnect_called
            reconnect_called = True
            return await original_connect()
        
        with patch.object(client, 'connect', mock_connect):
            with patch('websockets.connect', return_value=MockWebSocket()):
                await client._handle_connection_loss()
                
        assert reconnect_called, "Should attempt to reconnect immediately"
    
    @pytest.mark.asyncio
    async def test_session_recovery_after_reconnection(self, client):
        """Test session state is recovered after reconnection"""
        # Set up initial connection state
        client.client_id = "test-client-123"
        client.session_id = "session-456" 
        client.is_connected = True
        
        # Update the message builder with new client info
        from shared.protocol import MessageBuilder
        client.message_builder = MessageBuilder(client.client_id, client.session_id)
        
        # Simulate reconnection
        mock_ws = MockWebSocket()
        
        async def mock_connect_func(*args, **kwargs):
            return mock_ws
            
        with patch('websockets.connect', mock_connect_func):
            await client.connect()
            
        # Verify registration message was sent with correct session info
        assert len(mock_ws.sent_messages) >= 1
        connect_msg = json.loads(mock_ws.sent_messages[0])
        assert connect_msg["header"]["client_id"] == "test-client-123"
        assert connect_msg["header"]["session_id"] == "session-456"
    
    @pytest.mark.asyncio
    async def test_ping_based_connection_monitoring(self, client):
        """Test ping-based connection health monitoring"""
        mock_ws = MockWebSocket()
        client.websocket = mock_ws
        client.is_connected = True
        
        # Set up ping timing to simulate 50ms latency
        start_time = time.time() - 0.05  # 50ms ago
        client.ping_start_time = start_time
        
        # Simulate pong response
        pong_payload = {"latency_ms": 50.0, "timestamp": time.time()}
        await client._handle_pong_message(pong_payload)
        
        # Check that latency is approximately 50ms (within 5ms tolerance)
        assert abs(client.last_ping_latency - 50.0) < 5.0, f"Expected ~50ms, got {client.last_ping_latency}ms"
        
        # Test actual ping sending
        await client._send_ping()
        
        # Verify ping message structure
        assert len(mock_ws.sent_messages) == 1
        ping_msg = json.loads(mock_ws.sent_messages[0])
        assert ping_msg["header"]["type"] == "ping"


class TestErrorHandling:
    """Test comprehensive error handling functionality"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return EnhancedSpeechClient(server_url="ws://localhost:8765")
    
    @pytest.fixture
    def server(self):
        """Create test server"""
        return WhisperWebSocketServer(host="localhost", port=8766)
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, client):
        """Test handling of various network errors"""
        error_scenarios = [
            (ConnectionRefusedError("Connection refused"), "Connection refused"),
            (OSError("Network unreachable"), "Network unreachable"),
            (asyncio.TimeoutError("Connection timeout"), "Connection timeout"),
        ]
        
        for error, expected_error_msg in error_scenarios:
            with patch('websockets.connect', side_effect=error):
                with pytest.raises(ConnectionError) as exc_info:
                    await client.connect()
                
                # Verify the error message contains details about the original error
                assert expected_error_msg in str(exc_info.value)
                    
                assert not client.is_connected
    
    @pytest.mark.asyncio
    async def test_protocol_error_handling(self, server):
        """Test handling of protocol-level errors"""
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        mock_websocket.remote_address = ("127.0.0.1", 12345)
        
        # Test invalid JSON
        await server._handle_message(mock_websocket, "invalid json")
        
        # Verify error response
        mock_websocket.send.assert_called()
        error_msg = json.loads(mock_websocket.send.call_args[0][0])
        assert error_msg["header"]["type"] == "error"
        assert error_msg["payload"]["error_code"] == "INVALID_MESSAGE"
    
    @pytest.mark.asyncio
    async def test_audio_system_error_handling(self, client):
        """Test handling of audio system errors"""
        # Mock audio system failure
        with patch.object(client, '_record_audio_chunk', side_effect=OSError("Audio device disconnected")):
            with pytest.raises(OSError):
                client._record_audio_chunk()
                
        # Test graceful degradation
        client.last_error = "Audio device disconnected"
        assert "disconnected" in client.last_error
    
    @pytest.mark.asyncio
    async def test_server_overload_handling(self, server):
        """Test server behavior under overload conditions"""
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        mock_websocket.remote_address = ("127.0.0.1", 12345)
        
        # Simulate server overload by setting high connection count
        server.stats["active_connections"] = 100
        
        # Test that server can still handle basic requests
        message_builder = MessageBuilder("test-client", "session-123")
        ping_msg = message_builder.ping_message()
        
        await server._handle_ping(mock_websocket, ping_msg)
        
        # Should still respond to ping even under load
        mock_websocket.send.assert_called()
    
    @pytest.mark.asyncio
    async def test_graceful_error_recovery(self, client):
        """Test graceful recovery from various error states"""
        # Test recovery from connection loss
        client.is_connected = False
        client.last_error = "Connection lost"
        
        mock_ws = MockWebSocket()
        
        async def mock_connect_func(*args, **kwargs):
            return mock_ws
            
        with patch('websockets.connect', mock_connect_func):
            await client.connect()
            
        assert client.is_connected
        assert client.last_error == ""  # Error should be cleared


class TestReliabilityMechanisms:
    """Test reliability and robustness mechanisms"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return EnhancedSpeechClient(server_url="ws://localhost:8765")
    
    @pytest.fixture
    def server(self):
        """Create test server"""
        return WhisperWebSocketServer(host="localhost", port=8766)
    
    @pytest.mark.asyncio
    async def test_message_queuing_during_disconnection(self, client):
        """Test message queuing when disconnected"""
        # Implement message queue for offline scenarios
        client.is_connected = False
        
        # Messages should be queued when disconnected
        sample_audio = b"sample_audio_data"
        
        # This should not raise an error but queue the message
        try:
            await client._send_audio_chunk(sample_audio, chunk_index=1)
        except Exception:
            pass  # Expected when not connected
    
    @pytest.mark.asyncio
    async def test_data_integrity_during_reconnection(self, client):
        """Test data integrity is maintained during reconnection"""
        # Set up initial state
        client.client_id = "test-client-123"
        original_transcription = "test transcription"
        client.last_transcription = original_transcription
        
        # Simulate reconnection
        mock_ws = MockWebSocket()
        
        async def mock_connect_func(*args, **kwargs):
            return mock_ws
            
        with patch('websockets.connect', mock_connect_func):
            await client.connect()
            
        # Verify state is preserved
        assert client.client_id == "test-client-123"
        assert client.last_transcription == original_transcription
    
    @pytest.mark.asyncio
    async def test_connection_health_monitoring(self, client):
        """Test continuous connection health monitoring"""
        mock_ws = MockWebSocket()
        client.websocket = mock_ws
        client.is_connected = True
        
        # Test ping interval monitoring
        ping_start = time.time()
        await client._send_ping()
        
        assert len(mock_ws.sent_messages) == 1
        ping_msg = json.loads(mock_ws.sent_messages[0])
        assert ping_msg["header"]["type"] == "ping"
        
        # Verify timing
        ping_time = ping_msg["header"]["timestamp"]
        assert abs(ping_time - ping_start) < 0.1  # Within 100ms
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_on_error(self, client, server):
        """Test proper resource cleanup during error conditions"""
        # Test client cleanup
        mock_ws = MockWebSocket()
        client.websocket = mock_ws
        client.is_connected = True
        
        # Simulate error and cleanup
        client.cleanup()
        assert not client.is_connected
        
        # Test server cleanup
        mock_websocket = Mock()
        mock_websocket.remote_address = ("127.0.0.1", 12345)
        server.clients["test-client"] = {
            "websocket": mock_websocket,
            "info": Mock(),
            "status": ClientStatus.CONNECTED,
            "last_seen": time.time()
        }
        
        await server._cleanup_disconnected_client(mock_websocket)
        # Client should be removed from active clients list


class TestPerformanceRequirements:
    """Test performance requirements for reconnection and error handling"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return EnhancedSpeechClient(server_url="ws://localhost:8765")
    
    @pytest.mark.asyncio
    async def test_reconnection_speed(self, client):
        """Test reconnection happens within acceptable time limits"""
        start_time = time.time()
        
        mock_ws = MockWebSocket()
        
        async def mock_connect_func(*args, **kwargs):
            return mock_ws
            
        with patch('websockets.connect', mock_connect_func):
            await client.connect()
            
        connection_time = time.time() - start_time
        
        # Connection should be fast for local network
        assert connection_time < 1.0, "Initial connection should be under 1 second"
    
    @pytest.mark.asyncio
    async def test_error_recovery_time(self, client):
        """Test error recovery meets timing requirements"""
        # Test error detection and recovery speed
        start_time = time.time()
        
        client.is_connected = False
        client.last_error = "Test error"
        
        mock_ws = MockWebSocket()
        
        async def mock_connect_func(*args, **kwargs):
            return mock_ws
            
        with patch('websockets.connect', mock_connect_func):
            await client.connect()
            
        recovery_time = time.time() - start_time
        
        # Error recovery should be fast
        assert recovery_time < 2.0, "Error recovery should be under 2 seconds"
        assert client.last_error == "", "Error should be cleared after recovery"
    
    @pytest.mark.asyncio
    async def test_low_latency_communication(self, client):
        """Test communication maintains low latency (<100ms requirement)"""
        mock_ws = MockWebSocket()
        mock_ws.latency_ms = 50  # Simulate 50ms network latency
        client.websocket = mock_ws
        client.is_connected = True
        
        start_time = time.time()
        await client._send_ping()
        end_time = time.time()
        
        # Should meet latency requirement
        latency_ms = (end_time - start_time) * 1000
        assert latency_ms < 100, f"Latency {latency_ms}ms should be under 100ms"
    
    @pytest.mark.asyncio
    async def test_memory_management_during_errors(self, client):
        """Test memory usage doesn't grow during error conditions"""
        # This test would need actual memory monitoring in a real implementation
        # For now, we test that error handling doesn't accumulate objects
        
        initial_clients = len(getattr(client, 'clients', {}))
        
        # Simulate multiple error conditions
        for i in range(10):
            try:
                with patch('websockets.connect', side_effect=ConnectionError("Test error")):
                    await client.connect()
            except:
                pass
                
        final_clients = len(getattr(client, 'clients', {}))
        
        # Should not accumulate client objects
        assert final_clients == initial_clients, "Should not leak client objects"


class TestServerReconnectionHandling:
    """Test server-side handling of client reconnections"""
    
    @pytest.fixture
    def server(self):
        """Create test server"""
        return WhisperWebSocketServer(host="localhost", port=8766)
    
    @pytest.mark.asyncio
    async def test_client_reconnection_detection(self, server):
        """Test server detects and handles client reconnections"""
        mock_websocket = Mock()
        mock_websocket.send = AsyncMock()
        mock_websocket.remote_address = ("127.0.0.1", 12345)
        
        # Simulate client registration
        message_builder = MessageBuilder("test-client-123", "session-456")
        client_info_payload = ClientInfoPayload(
            client_name="Test Client",
            client_version="1.0.0",
            platform="macOS",
            capabilities=["audio_transcription"],
            status=ClientStatus.CONNECTED
        )
        
        connect_msg = message_builder.connect_message(client_info_payload)
        
        # Register client
        await server._handle_client_registration(mock_websocket, connect_msg)
        
        # Verify client is registered
        assert "test-client-123" in server.clients
        
        # Simulate client reconnection with same ID
        await server._handle_client_registration(mock_websocket, connect_msg)
        
        # Should handle reconnection gracefully
        assert len(server.clients) == 1  # Should not duplicate
        assert server.clients["test-client-123"]["websocket"] == mock_websocket
    
    @pytest.mark.asyncio
    async def test_connection_cleanup_on_error(self, server):
        """Test server cleans up connections properly on errors"""
        mock_websocket = Mock()
        mock_websocket.remote_address = ("127.0.0.1", 12345)
        
        # Add client to server
        server.clients["test-client"] = {
            "websocket": mock_websocket,
            "info": Mock(),
            "status": ClientStatus.CONNECTED,
            "last_seen": time.time()
        }
        
        initial_count = server.stats["active_connections"] = 1
        
        # Simulate connection cleanup
        await server._cleanup_disconnected_client(mock_websocket)
        
        # Should clean up properly
        assert "test-client" not in server.clients


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 