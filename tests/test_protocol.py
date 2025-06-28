"""
Test suite for Pi-Whispr WebSocket protocol

This test suite validates the message structure and types for WebSocket
communication, ensuring reliable ordering and structured communication.

Test Coverage:
- Message structure validation
- Sequence ID ordering
- Payload validation for all message types
- JSON serialization/deserialization
- Error handling and edge cases
- Message builder functionality
- Automated schema checks
"""

import pytest
import json
import time
import base64
from typing import Dict, Any

from shared.protocol import (
    MessageType, Priority, ClientStatus,
    MessageHeader, WebSocketMessage,
    AudioConfigPayload, AudioDataPayload, TranscriptionResultPayload,
    PerformanceMetricsPayload, ClientInfoPayload, ErrorPayload,
    MessageBuilder, MessageValidator
)


class TestMessageHeader:
    """Test cases for MessageHeader validation"""
    
    def test_valid_header_creation(self):
        """Test creating a valid message header"""
        header = MessageHeader(
            message_type=MessageType.CONNECT,
            sequence_id=1,
            timestamp=time.time(),
            client_id="test-client-123"
        )
        assert header.message_type == MessageType.CONNECT
        assert header.sequence_id == 1
        assert header.client_id == "test-client-123"
        assert header.priority == Priority.NORMAL  # Default value
    
    def test_header_validation_errors(self):
        """Test header validation catches invalid values"""
        with pytest.raises(ValueError, match="Sequence ID must be non-negative"):
            MessageHeader(
                message_type=MessageType.CONNECT,
                sequence_id=-1,
                timestamp=time.time(),
                client_id="test-client"
            )
        
        with pytest.raises(ValueError, match="Timestamp must be positive"):
            MessageHeader(
                message_type=MessageType.CONNECT,
                sequence_id=1,
                timestamp=0,
                client_id="test-client"
            )
        
        with pytest.raises(ValueError, match="Client ID is required"):
            MessageHeader(
                message_type=MessageType.CONNECT,
                sequence_id=1,
                timestamp=time.time(),
                client_id=""
            )


class TestWebSocketMessage:
    """Test cases for WebSocketMessage functionality"""
    
    def test_message_creation_and_serialization(self):
        """Test creating and serializing a WebSocket message"""
        header = MessageHeader(
            message_type=MessageType.CONNECT,
            sequence_id=1,
            timestamp=time.time(),
            client_id="test-client",
            session_id="session-123"
        )
        
        payload = {"test": "data", "number": 42}
        message = WebSocketMessage(header=header, payload=payload)
        
        # Test JSON serialization
        json_str = message.to_json()
        assert isinstance(json_str, str)
        
        # Validate JSON structure
        data = json.loads(json_str)
        assert "header" in data
        assert "payload" in data
        assert data["header"]["type"] == "connect"
        assert data["header"]["sequence_id"] == 1
        assert data["header"]["client_id"] == "test-client"
        assert data["payload"]["test"] == "data"
        assert data["payload"]["number"] == 42
    
    def test_message_deserialization(self):
        """Test deserializing a WebSocket message from JSON"""
        json_data = {
            "header": {
                "type": "audio_data",
                "sequence_id": 5,
                "timestamp": time.time(),
                "client_id": "test-client",
                "session_id": "session-123",
                "priority": "high",
                "ttl": 10.0,
                "correlation_id": "corr-123"
            },
            "payload": {
                "audio_data": "base64encodeddata",
                "chunk_index": 1
            }
        }
        
        json_str = json.dumps(json_data)
        message = WebSocketMessage.from_json(json_str)
        
        assert message.header.message_type == MessageType.AUDIO_DATA
        assert message.header.sequence_id == 5
        assert message.header.client_id == "test-client"
        assert message.header.priority == Priority.HIGH
        assert message.payload["audio_data"] == "base64encodeddata"
        assert message.payload["chunk_index"] == 1
    
    def test_message_deserialization_errors(self):
        """Test error handling in message deserialization"""
        # Missing header
        with pytest.raises(ValueError, match="Missing required 'header' field"):
            WebSocketMessage.from_json('{"payload": {}}')
        
        # Invalid JSON
        with pytest.raises(ValueError, match="Invalid message format"):
            WebSocketMessage.from_json('invalid json')
        
        # Invalid message type
        invalid_json = {
            "header": {
                "type": "invalid_type",
                "sequence_id": 1,
                "timestamp": time.time(),
                "client_id": "test"
            },
            "payload": {}
        }
        with pytest.raises(ValueError):
            WebSocketMessage.from_json(json.dumps(invalid_json))
    
    def test_message_expiration(self):
        """Test message TTL expiration functionality"""
        # Non-expiring message
        header = MessageHeader(
            message_type=MessageType.PING,
            sequence_id=1,
            timestamp=time.time(),
            client_id="test-client"
        )
        message = WebSocketMessage(header=header)
        assert not message.is_expired()
        
        # Expired message
        header_expired = MessageHeader(
            message_type=MessageType.PING,
            sequence_id=2,
            timestamp=time.time() - 20,  # 20 seconds ago
            client_id="test-client",
            ttl=10.0  # 10 second TTL
        )
        message_expired = WebSocketMessage(header=header_expired)
        assert message_expired.is_expired()
    
    def test_message_acknowledgment(self):
        """Test creating acknowledgment messages"""
        original_header = MessageHeader(
            message_type=MessageType.AUDIO_DATA,
            sequence_id=10,
            timestamp=time.time(),
            client_id="test-client",
            correlation_id="test-corr"
        )
        original_message = WebSocketMessage(header=original_header)
        
        # Success ACK
        ack_message = original_message.create_ack("success")
        assert ack_message.header.message_type == MessageType.ACK
        assert ack_message.header.client_id == "test-client"
        assert ack_message.payload["original_sequence_id"] == 10
        assert ack_message.payload["status"] == "success"
        
        # Error NACK
        nack_message = original_message.create_ack("error", "Processing failed")
        assert nack_message.header.message_type == MessageType.NACK
        assert nack_message.payload["status"] == "error"
        assert nack_message.payload["error_message"] == "Processing failed"


class TestPayloadClasses:
    """Test cases for specialized payload classes"""
    
    def test_audio_config_payload(self):
        """Test AudioConfigPayload functionality"""
        config = AudioConfigPayload(
            sample_rate=16000,
            channels=1,
            chunk_size=320,
            vad_enabled=True
        )
        
        data = config.to_dict()
        assert data["sample_rate"] == 16000
        assert data["channels"] == 1
        assert data["chunk_size"] == 320
        assert data["vad_enabled"] is True
        assert data["vad_aggressiveness"] == 2  # Default value
    
    def test_audio_data_payload(self):
        """Test AudioDataPayload functionality"""
        audio_bytes = b"fake audio data"
        encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
        
        payload = AudioDataPayload(
            audio_data=encoded_audio,
            chunk_index=5,
            is_final=True,
            timestamp_offset=1.5,
            energy_level=0.8
        )
        
        data = payload.to_dict()
        assert data["audio_data"] == encoded_audio
        assert data["chunk_index"] == 5
        assert data["is_final"] is True
        assert data["timestamp_offset"] == 1.5
        assert data["energy_level"] == 0.8
    
    def test_transcription_result_payload(self):
        """Test TranscriptionResultPayload functionality"""
        result = TranscriptionResultPayload(
            text="Hello world",
            confidence=0.95,
            processing_time=1.2,
            model_used="tiny.en",
            language="en",
            word_timestamps=[{"word": "Hello", "start": 0.0, "end": 0.5}],
            audio_duration=2.0,
            is_partial=False
        )
        
        data = result.to_dict()
        assert data["text"] == "Hello world"
        assert data["confidence"] == 0.95
        assert data["processing_time"] == 1.2
        assert data["model_used"] == "tiny.en"
        assert data["language"] == "en"
        assert len(data["word_timestamps"]) == 1
        assert data["audio_duration"] == 2.0
        assert data["is_partial"] is False
    
    def test_performance_metrics_payload(self):
        """Test PerformanceMetricsPayload functionality"""
        metrics = PerformanceMetricsPayload(
            latency_ms=150.5,
            throughput_mbps=10.2,
            cpu_usage=45.0,
            memory_usage=60.5,
            temperature=42.0,
            network_quality=0.95,
            processing_queue_size=3,
            error_count=0,
            uptime_seconds=3600.0
        )
        
        data = metrics.to_dict()
        assert data["latency_ms"] == 150.5
        assert data["cpu_usage"] == 45.0
        assert data["temperature"] == 42.0
        assert data["uptime_seconds"] == 3600.0
    
    def test_client_info_payload(self):
        """Test ClientInfoPayload functionality"""
        client_info = ClientInfoPayload(
            client_name="Pi-Whispr macOS Client",
            client_version="1.0.0",
            platform="macOS",
            capabilities=["audio_recording", "text_insertion", "vad"],
            status=ClientStatus.CONNECTED,
            last_seen=time.time()
        )
        
        data = client_info.to_dict()
        assert data["client_name"] == "Pi-Whispr macOS Client"
        assert data["client_version"] == "1.0.0"
        assert data["platform"] == "macOS"
        assert "audio_recording" in data["capabilities"]
        assert data["status"] == "connected"
        assert "last_seen" in data
    
    def test_error_payload(self):
        """Test ErrorPayload functionality"""
        error = ErrorPayload(
            error_code="AUDIO_001",
            error_message="Failed to initialize audio device",
            error_details={"device_id": "default", "sample_rate": 16000},
            recoverable=True,
            suggested_action="Check microphone permissions"
        )
        
        data = error.to_dict()
        assert data["error_code"] == "AUDIO_001"
        assert data["error_message"] == "Failed to initialize audio device"
        assert data["error_details"]["device_id"] == "default"
        assert data["recoverable"] is True
        assert data["suggested_action"] == "Check microphone permissions"


class TestMessageBuilder:
    """Test cases for MessageBuilder functionality"""
    
    def test_message_builder_initialization(self):
        """Test MessageBuilder initialization"""
        builder = MessageBuilder("test-client-456")
        assert builder.client_id == "test-client-456"
        assert builder.session_id is not None
        assert builder._sequence_counter == 0
    
    def test_sequence_id_increment(self):
        """Test sequence ID auto-increment"""
        builder = MessageBuilder("test-client")
        
        # Create multiple messages and verify sequence IDs increment
        msg1 = builder.ping_message()
        msg2 = builder.ping_message()
        msg3 = builder.ping_message()
        
        assert msg1.header.sequence_id == 1
        assert msg2.header.sequence_id == 2
        assert msg3.header.sequence_id == 3
    
    def test_connect_message_building(self):
        """Test building connection messages"""
        builder = MessageBuilder("test-client")
        client_info = ClientInfoPayload(
            client_name="Test Client",
            client_version="1.0.0",
            platform="macOS",
            capabilities=["test"],
            status=ClientStatus.CONNECTED
        )
        
        message = builder.connect_message(client_info)
        assert message.header.message_type == MessageType.CONNECT
        assert message.header.priority == Priority.HIGH
        assert message.payload["client_name"] == "Test Client"
    
    def test_audio_data_message_building(self):
        """Test building audio data messages"""
        builder = MessageBuilder("test-client")
        audio_payload = AudioDataPayload(
            audio_data="encodedaudiodata",
            chunk_index=1,
            is_final=False
        )
        
        message = builder.audio_data_message(audio_payload)
        assert message.header.message_type == MessageType.AUDIO_DATA
        assert message.header.priority == Priority.HIGH
        assert message.header.ttl == 5.0  # Audio messages have TTL
        assert message.payload["chunk_index"] == 1
    
    def test_error_message_building(self):
        """Test building error messages"""
        builder = MessageBuilder("test-client")
        error_payload = ErrorPayload(
            error_code="TEST_001",
            error_message="Test error"
        )
        
        message = builder.error_message(error_payload)
        assert message.header.message_type == MessageType.ERROR
        assert message.header.priority == Priority.CRITICAL
        assert message.payload["error_code"] == "TEST_001"


class TestMessageValidator:
    """Test cases for MessageValidator functionality"""
    
    def test_valid_message_validation(self):
        """Test validation of valid messages"""
        builder = MessageBuilder("test-client")
        
        # Valid ping message
        ping_msg = builder.ping_message()
        assert MessageValidator.validate_message(ping_msg) is True
        
        # Valid audio data message
        audio_payload = AudioDataPayload(audio_data="data", chunk_index=1)
        audio_msg = builder.audio_data_message(audio_payload)
        assert MessageValidator.validate_message(audio_msg) is True
    
    def test_expired_message_validation(self):
        """Test validation rejects expired messages"""
        header = MessageHeader(
            message_type=MessageType.PING,
            sequence_id=1,
            timestamp=time.time() - 20,  # 20 seconds ago
            client_id="test-client",
            ttl=10.0  # 10 second TTL - expired
        )
        message = WebSocketMessage(header=header)
        
        assert MessageValidator.validate_message(message) is False
    
    def test_invalid_payload_validation(self):
        """Test validation catches invalid payloads"""
        # Audio data message missing required fields
        header = MessageHeader(
            message_type=MessageType.AUDIO_DATA,
            sequence_id=1,
            timestamp=time.time(),
            client_id="test-client"
        )
        
        # Missing required 'audio_data' field
        invalid_payload = {"chunk_index": 1}
        message = WebSocketMessage(header=header, payload=invalid_payload)
        
        assert MessageValidator.validate_message(message) is False


class TestCompleteWorkflow:
    """Integration tests for complete message workflows"""
    
    def test_audio_transcription_workflow(self):
        """Test a complete audio transcription message workflow"""
        builder = MessageBuilder("macos-client-001", "session-abc123")
        
        # 1. Client connects
        client_info = ClientInfoPayload(
            client_name="Pi-Whispr macOS Client",
            client_version="1.0.0",
            platform="macOS",
            capabilities=["audio_recording", "text_insertion"],
            status=ClientStatus.CONNECTED
        )
        connect_msg = builder.connect_message(client_info)
        
        # 2. Client sends audio data
        audio_payload = AudioDataPayload(
            audio_data=base64.b64encode(b"fake audio").decode(),
            chunk_index=1,
            is_final=True,
            energy_level=0.7
        )
        audio_msg = builder.audio_data_message(audio_payload)
        
        # 3. Server responds with transcription
        transcription_payload = TranscriptionResultPayload(
            text="Hello world",
            confidence=0.95,
            processing_time=1.2,
            model_used="tiny.en"
        )
        transcription_msg = builder.transcription_result_message(transcription_payload)
        
        # Validate all messages
        assert MessageValidator.validate_message(connect_msg)
        assert MessageValidator.validate_message(audio_msg)
        assert MessageValidator.validate_message(transcription_msg)
        
        # Verify sequence ordering
        assert connect_msg.header.sequence_id == 1
        assert audio_msg.header.sequence_id == 2
        assert transcription_msg.header.sequence_id == 3
        
        # Test JSON round-trip for all messages
        for msg in [connect_msg, audio_msg, transcription_msg]:
            json_str = msg.to_json()
            restored_msg = WebSocketMessage.from_json(json_str)
            assert restored_msg.header.message_type == msg.header.message_type
            assert restored_msg.header.sequence_id == msg.header.sequence_id
            assert restored_msg.header.client_id == msg.header.client_id
    
    def test_error_handling_workflow(self):
        """Test error handling and acknowledgment workflow"""
        builder = MessageBuilder("test-client")
        
        # Create an error message
        error_payload = ErrorPayload(
            error_code="TRANSCRIPTION_001",
            error_message="Model loading failed",
            recoverable=True,
            suggested_action="Restart the service"
        )
        error_msg = builder.error_message(error_payload)
        
        # Create acknowledgment
        ack_msg = error_msg.create_ack("success")
        
        # Validate both messages
        assert MessageValidator.validate_message(error_msg)
        assert MessageValidator.validate_message(ack_msg)
        
        # Verify acknowledgment structure
        assert ack_msg.header.message_type == MessageType.ACK
        assert ack_msg.payload["original_sequence_id"] == error_msg.header.sequence_id
        assert ack_msg.payload["original_type"] == "error"


if __name__ == "__main__":
    # Run specific test for development
    import sys
    
    print("Running Pi-Whispr Protocol Tests...")
    
    # Simple test runner for development
    test_classes = [
        TestMessageHeader,
        TestWebSocketMessage,
        TestPayloadClasses,
        TestMessageBuilder,
        TestMessageValidator,
        TestCompleteWorkflow
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                print(f"✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"✗ {method_name}: {e}")
    
    print(f"\n--- Results ---")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed! ✗")
        sys.exit(1)