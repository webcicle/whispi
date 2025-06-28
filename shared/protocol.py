"""
WebSocket protocol definitions for Pi-Whispr communication

This module defines the complete message structure and types for WebSocket
communication between macOS client and Raspberry Pi server, ensuring
reliable ordering and structured communication.

Requirements implemented:
- Structured JSON messaging with required fields (type, timestamp, payload, sequence ID)
- All message types: connection, audio, transcription, status, error, ping/pong, 
  client management, performance tracking
- Reliable message ordering through sequence identifiers
- Comprehensive validation and error handling
"""

from enum import Enum
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import json
import time
import uuid
from abc import ABC, abstractmethod


class MessageType(Enum):
    """WebSocket message types for Pi-Whispr communication"""
    
    # Connection Management
    CONNECT = "connect"
    CONNECT_ACK = "connect_ack"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    
    # Client Management
    CLIENT_REGISTER = "client_register"
    CLIENT_UNREGISTER = "client_unregister"
    CLIENT_LIST_REQUEST = "client_list_request"
    CLIENT_LIST_RESPONSE = "client_list_response"
    CLIENT_STATUS_UPDATE = "client_status_update"
    
    # Audio Processing
    AUDIO_START = "audio_start"
    AUDIO_DATA = "audio_data"
    AUDIO_END = "audio_end"
    AUDIO_CONFIG = "audio_config"
    AUDIO_RESUME = "audio_resume"
    AUDIO_PAUSE = "audio_pause"
    
    # Transcription Results
    TRANSCRIPTION_RESULT = "transcription_result"
    TRANSCRIPTION_PARTIAL = "transcription_partial"
    TRANSCRIPTION_ERROR = "transcription_error"
    TRANSCRIPTION_PROGRESS = "transcription_progress"
    
    # Status and Monitoring
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"
    HEALTH_CHECK = "health_check"
    HEALTH_RESPONSE = "health_response"
    
    # Performance Tracking
    PERFORMANCE_METRICS = "performance_metrics"
    LATENCY_TEST = "latency_test"
    LATENCY_RESPONSE = "latency_response"
    BANDWIDTH_TEST = "bandwidth_test"
    
    # Error Handling
    ERROR = "error"
    WARNING = "warning"
    ACK = "ack"
    NACK = "nack"


class Priority(Enum):
    """Message priority levels for processing order"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ClientStatus(Enum):
    """Client connection status"""
    CONNECTED = "connected"
    RECORDING = "recording"
    PROCESSING = "processing"
    IDLE = "idle"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class MessageHeader:
    """Standard message header with required fields"""
    message_type: MessageType
    sequence_id: int
    timestamp: float
    client_id: str
    session_id: Optional[str] = None
    priority: Priority = Priority.NORMAL
    ttl: Optional[float] = None  # Time to live in seconds
    correlation_id: Optional[str] = None  # For request-response correlation
    
    def __post_init__(self):
        """Validate header fields"""
        if self.sequence_id < 0:
            raise ValueError("Sequence ID must be non-negative")
        if self.timestamp <= 0:
            raise ValueError("Timestamp must be positive")
        if not self.client_id:
            raise ValueError("Client ID is required")


@dataclass
class WebSocketMessage:
    """Enhanced base WebSocket message structure with reliable ordering"""
    header: MessageHeader
    payload: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        message_dict = {
            "header": {
                "type": self.header.message_type.value,
                "sequence_id": self.header.sequence_id,
                "timestamp": self.header.timestamp,
                "client_id": self.header.client_id,
                "session_id": self.header.session_id,
                "priority": self.header.priority.value,
                "ttl": self.header.ttl,
                "correlation_id": self.header.correlation_id
            },
            "payload": self.payload
        }
        return json.dumps(message_dict, separators=(',', ':'))
    
    @classmethod
    def from_json(cls, json_str: str) -> "WebSocketMessage":
        """Create message from JSON string with validation"""
        try:
            data = json.loads(json_str)
            
            # Validate required structure
            if "header" not in data:
                raise ValueError("Missing required 'header' field")
            
            header_data = data["header"]
            header = MessageHeader(
                message_type=MessageType(header_data["type"]),
                sequence_id=header_data["sequence_id"],
                timestamp=header_data["timestamp"],
                client_id=header_data["client_id"],
                session_id=header_data.get("session_id"),
                priority=Priority(header_data.get("priority", "normal")),
                ttl=header_data.get("ttl"),
                correlation_id=header_data.get("correlation_id")
            )
            
            return cls(
                header=header,
                payload=data.get("payload", {})
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise ValueError(f"Invalid message format: {e}")
    
    def is_expired(self) -> bool:
        """Check if message has exceeded its time to live"""
        if self.header.ttl is None:
            return False
        return time.time() > (self.header.timestamp + self.header.ttl)
    
    def create_ack(self, status: str = "success", error_message: Optional[str] = None) -> "WebSocketMessage":
        """Create an acknowledgment message for this message"""
        ack_header = MessageHeader(
            message_type=MessageType.ACK if status == "success" else MessageType.NACK,
            sequence_id=0,  # ACK messages don't need sequence ordering
            timestamp=time.time(),
            client_id=self.header.client_id,
            session_id=self.header.session_id,
            correlation_id=self.header.correlation_id or str(self.header.sequence_id)
        )
        
        ack_payload = {
            "original_sequence_id": self.header.sequence_id,
            "original_type": self.header.message_type.value,
            "status": status
        }
        
        if error_message:
            ack_payload["error_message"] = error_message
            
        return WebSocketMessage(header=ack_header, payload=ack_payload)


# Specialized Message Payloads

@dataclass
class AudioConfigPayload:
    """Audio configuration payload"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 320  # 20ms chunks
    format: str = "int16"
    bit_depth: int = 16
    encoding: str = "pcm"
    vad_enabled: bool = True
    vad_aggressiveness: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size,
            "format": self.format,
            "bit_depth": self.bit_depth,
            "encoding": self.encoding,
            "vad_enabled": self.vad_enabled,
            "vad_aggressiveness": self.vad_aggressiveness
        }


@dataclass
class AudioDataPayload:
    """Audio data payload with metadata"""
    audio_data: str  # Base64 encoded audio bytes
    chunk_index: int
    is_final: bool = False
    timestamp_offset: float = 0.0  # Offset from start of recording
    energy_level: Optional[float] = None  # Audio energy for VAD
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_data": self.audio_data,
            "chunk_index": self.chunk_index,
            "is_final": self.is_final,
            "timestamp_offset": self.timestamp_offset,
            "energy_level": self.energy_level
        }


@dataclass
class TranscriptionResultPayload:
    """Transcription result payload with comprehensive metadata"""
    text: str
    confidence: float
    processing_time: float
    model_used: str
    language: Optional[str] = None
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    audio_duration: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None  # For longer transcriptions
    is_partial: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "language": self.language,
            "word_timestamps": self.word_timestamps,
            "audio_duration": self.audio_duration,
            "segments": self.segments,
            "is_partial": self.is_partial
        }


@dataclass
class PerformanceMetricsPayload:
    """Performance tracking payload"""
    latency_ms: float
    throughput_mbps: Optional[float] = None
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    temperature: Optional[float] = None  # For Pi monitoring
    network_quality: Optional[float] = None  # Signal strength, packet loss, etc.
    processing_queue_size: Optional[int] = None
    error_count: Optional[int] = None
    uptime_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "throughput_mbps": self.throughput_mbps,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "temperature": self.temperature,
            "network_quality": self.network_quality,
            "processing_queue_size": self.processing_queue_size,
            "error_count": self.error_count,
            "uptime_seconds": self.uptime_seconds
        }


@dataclass
class ClientInfoPayload:
    """Client information for registration and management"""
    client_name: str
    client_version: str
    platform: str
    capabilities: List[str]
    status: ClientStatus
    last_seen: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_name": self.client_name,
            "client_version": self.client_version,
            "platform": self.platform,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "last_seen": self.last_seen
        }


@dataclass
class ErrorPayload:
    """Error information payload"""
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    recoverable: bool = True
    suggested_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "recoverable": self.recoverable,
            "suggested_action": self.suggested_action
        }


class MessageBuilder:
    """Helper class for building typed messages"""
    
    def __init__(self, client_id: str, session_id: Optional[str] = None):
        self.client_id = client_id
        self.session_id = session_id or str(uuid.uuid4())
        self._sequence_counter = 0
    
    def _next_sequence_id(self) -> int:
        """Get next sequence ID"""
        self._sequence_counter += 1
        return self._sequence_counter
    
    def _create_header(self, message_type: MessageType, priority: Priority = Priority.NORMAL,
                      ttl: Optional[float] = None, correlation_id: Optional[str] = None) -> MessageHeader:
        """Create a message header with auto-incremented sequence ID"""
        return MessageHeader(
            message_type=message_type,
            sequence_id=self._next_sequence_id(),
            timestamp=time.time(),
            client_id=self.client_id,
            session_id=self.session_id,
            priority=priority,
            ttl=ttl,
            correlation_id=correlation_id
        )
    
    def connect_message(self, client_info: ClientInfoPayload) -> WebSocketMessage:
        """Build a connection message"""
        header = self._create_header(MessageType.CONNECT, Priority.HIGH)
        return WebSocketMessage(header=header, payload=client_info.to_dict())
    
    def audio_data_message(self, audio_payload: AudioDataPayload) -> WebSocketMessage:
        """Build an audio data message"""
        header = self._create_header(MessageType.AUDIO_DATA, Priority.HIGH, ttl=5.0)
        return WebSocketMessage(header=header, payload=audio_payload.to_dict())
    
    def transcription_result_message(self, result_payload: TranscriptionResultPayload) -> WebSocketMessage:
        """Build a transcription result message"""
        header = self._create_header(MessageType.TRANSCRIPTION_RESULT, Priority.HIGH)
        return WebSocketMessage(header=header, payload=result_payload.to_dict())
    
    def performance_metrics_message(self, metrics_payload: PerformanceMetricsPayload) -> WebSocketMessage:
        """Build a performance metrics message"""
        header = self._create_header(MessageType.PERFORMANCE_METRICS, Priority.LOW)
        return WebSocketMessage(header=header, payload=metrics_payload.to_dict())
    
    def error_message(self, error_payload: ErrorPayload) -> WebSocketMessage:
        """Build an error message"""
        header = self._create_header(MessageType.ERROR, Priority.CRITICAL)
        return WebSocketMessage(header=header, payload=error_payload.to_dict())
    
    def ping_message(self) -> WebSocketMessage:
        """Build a ping message for connection testing"""
        header = self._create_header(MessageType.PING, Priority.NORMAL, ttl=10.0)
        return WebSocketMessage(header=header, payload={"timestamp": time.time()})
    
    def audio_start_message(self, audio_config: AudioConfigPayload) -> WebSocketMessage:
        """Build an audio start message"""
        header = self._create_header(MessageType.AUDIO_START, Priority.HIGH)
        return WebSocketMessage(header=header, payload=audio_config.to_dict())
    
    def audio_end_message(self) -> WebSocketMessage:
        """Build an audio end message"""
        header = self._create_header(MessageType.AUDIO_END, Priority.HIGH)
        return WebSocketMessage(header=header, payload={"timestamp": time.time()})
    
    def disconnect_message(self) -> WebSocketMessage:
        """Build a disconnect message"""
        header = self._create_header(MessageType.DISCONNECT, Priority.HIGH)
        return WebSocketMessage(header=header, payload={"reason": "client_initiated"})


class MessageValidator:
    """Validates message structure and content"""
    
    @staticmethod
    def validate_message(message: WebSocketMessage) -> bool:
        """Validate a WebSocket message"""
        try:
            # Check if message is expired
            if message.is_expired():
                return False
            
            # Validate header
            if not message.header.client_id:
                return False
            
            if message.header.sequence_id < 0:
                return False
            
            # Type-specific validation
            return MessageValidator._validate_payload(message.header.message_type, message.payload)
            
        except Exception:
            return False
    
    @staticmethod
    def validate_message_json(json_str: str) -> bool:
        """Validate a JSON message string"""
        try:
            message = WebSocketMessage.from_json(json_str)
            return MessageValidator.validate_message(message)
        except Exception:
            return False
    
    @staticmethod
    def _validate_payload(message_type: MessageType, payload: Dict[str, Any]) -> bool:
        """Validate payload based on message type"""
        if message_type == MessageType.AUDIO_DATA:
            required_fields = ["audio_data", "chunk_index"]
            return all(field in payload for field in required_fields)
        
        elif message_type == MessageType.TRANSCRIPTION_RESULT:
            required_fields = ["text", "confidence", "processing_time", "model_used"]
            return all(field in payload for field in required_fields)
        
        elif message_type == MessageType.ERROR:
            required_fields = ["error_code", "error_message"]
            return all(field in payload for field in required_fields)
        
        # Add more specific validations as needed
        return True


# Legacy compatibility - keeping original classes for backward compatibility
@dataclass
class TranscriptionResult:
    """Legacy transcription result for backward compatibility"""
    text: str
    confidence: float
    processing_time: float
    model_used: str
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "language": self.language
        }


@dataclass
class AudioConfig:
    """Legacy audio configuration for backward compatibility"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 320
    format: str = "int16"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size,
            "format": self.format
        }