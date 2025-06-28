"""
WebSocket protocol definitions for Pi-Whispr communication
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json


class MessageType(Enum):
    """WebSocket message types"""
    # Connection
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    
    # Audio processing
    AUDIO_START = "audio_start"
    AUDIO_DATA = "audio_data"
    AUDIO_END = "audio_end"
    
    # Results
    TRANSCRIPTION_RESULT = "transcription_result"
    TRANSCRIPTION_ERROR = "transcription_error"
    
    # Status
    STATUS = "status"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    """Base WebSocket message structure"""
    type: MessageType
    client_id: Optional[str] = None
    timestamp: Optional[float] = None
    data: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps({
            "type": self.type.value,
            "client_id": self.client_id,
            "timestamp": self.timestamp,
            "data": self.data or {}
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> "WebSocketMessage":
        """Create message from JSON string"""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data["type"]),
            client_id=data.get("client_id"),
            timestamp=data.get("timestamp"),
            data=data.get("data", {})
        )


@dataclass
class TranscriptionResult:
    """Transcription result data structure"""
    text: str
    confidence: float
    processing_time: float
    model_used: str
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "language": self.language
        }


@dataclass
class AudioConfig:
    """Audio configuration for recording"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 320
    format: str = "int16"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size,
            "format": self.format
        } 