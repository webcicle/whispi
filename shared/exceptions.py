"""
Custom exceptions for Pi-Whispr system
"""


class PiWhisprError(Exception):
    """Base exception for Pi-Whispr system"""
    pass


class AudioError(PiWhisprError):
    """Audio recording/processing related errors"""
    pass


class TranscriptionError(PiWhisprError):
    """Speech transcription related errors"""
    pass


class NetworkError(PiWhisprError):
    """WebSocket/network communication errors"""
    pass


class ConfigurationError(PiWhisprError):
    """Configuration and setup errors"""
    pass


class PermissionError(PiWhisprError):
    """macOS permission related errors"""
    pass


class ModelError(PiWhisprError):
    """Whisper model loading/processing errors"""
    pass 