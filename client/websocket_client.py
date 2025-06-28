#!/usr/bin/env python3
"""
Enhanced MacBook Speech Client with Improved Audio Processing

This implements the macOS WebSocket client with enhanced features based on Option 3.
"""

import asyncio
import websockets
import json
import base64
import time
import threading
import uuid
import logging
from typing import Dict, Any, Optional, Callable

# Mock imports for testing - real implementation would import actual modules
try:
    import pyaudio
    import numpy as np
    import webrtcvad
    from scipy.signal import butter, lfilter
except ImportError:
    pyaudio = None
    np = None
    webrtcvad = None

from shared.protocol import (
    MessageType, Priority, ClientStatus,
    MessageHeader, WebSocketMessage,
    AudioConfigPayload, AudioDataPayload, TranscriptionResultPayload,
    ClientInfoPayload, MessageBuilder
)
from shared.constants import SAMPLE_RATE, CHANNELS, CHUNK_SIZE


logger = logging.getLogger(__name__)


class ClientConnectionManager:
    """Manages WebSocket connection with retry logic and connection health"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0
        self.is_connected = False
        self.websocket = None
    
    async def connect_with_retry(self) -> Optional[websockets.WebSocketServerProtocol]:
        """Connect to server with retry logic and exponential backoff"""
        for attempt in range(self.max_reconnect_attempts):
            try:
                self.websocket = await websockets.connect(self.server_url)
                self.is_connected = True
                return self.websocket
            except Exception as e:
                if attempt < self.max_reconnect_attempts - 1:
                    delay = self.reconnect_delay * (2 ** attempt)
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Failed to connect after {self.max_reconnect_attempts} attempts")
                    raise
        return None
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False


class EnhancedSpeechClient:
    """Enhanced WebSocket client for macOS with advanced audio processing"""
    
    def __init__(self, server_url: str = "ws://192.168.1.100:8765"):
        # Connection settings
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
        self.is_recording = False
        
        # Connection manager
        self.connection_manager = ClientConnectionManager(server_url)
        
        # Client identification
        self.client_id = f"macos-client-{uuid.uuid4().hex[:8]}"
        self.session_id = f"session-{uuid.uuid4().hex[:8]}"
        
        # Audio configuration
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.chunk_size = CHUNK_SIZE
        
        # Audio processing
        self.audio = None
        self.vad = None
        if webrtcvad:
            self.vad = webrtcvad.Vad(2)
        
        # Message handling
        self.message_builder = MessageBuilder(self.client_id, self.session_id)
        
        # State tracking
        self.last_transcription = ""
        self.last_error = ""
        self.last_ping_latency = 0.0
        self.audio_processing_time = 0.0
        self.audio_start_time = 0.0
        self.ping_start_time = 0.0
        
        # Initialize audio if available
        if pyaudio:
            self.audio = pyaudio.PyAudio()
    
    async def connect(self) -> None:
        """Establish WebSocket connection to server"""
        self.websocket = await self.connection_manager.connect_with_retry()
        self.is_connected = True
        
        # Send client registration
        await self._send_connect_message()
        
        # Start message listener
        asyncio.create_task(self._message_listener())
    
    async def _handle_connection_loss(self) -> None:
        """Handle connection loss with automatic reconnection"""
        self.is_connected = False
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
    
    def _setup_hotkeys(self) -> None:
        """Setup global hotkey listeners"""
        # Mock implementation for testing
        logger.info("Setting up hotkeys...")
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "latency_ms": self.last_ping_latency,
            "audio_processing_time": self.audio_processing_time,
            "timestamp": time.time()
        }
    
    def _record_audio_chunk(self) -> bytes:
        """Record a single audio chunk"""
        # Mock implementation for testing
        return b'\x00\x01\x02\x03' * 80  # 320 bytes = 20ms at 16kHz
    
    async def _process_audio_chunk(self, audio_data: bytes, chunk_index: int, is_final: bool = False) -> None:
        """Process and send audio chunk"""
        processed_audio = self._preprocess_audio(audio_data)
        await self._send_audio_chunk(processed_audio, chunk_index, is_final)
    
    async def _send_connect_message(self) -> None:
        """Send client registration message"""
        client_info = ClientInfoPayload(
            client_name="Enhanced macOS Speech Client",
            client_version="1.0.0",
            platform="macOS",
            capabilities=["audio_recording", "text_insertion"],
            status=ClientStatus.CONNECTED
        )
        
        connect_msg = self.message_builder.connect_message(client_info)
        await self.websocket.send(connect_msg.to_json())
    
    async def _message_listener(self) -> None:
        """Listen for incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                await self._handle_incoming_message(message)
        except websockets.exceptions.ConnectionClosed:
            self.is_connected = False
    
    async def _handle_incoming_message(self, raw_message: str) -> None:
        """Handle incoming WebSocket message"""
        try:
            message = WebSocketMessage.from_json(raw_message)
            
            if message.header.message_type == MessageType.TRANSCRIPTION_RESULT:
                await self._handle_transcription_result(message.payload)
            elif message.header.message_type == MessageType.PONG:
                await self._handle_pong_message(message.payload)
            elif message.header.message_type == MessageType.ERROR:
                await self._handle_error_message(message.payload)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_transcription_result(self, payload: Dict[str, Any]) -> None:
        """Handle transcription result"""
        text = payload.get("text", "")
        self.last_transcription = text
        await self._insert_text(text)
    
    async def _handle_pong_message(self, payload: Dict[str, Any]) -> None:
        """Handle pong response"""
        if self.ping_start_time > 0:
            self.last_ping_latency = (time.time() - self.ping_start_time) * 1000
            self.ping_start_time = 0
    
    async def _handle_error_message(self, payload: Dict[str, Any]) -> None:
        """Handle error message"""
        error_code = payload.get("error_code", "UNKNOWN")
        error_message = payload.get("error_message", "Unknown error")
        self.last_error = f"{error_code}: {error_message}"
    
    async def _send_ping(self) -> None:
        """Send ping message"""
        if not self.is_connected:
            return
        
        self.ping_start_time = time.time()
        ping_msg = self.message_builder.ping_message()
        await self.websocket.send(ping_msg.to_json())
    
    async def start_recording(self) -> None:
        """Start audio recording"""
        if self.is_recording or not self.is_connected:
            return
        
        self.is_recording = True
        self.audio_start_time = time.time()
        
        # Send audio start message
        audio_config = AudioConfigPayload(
            sample_rate=self.sample_rate,
            channels=self.channels,
            chunk_size=self.chunk_size
        )
        
        start_msg = self.message_builder.audio_start_message(audio_config)
        await self.websocket.send(start_msg.to_json())
    
    async def stop_recording(self) -> None:
        """Stop audio recording"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Send audio end message
        header = MessageHeader(
            message_type=MessageType.AUDIO_END,
            sequence_id=self.message_builder._next_sequence_id(),
            timestamp=time.time(),
            client_id=self.client_id,
            session_id=self.session_id
        )
        
        end_msg = WebSocketMessage(header=header, payload={})
        await self.websocket.send(end_msg.to_json())
    
    async def _send_audio_chunk(self, audio_data: bytes, chunk_index: int, is_final: bool = False) -> None:
        """Send audio chunk to server"""
        if not self.is_connected:
            return
        
        encoded_audio = base64.b64encode(audio_data).decode('utf-8')
        
        audio_payload = AudioDataPayload(
            audio_data=encoded_audio,
            chunk_index=chunk_index,
            is_final=is_final,
            timestamp_offset=time.time() - self.audio_start_time
        )
        
        audio_msg = self.message_builder.audio_data_message(audio_payload)
        await self.websocket.send(audio_msg.to_json())
    
    def _preprocess_audio(self, audio_data: bytes) -> bytes:
        """Apply noise reduction and filtering"""
        if np is None:
            return audio_data
        
        try:
            # Convert to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Apply simple normalization
            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                audio_np = audio_np / max_val
            
            # Convert back to bytes
            return (audio_np * 32767).astype(np.int16).tobytes()
        except Exception:
            return audio_data
    
    def _has_speech(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk contains speech using VAD"""
        if self.vad is None:
            return True
        
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except Exception:
            return True
    
    async def _insert_text(self, text: str) -> None:
        """Insert text at cursor position"""
        # Mock implementation for testing
        logger.info(f"Inserting text: '{text}'")
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.audio:
            self.audio.terminate() 