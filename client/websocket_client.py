#!/usr/bin/env python3
"""
Enhanced MacBook Speech Client with Improved Audio Processing

This implements the macOS WebSocket client with enhanced features based on Option 3.
Enhanced with automatic reconnection and robust error handling (Task 1.4).
"""

import asyncio
import websockets
import json
import base64
import time
import threading
import uuid
import logging
from typing import Dict, Any, Optional, Callable, List
from enum import Enum

# Audio recording system
from .audio_recorder import AudioRecorder, AudioChunk, RecordingState, PermissionStatus

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


class ConnectionState(Enum):
    """Connection state enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class ClientConnectionManager:
    """Enhanced WebSocket connection manager with automatic reconnection and error handling"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.max_reconnect_attempts = 5
        self.initial_reconnect_delay = 1.0
        self.max_reconnect_delay = 30.0
        self.backoff_multiplier = 2.0
        
        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.websocket = None
        self.connection_attempts = 0
        self.last_connection_time = 0.0
        self.connection_start_time = 0.0
        
        # Health monitoring
        self.ping_interval = 20.0  # seconds
        self.ping_timeout = 10.0   # seconds
        self.last_ping_time = 0.0
        self.last_pong_time = 0.0
        self.missed_pings = 0
        self.max_missed_pings = 3
        
        # Error tracking
        self.last_error = ""
        self.error_count = 0
        self.consecutive_failures = 0
        
        # Message queue for offline scenarios
        self.message_queue: List[str] = []
        self.max_queue_size = 100
        
    async def connect_with_retry(self) -> Optional[websockets.WebSocketServerProtocol]:
        """Connect to server with enhanced retry logic and exponential backoff"""
        self.state = ConnectionState.CONNECTING
        self.connection_start_time = time.time()
        
        for attempt in range(self.max_reconnect_attempts):
            self.connection_attempts += 1
            
            try:
                logger.info(f"Connection attempt {attempt + 1}/{self.max_reconnect_attempts} to {self.server_url}")
                
                # Calculate backoff delay
                if attempt > 0:
                    delay = min(
                        self.initial_reconnect_delay * (self.backoff_multiplier ** (attempt - 1)),
                        self.max_reconnect_delay
                    )
                    logger.info(f"Waiting {delay:.1f}s before retry...")
                    await asyncio.sleep(delay)
                
                # Attempt connection with timeout
                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        self.server_url,
                        ping_interval=self.ping_interval,
                        ping_timeout=self.ping_timeout
                    ),
                    timeout=10.0  # 10-second connection timeout
                )
                
                # Connection successful
                self.state = ConnectionState.CONNECTED
                self.last_connection_time = time.time()
                self.consecutive_failures = 0
                self.last_error = ""
                
                logger.info(f"Successfully connected to {self.server_url}")
                
                # Send queued messages
                await self._process_message_queue()
                
                return self.websocket
                
            except asyncio.TimeoutError as e:
                error_msg = f"Connection timeout on attempt {attempt + 1}"
                logger.warning(error_msg)
                self.last_error = error_msg
                self.consecutive_failures += 1
                
            except ConnectionRefusedError as e:
                error_msg = f"Connection refused on attempt {attempt + 1}: {e}"
                logger.warning(error_msg)
                self.last_error = error_msg
                self.consecutive_failures += 1
                
            except OSError as e:
                error_msg = f"Network error on attempt {attempt + 1}: {e}"
                logger.warning(error_msg)
                self.last_error = error_msg
                self.consecutive_failures += 1
                
            except Exception as e:
                error_msg = f"Unexpected error on attempt {attempt + 1}: {e}"
                logger.error(error_msg)
                self.last_error = error_msg
                self.consecutive_failures += 1
        
        # All attempts failed
        self.state = ConnectionState.FAILED
        logger.error(f"Failed to connect after {self.max_reconnect_attempts} attempts")
        raise ConnectionError(f"Failed to connect after {self.max_reconnect_attempts} attempts. Last error: {self.last_error}")
    
    async def _process_message_queue(self):
        """Process queued messages after reconnection"""
        if not self.websocket or not self.message_queue:
            return
            
        logger.info(f"Processing {len(self.message_queue)} queued messages")
        
        # Send queued messages
        while self.message_queue and self.state == ConnectionState.CONNECTED:
            try:
                message = self.message_queue.pop(0)
                await self.websocket.send(message)
            except Exception as e:
                logger.error(f"Failed to send queued message: {e}")
                # Re-queue the message
                self.message_queue.insert(0, message)
                break
    
    async def queue_message(self, message: str):
        """Queue message for sending when connection is restored"""
        if len(self.message_queue) >= self.max_queue_size:
            # Remove oldest message to make room
            self.message_queue.pop(0)
            logger.warning("Message queue full, removing oldest message")
        
        self.message_queue.append(message)
        logger.debug(f"Message queued. Queue size: {len(self.message_queue)}")
    
    async def disconnect(self):
        """Gracefully disconnect from server"""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self.websocket = None
                self.state = ConnectionState.DISCONNECTED
    
    def is_connected(self) -> bool:
        """Check if currently connected"""
        return self.state == ConnectionState.CONNECTED and self.websocket is not None
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "state": self.state.value,
            "attempts": self.connection_attempts,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "queue_size": len(self.message_queue),
            "uptime": time.time() - self.last_connection_time if self.last_connection_time > 0 else 0
        }


class EnhancedSpeechClient:
    """Enhanced Speech Client with real audio recording capabilities"""
    
    def __init__(self, server_url: str = "ws://192.168.1.100:8765"):
        # Connection settings
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
        
        # Enhanced connection management
        self.connection_manager = ClientConnectionManager(server_url)
        self.message_builder = MessageBuilder("enhanced-macos-client", str(uuid.uuid4()))
        
        # Audio recording system - use real AudioRecorder
        self.audio_recorder = AudioRecorder(
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            chunk_size=CHUNK_SIZE
        )
        
        # Audio configuration
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.chunk_size = CHUNK_SIZE
        
        # Recording state
        self.is_recording = False
        self.audio_start_time = 0.0
        self.audio_processing_time = 0.0
        
        # Performance tracking
        self.last_ping_latency = 0.0
        self.ping_start_time = 0.0
        self.message_count = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.last_error = ""
        
        # Background tasks
        self.health_monitor_task = None
        self.reconnect_task = None
        
        # VAD (Voice Activity Detection)
        self.vad = None
        if webrtcvad:
            try:
                self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
            except Exception as e:
                logger.warning(f"VAD initialization failed: {e}")
        
        # Setup callbacks for audio recorder
        self._setup_audio_callbacks()
        
    def _setup_audio_callbacks(self):
        """Setup callbacks for the audio recorder"""
        def handle_audio_chunk(chunk: AudioChunk):
            """Handle new audio chunk from recorder"""
            if self.is_connected and self.is_recording:
                # Process chunk asynchronously
                asyncio.create_task(self._process_audio_chunk(chunk.data, chunk.chunk_index))
        
        def handle_audio_error(error: str):
            """Handle audio recording errors"""
            logger.error(f"Audio recording error: {error}")
            self.last_error = error
            self.consecutive_errors += 1
        
        def handle_audio_level(level_db: float, visual_level: str):
            """Handle audio level updates"""
            # Could be used for UI feedback
            logger.debug(f"Audio level: {level_db:.1f} dB {visual_level}")
        
        self.audio_recorder.set_callbacks(
            chunk_callback=handle_audio_chunk,
            error_callback=handle_audio_error,
            level_callback=handle_audio_level
        )

    async def connect(self) -> None:
        """Establish WebSocket connection to server with enhanced error handling"""
        try:
            self.websocket = await self.connection_manager.connect_with_retry()
            self.is_connected = True
            self.last_error = ""
            self.consecutive_errors = 0
            
            # Send client registration
            await self._send_connect_message()
            
            # Start enhanced message listener
            asyncio.create_task(self._enhanced_message_listener())
            
            # Start health monitoring
            if self.health_monitor_task:
                self.health_monitor_task.cancel()
            self.health_monitor_task = asyncio.create_task(self._health_monitor())
            
        except Exception as e:
            self.last_error = str(e)
            self.is_connected = False
            logger.error(f"Connection failed: {e}")
            raise
    
    async def _enhanced_message_listener(self) -> None:
        """Enhanced message listener with automatic reconnection"""
        try:
            async for message in self.websocket:
                await self._handle_incoming_message(message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}")
            self.is_connected = False
            await self._handle_connection_loss()
        except Exception as e:
            logger.error(f"Message listener error: {e}")
            self.consecutive_errors += 1
            if self.consecutive_errors < self.max_consecutive_errors:
                await self._handle_connection_loss()
            else:
                logger.error("Too many consecutive errors, stopping message listener")
                self.is_connected = False
    
    async def _handle_connection_loss(self) -> None:
        """Enhanced connection loss handling with automatic reconnection"""
        if self.reconnect_task and not self.reconnect_task.done():
            return  # Reconnection already in progress
        
        self.is_connected = False
        self.connection_manager.state = ConnectionState.RECONNECTING
        
        logger.info("Connection lost, attempting to reconnect...")
        
        try:
            # Cancel health monitor during reconnection
            if self.health_monitor_task:
                self.health_monitor_task.cancel()
            
            # Attempt reconnection
            await self.connect()
            
            logger.info("Successfully reconnected")
            
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            self.last_error = f"Reconnection failed: {e}"
            
            # Schedule retry after delay
            await asyncio.sleep(5.0)
            if not self.is_connected:
                self.reconnect_task = asyncio.create_task(self._handle_connection_loss())
    
    async def _health_monitor(self) -> None:
        """Monitor connection health with ping/pong"""
        while self.is_connected:
            try:
                await asyncio.sleep(30.0)
                
                if self.is_connected:
                    await self._send_ping()
                    
                    # Check for missed pongs
                    if (time.time() - self.ping_start_time) > 10.0:
                        self.connection_manager.missed_pings += 1
                        
                        if self.connection_manager.missed_pings >= self.connection_manager.max_missed_pings:
                            logger.warning("Too many missed pings, considering connection lost")
                            await self._handle_connection_loss()
                            break
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5.0)  # Brief pause before continuing
    
    async def _send_with_error_handling(self, message: str) -> bool:
        """Send message with enhanced error handling"""
        if not self.is_connected or not self.websocket:
            # Queue message for later sending
            await self.connection_manager.queue_message(message)
            return False
        
        try:
            await self.websocket.send(message)
            return True
        except websockets.exceptions.ConnectionClosed:
            self.is_connected = False
            await self.connection_manager.queue_message(message)
            await self._handle_connection_loss()
            return False
        except Exception as e:
            logger.error(f"Send error: {e}")
            self.consecutive_errors += 1
            await self.connection_manager.queue_message(message)
            return False
    
    async def _handle_incoming_message(self, raw_message: str) -> None:
        """Enhanced message handling with error recovery"""
        try:
            message = WebSocketMessage.from_json(raw_message)
            
            if message.header.message_type == MessageType.TRANSCRIPTION_RESULT:
                await self._handle_transcription_result(message.payload)
            elif message.header.message_type == MessageType.PONG:
                await self._handle_pong_message(message.payload)
            elif message.header.message_type == MessageType.ERROR:
                await self._handle_error_message(message.payload)
            elif message.header.message_type == MessageType.CONNECT_ACK:
                await self._handle_connect_ack(message.payload)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            self.consecutive_errors += 1
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self.consecutive_errors += 1
    
    async def _handle_transcription_result(self, payload: Dict[str, Any]) -> None:
        """Handle transcription result with error recovery"""
        try:
            text = payload.get("text", "")
            await self._insert_text(text)
            self.consecutive_errors = 0  # Reset error count on success
        except Exception as e:
            logger.error(f"Error handling transcription result: {e}")
            self.consecutive_errors += 1
    
    async def _handle_pong_message(self, payload: Dict[str, Any]) -> None:
        """Enhanced pong handling with latency tracking"""
        if self.ping_start_time > 0:
            current_time = time.time()
            self.last_ping_latency = (current_time - self.ping_start_time) * 1000
            self.ping_start_time = 0
            self.connection_manager.missed_pings = 0
            self.connection_manager.last_pong_time = current_time
    
    async def _handle_error_message(self, payload: Dict[str, Any]) -> None:
        """Enhanced error message handling"""
        error_code = payload.get("error_code", "UNKNOWN")
        error_message = payload.get("error_message", "Unknown error")
        self.last_error = f"{error_code}: {error_message}"
        
        logger.warning(f"Server error: {self.last_error}")
        
        # Handle specific error types
        if error_code in ["CONNECTION_ERROR", "INTERNAL_ERROR"]:
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.max_consecutive_errors:
                await self._handle_connection_loss()
    
    async def _handle_connect_ack(self, payload: Dict[str, Any]) -> None:
        """Handle connection acknowledgment"""
        logger.info("Connection acknowledged by server")
        self.consecutive_errors = 0
    
    async def _send_ping(self) -> None:
        """Enhanced ping with timeout tracking"""
        if not self.is_connected:
            return
        
        self.ping_start_time = time.time()
        ping_msg = self.message_builder.ping_message()
        success = await self._send_with_error_handling(ping_msg.to_json())
        
        if not success:
            logger.warning("Failed to send ping")
    
    async def _send_connect_message(self) -> None:
        """Enhanced connection message with session recovery"""
        client_info = ClientInfoPayload(
            client_name="Enhanced macOS Speech Client",
            client_version="1.0.0",
            platform="macOS",
            capabilities=["audio_recording", "text_insertion", "auto_reconnect"],
            status=ClientStatus.CONNECTED
        )
        
        connect_msg = self.message_builder.connect_message(client_info)
        success = await self._send_with_error_handling(connect_msg.to_json())
        
        if not success:
            logger.error("Failed to send connect message")
            raise ConnectionError("Failed to register with server")
    
    def _setup_hotkeys(self) -> None:
        """Setup global hotkey listeners"""
        # Mock implementation for testing
        logger.info("Setting up hotkeys...")
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics"""
        connection_stats = self.connection_manager.get_connection_stats()
        audio_status = self.audio_recorder.get_status_info()
        
        return {
            "latency_ms": self.last_ping_latency,
            "audio_processing_time": self.audio_processing_time,
            "timestamp": time.time(),
            "connection_state": connection_stats["state"],
            "consecutive_errors": self.consecutive_errors,
            "queue_size": connection_stats["queue_size"],
            "uptime": connection_stats["uptime"],
            "audio_state": audio_status["state"],
            "audio_level_db": audio_status["audio_level_db"],
            "audio_buffer_size": audio_status["buffer_size"],
            "audio_chunk_count": audio_status["chunk_count"]
        }
    
    def _record_audio_chunk(self) -> bytes:
        """Record a single audio chunk using the real AudioRecorder"""
        chunk = self.audio_recorder.get_audio_chunk(timeout=0.05)
        if chunk:
            return chunk.data
        else:
            # Return silence if no audio available
            return b'\x00' * (self.chunk_size * 2)  # 16-bit silence

    async def _process_audio_chunk(self, audio_data: bytes, chunk_index: int, is_final: bool = False) -> None:
        """Process and send audio chunk with error handling"""
        try:
            # Apply preprocessing
            processed_audio = self._preprocess_audio(audio_data)
            
            # Check for speech activity (optional optimization)
            if self._has_speech(processed_audio):
                await self._send_audio_chunk(processed_audio, chunk_index, is_final)
            elif is_final:
                # Always send final chunk even if no speech detected
                await self._send_audio_chunk(processed_audio, chunk_index, is_final)
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            self.consecutive_errors += 1

    async def start_recording(self) -> None:
        """Start audio recording with connection check"""
        if self.is_recording or not self.is_connected:
            logger.warning("Cannot start recording: already recording or not connected")
            return
        
        # Initialize audio recorder if not done
        if not await self._ensure_audio_initialized():
            logger.error("Failed to initialize audio recorder")
            return
        
        # Start the audio recorder
        if not self.audio_recorder.start_recording():
            logger.error("Failed to start audio recorder")
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
        success = await self._send_with_error_handling(start_msg.to_json())
        
        if not success:
            self.is_recording = False
            self.audio_recorder.stop_recording()
            logger.error("Failed to start recording - connection issue")

    async def stop_recording(self) -> None:
        """Stop audio recording with proper cleanup"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Stop the audio recorder
        self.audio_recorder.stop_recording()
        
        # Calculate processing time
        if self.audio_start_time > 0:
            self.audio_processing_time = time.time() - self.audio_start_time
            self.audio_start_time = 0
        
        # Send audio end message
        end_msg = self.message_builder.audio_end_message()
        await self._send_with_error_handling(end_msg.to_json())

    async def _ensure_audio_initialized(self) -> bool:
        """Ensure audio recorder is initialized"""
        if not hasattr(self.audio_recorder, 'audio') or self.audio_recorder.audio is None:
            return await self.audio_recorder.initialize()
        return True

    def _preprocess_audio(self, audio_data: bytes) -> bytes:
        """Preprocess audio data with error handling"""
        try:
            if np is None:
                return audio_data
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Apply basic audio processing
            # 1. Normalize audio levels
            if len(audio_array) > 0:
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    # Normalize to 80% of max range to prevent clipping
                    audio_array = (audio_array * 0.8 * 32767 / max_val).astype(np.int16)
            
            # 2. Apply simple noise gate (remove very quiet audio)
            noise_floor = 500  # Adjust based on testing
            audio_array = np.where(np.abs(audio_array) < noise_floor, 0, audio_array)
            
            return audio_array.tobytes()
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return audio_data

    def _has_speech(self, audio_chunk: bytes) -> bool:
        """Detect speech in audio chunk using WebRTC VAD"""
        if self.vad is None:
            return True  # Assume speech if VAD not available
        
        try:
            # VAD requires specific format: 16kHz, 16-bit, mono
            # Chunk must be 10ms, 20ms, or 30ms
            chunk_duration_ms = (len(audio_chunk) // 2) * 1000 // self.sample_rate
            
            if chunk_duration_ms in [10, 20, 30]:
                return self.vad.is_speech(audio_chunk, self.sample_rate)
            else:
                # If chunk is wrong size, assume speech
                return True
                
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True

    async def _send_audio_chunk(self, audio_data: bytes, chunk_index: int, is_final: bool = False) -> None:
        """Send audio chunk with enhanced error handling"""
        if not self.is_connected:
            logger.warning("Cannot send audio chunk - not connected")
            return
        
        encoded_audio = base64.b64encode(audio_data).decode('utf-8')
        audio_payload = AudioDataPayload(
            audio_data=encoded_audio,
            chunk_index=chunk_index,
            is_final=is_final
        )
        
        audio_msg = self.message_builder.audio_data_message(audio_payload)
        success = await self._send_with_error_handling(audio_msg.to_json())
        
        if not success:
            logger.warning(f"Failed to send audio chunk {chunk_index}")
    
    async def _insert_text(self, text: str) -> None:
        """Insert text using accessibility API with error handling"""
        try:
            # Mock implementation for testing
            logger.info(f"Inserting text: {text}")
        except Exception as e:
            logger.error(f"Text insertion error: {e}")
    
    def cleanup(self) -> None:
        """Enhanced cleanup with proper resource management"""
        logger.info("Starting client cleanup...")
        
        # Cancel background tasks
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        if self.reconnect_task:
            self.reconnect_task.cancel()
        
        # Stop recording
        if self.is_recording:
            asyncio.create_task(self.stop_recording())
        
        # Disconnect
        self.is_connected = False
        asyncio.create_task(self.connection_manager.disconnect())
        
        # Cleanup audio recorder
        self.audio_recorder.cleanup()
        
        logger.info("Client cleanup completed") 