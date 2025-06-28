"""
Mock WebSocket Server for Docker Testing

This module provides a comprehensive mock implementation of the Pi-Whispr WebSocket server
for development and testing purposes. It implements the complete protocol from task 1
with the same message handling as the real server but without requiring faster-whisper.

Features:
- Complete WebSocket protocol implementation matching the real server
- All message types: connection, audio, transcription, status, error, ping/pong, client management, performance tracking
- Enhanced faster-whisper simulation with realistic timing and responses (Task 2.3)
- Audio chunk accumulation and processing pipeline simulation
- Model-specific processing times (tiny.en vs small.en)
- Realistic transcription generation with confidence scoring
- Resource usage simulation (memory, temperature, CPU)
- Error injection and queue management simulation
- Client connection management and registration
- Configurable latency simulation and error scenarios
- Performance monitoring and metrics
- Concurrent client support
"""

import asyncio
import websockets
import json
import base64
import time
import logging
import argparse
import random
import math
import functools
from typing import Dict, Any, Optional, Set, List
from pathlib import Path
import uuid

# Import shared modules for protocol compatibility
from shared.protocol import (
    MessageType, Priority, ClientStatus,
    MessageHeader, WebSocketMessage,
    AudioConfigPayload, AudioDataPayload, TranscriptionResultPayload,
    PerformanceMetricsPayload, ClientInfoPayload, ErrorPayload,
    MessageBuilder, MessageValidator
)
from shared.constants import WEBSOCKET_HOST, WEBSOCKET_PORT, DEFAULT_MODEL
from shared.exceptions import NetworkError, TranscriptionError

# Import configuration system
from server.mock_configuration import MockServerConfig, ConfigurationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global handler function for websockets 15.x compatibility
_server_instance = None

async def websocket_handler(websocket):
    """Global WebSocket handler function for websockets 15.x compatibility"""
    if _server_instance is None:
        await websocket.close(code=1011, reason="Server not initialized")
        return
    # websockets 15.x only passes websocket, not path
    # Use empty string for path since it's not needed for our protocol
    await _server_instance._handle_client_connection(websocket, "")


class MockWhisperServer:
    """Mock WebSocket server that fully implements the Pi-Whispr protocol"""
    
    def __init__(self, config: MockServerConfig = None, host: str = None, port: int = None, 
                 latency_ms: int = None, model_size: str = None):
        # Support both new config-based interface and legacy parameter interface
        if config is not None:
            # New configuration-based interface
            self.config = config
        elif any(param is not None for param in [host, port, latency_ms, model_size]):
            # Legacy parameter interface - create config from individual parameters
            from server.mock_configuration import LatencyProfile, ErrorScenario, ResourceConstraints
            
            latency_profile = LatencyProfile(
                processing_delay_ms=latency_ms or 0,
                network_delay_ms=10,
                variability_factor=0.2
            )
            
            error_scenario = ErrorScenario(
                failure_rate=0.0,
                timeout_rate=0.0,
                processing_error_rate=0.0,
                error_types=[]
            )
            
            resource_constraints = ResourceConstraints(
                max_memory_mb=2048,
                max_cpu_percent=80,
                max_concurrent_connections=10,
                memory_pressure_threshold=0.75,
                cpu_throttle_threshold=0.75
            )
            
            self.config = MockServerConfig(
                host=host or "localhost",
                port=port or 8765,
                model_size=model_size or "tiny.en",
                latency_profile=latency_profile,
                error_scenario=error_scenario,
                resource_constraints=resource_constraints,
                enable_detailed_logging=False
            )
        else:
            # No config or parameters provided - try environment or use default
            try:
                self.config = ConfigurationManager.load_from_environment()
            except Exception as e:
                logger.warning(f"Failed to load configuration from environment: {e}")
                logger.info("Using default configuration")
                self.config = MockServerConfig.default()
        
        # Configure logging level based on config
        if self.config.enable_detailed_logging:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        self.host = self.config.host
        self.port = self.config.port
        self.model_size = self.config.model_size
        
        # Server state
        self._start_time = time.time()
        self._model_loaded = True  # Mock server always has model "loaded"
        
        # Client management - matches real server structure
        self.clients: Dict[str, Dict[str, Any]] = {}
        
        # Mock statistics - matches real server
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_transcriptions": 0,
            "avg_processing_time": 0.0,
            "error_count": 0,
            "uptime_seconds": 0.0
        }
        
        # Mock transcription responses for variety
        self.mock_transcriptions = [
            "Hello, this is a mock transcription.",
            "Testing the WebSocket audio processing system.",
            "Mock faster-whisper transcription result.",
            "The quick brown fox jumps over the lazy dog.",
            "This is simulated speech-to-text output for development.",
            "Voice activity detection is working correctly.",
            "Real-time audio streaming and transcription.",
            "WebSocket communication protocol test message."
        ]
        
        # Enhanced faster-whisper simulation attributes (Task 2.3)
        self.audio_buffers: Dict[str, List[Dict[str, Any]]] = {}
        self.enable_partial_results = True
        self.processing_queue: List[str] = []
        self.error_injection_rate = self.config.error_scenario.failure_rate
        self._latency_ms = self.config.latency_profile.processing_delay_ms
        
        # Model characteristics for realistic simulation
        self.model_characteristics = {
            "tiny.en": {
                "base_processing_ratio": 0.15,  # 15% of audio duration
                "memory_usage": 39,  # MB
                "processing_variance": 0.3,
                "startup_time": 0.5,
                "accuracy_range": (0.85, 0.92)
            },
            "small.en": {
                "base_processing_ratio": 0.35,  # 35% of audio duration
                "memory_usage": 244,  # MB
                "processing_variance": 0.25,
                "startup_time": 1.0,
                "accuracy_range": (0.88, 0.96)
            },
            "base.en": {
                "base_processing_ratio": 0.65,  # 65% of audio duration
                "memory_usage": 390,  # MB
                "processing_variance": 0.2,
                "startup_time": 1.5,
                "accuracy_range": (0.90, 0.97)
            },
            "medium.en": {
                "base_processing_ratio": 1.2,  # 120% of audio duration
                "memory_usage": 769,  # MB
                "processing_variance": 0.2,
                "startup_time": 2.0,
                "accuracy_range": (0.92, 0.98)
            },
            "large": {
                "base_processing_ratio": 2.0,  # 200% of audio duration
                "memory_usage": 1550,  # MB
                "processing_variance": 0.15,
                "startup_time": 3.0,
                "accuracy_range": (0.94, 0.99)
            }
        }
        
        # Resource simulation state
        self._processing_load = 0.0  # Current processing load (0.0 to 1.0)
        self._baseline_memory_usage = 45.0  # Base memory usage percentage
        self._current_memory_usage = self._baseline_memory_usage
        self._baseline_temperature = 40.0  # Base CPU temperature in Celsius
        self._current_temperature = self._baseline_temperature
        self._last_processing_time = time.time()
        
        logger.info(f"MockWhisperServer initialized with configuration:")
        logger.info(f"  - Host: {self.host}:{self.port}")
        logger.info(f"  - Model: {self.model_size}")
        logger.info(f"  - Latency Profile: {self.config.latency_profile.processing_delay_ms}ms processing, {self.config.latency_profile.network_delay_ms}ms network")
        logger.info(f"  - Error Scenario: {self.config.error_scenario.failure_rate*100:.1f}% failure rate")
        logger.info(f"  - Resource Constraints: {self.config.resource_constraints.max_concurrent_connections} max connections")
        logger.info(f"  - Detailed Logging: {self.config.enable_detailed_logging}")
    
    @property
    def latency_ms(self) -> int:
        """Get the current latency in milliseconds (backward compatibility)"""
        return self._latency_ms
    
    @latency_ms.setter
    def latency_ms(self, value: int):
        """Set the latency in milliseconds (backward compatibility)"""
        self._latency_ms = value
        # Update the configuration to maintain consistency
        self.config.latency_profile.processing_delay_ms = value
        # Ensure no variability for legacy tests
        self.config.latency_profile.variability_factor = 0.0
    
    async def _connection_handler_wrapper(self, websocket, path):
        """Wrapper method for WebSocket connection handling"""
        await self._handle_client_connection(websocket, path)
    
    async def start(self) -> None:
        """Start the mock WebSocket server"""
        global _server_instance
        _server_instance = self
        
        logger.info(f"Starting Mock WebSocket server on {self.host}:{self.port}")
        
        try:
            async with websockets.serve(
                websocket_handler,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10
            ):
                logger.info("Mock WebSocket server started successfully")
                await asyncio.Future()  # Run forever
        except Exception as e:
            logger.error(f"Failed to start Mock WebSocket server: {e}")
            raise
        finally:
            _server_instance = None
    
    async def _handle_client_connection(self, websocket, path: str) -> None:
        """Handle new client WebSocket connection"""
        client_address = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New mock client connection from {client_address}, path: {path}")
        
        # Check connection limit
        if self.config.resource_constraints.is_connection_limit_exceeded(self.stats["active_connections"]):
            logger.warning(f"Connection limit exceeded, rejecting client {client_address}")
            await websocket.close(code=1013, reason="Server overloaded")
            return
        
        self.stats["total_connections"] += 1
        self.stats["active_connections"] += 1
        
        try:
            async for message in websocket:
                await self._handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Mock client {client_address} disconnected normally")
        except Exception as e:
            logger.error(f"Error handling mock client {client_address}: {e}")
            await self._send_error(websocket, "CONNECTION_ERROR", str(e))
        finally:
            self.stats["active_connections"] -= 1
            # Clean up client if registered
            await self._cleanup_disconnected_client(websocket)
    
    async def _handle_message(self, websocket, raw_message: str) -> None:
        """Handle incoming WebSocket message using proper protocol validation"""
        try:
            # Parse JSON first to catch JSON syntax errors specifically
            json.loads(raw_message)  # This will raise JSONDecodeError for invalid JSON
        except json.JSONDecodeError as e:
            await self._send_error(websocket, "INVALID_JSON", f"JSON parsing error: {e}")
            return
        
        try:
            # Inject connection failure if configured
            if self.config.error_scenario.should_inject_failure():
                logger.debug("Injecting connection failure")
                await websocket.close(code=1011, reason="Simulated connection failure")
                return
            
            # Now create the message object and validate structure
            message = WebSocketMessage.from_json(raw_message)
            
            # Check if message is expired - if so, silently ignore it
            if message.is_expired():
                logger.warning(f"Received expired message: {message.header.sequence_id}")
                return
            
            # Validate the parsed message structure
            if not MessageValidator.validate_message_json(raw_message):
                await self._send_error(websocket, "INVALID_MESSAGE", "Message validation failed")
                return
            
            # Apply configured latency
            latency_ms = self.config.latency_profile.calculate_actual_latency()
            if latency_ms > 0:
                logger.debug(f"Applying {latency_ms}ms latency to message processing")
                await asyncio.sleep(latency_ms / 1000.0)
            
            # Inject timeout error if configured
            if self.config.error_scenario.should_inject_timeout():
                logger.debug("Injecting timeout error")
                await self._send_error(websocket, "TIMEOUT_ERROR", "Simulated request timeout")
                return
            
            # Route message based on type
            await self._route_message(websocket, message)
            
        except ValueError as e:
            # This catches structure validation errors from WebSocketMessage.from_json
            await self._send_error(websocket, "INVALID_MESSAGE", f"Invalid message format: {e}")
        except Exception as e:
            logger.error(f"Unexpected error handling mock message: {e}")
            self.stats["error_count"] += 1
            await self._send_error(websocket, "INTERNAL_ERROR", "Internal server error")
    
    async def _route_message(self, websocket, message: WebSocketMessage) -> None:
        """Route message to appropriate handler based on message type"""
        handlers = {
            MessageType.CONNECT: self._handle_client_registration,
            MessageType.DISCONNECT: self._handle_client_disconnect_msg,
            MessageType.AUDIO_DATA: self._handle_audio_data,
            MessageType.AUDIO_START: self._handle_audio_start,
            MessageType.AUDIO_END: self._handle_audio_end,
            MessageType.PING: self._handle_ping,
            MessageType.STATUS_REQUEST: self._handle_status_request,
            MessageType.CLIENT_LIST_REQUEST: self._handle_client_list_request,
            MessageType.PERFORMANCE_METRICS: self._handle_performance_request,
            MessageType.HEALTH_CHECK: self._handle_health_check,
        }
        
        handler = handlers.get(message.header.message_type, self._handle_unknown)
        await handler(websocket, message)
    
    async def _handle_client_registration(self, websocket, message: WebSocketMessage) -> None:
        """Handle client connection/registration using proper protocol"""
        try:
            # Extract client info from payload
            payload = message.payload
            client_info = ClientInfoPayload(
                client_name=payload.get("client_name", "Mock Client"),
                client_version=payload.get("client_version", "1.0.0"),
                platform=payload.get("platform", "mock"),
                capabilities=payload.get("capabilities", ["transcription"]),
                status=ClientStatus(payload.get("status", "connected"))
            )
            
            client_id = message.header.client_id
            
            # Register client with same structure as real server
            self.clients[client_id] = {
                "websocket": websocket,
                "info": client_info,
                "status": ClientStatus.CONNECTED,
                "connected_at": time.time(),
                "last_seen": time.time(),
                "session_id": message.header.session_id
            }
            
            # Create acknowledgment response
            response_header = MessageHeader(
                message_type=MessageType.CONNECT_ACK,
                sequence_id=message.header.sequence_id + 1,
                timestamp=time.time(),
                client_id=client_id,
                session_id=message.header.session_id,
                correlation_id=message.header.correlation_id
            )
            
            response_payload = {
                "status": "connected",
                "client_id": client_id,
                "server_capabilities": ["transcription", "audio_processing", "performance_monitoring"],
                "model_loaded": True,
                "model_name": f"mock-whisper-{self.model_size}"
            }
            
            response = WebSocketMessage(header=response_header, payload=response_payload)
            await websocket.send(response.to_json())
            
            logger.info(f"Mock client {client_id} registered successfully")
            
        except Exception as e:
            logger.error(f"Error in mock client registration: {e}")
            await self._send_error(websocket, "REGISTRATION_ERROR", str(e), message.header.correlation_id)
    
    async def _handle_audio_data(self, websocket, message: WebSocketMessage) -> None:
        """Enhanced audio data handling with realistic faster-whisper simulation (Task 2.3)"""
        try:
            client_id = message.header.client_id
            
            # Verify client is registered
            if client_id not in self.clients:
                await self._send_error(websocket, "CLIENT_NOT_REGISTERED", 
                                     "Client must register before sending audio data")
                return
            
            # Check for error injection
            if self._should_inject_error():
                await self._send_error(websocket, "PROCESSING_ERROR", 
                                     "Simulated processing error", message.header.correlation_id)
                return
            
            # Update client status
            self.clients[client_id]["status"] = ClientStatus.PROCESSING
            self.clients[client_id]["last_seen"] = time.time()
            
            # Extract audio data info
            payload = message.payload
            chunk_index = payload.get("chunk_index", 0)
            is_final = payload.get("is_final", False)
            audio_data = payload.get("audio_data", "")
            trigger_partial = payload.get("trigger_partial", False)
            
            # Initialize audio buffer for client if needed
            if client_id not in self.audio_buffers:
                self.audio_buffers[client_id] = []
            
            # Store audio chunk with metadata
            chunk_data = {
                "chunk_index": chunk_index,
                "audio_data": audio_data,
                "timestamp": time.time(),
                "is_final": is_final
            }
            self.audio_buffers[client_id].append(chunk_data)
            
            # Handle partial results if enabled and requested
            if self.enable_partial_results and trigger_partial and not is_final:
                await self._send_partial_transcription(websocket, message, client_id)
            
            # Process final chunk or when buffer is sufficient
            if is_final:
                await self._process_audio_buffer(websocket, message, client_id)
                # Clear buffer after processing
                self.audio_buffers[client_id] = []
            
            # Update client status back to connected
            self.clients[client_id]["status"] = ClientStatus.CONNECTED
            
        except Exception as e:
            logger.error(f"Error processing enhanced audio data: {e}")
            await self._send_error(websocket, "AUDIO_PROCESSING_ERROR", str(e), message.header.correlation_id)
    
    async def _process_audio_buffer(self, websocket, message: WebSocketMessage, client_id: str) -> None:
        """Process accumulated audio buffer with realistic timing simulation"""
        try:
            # Calculate total audio duration from buffer
            audio_chunks = self.audio_buffers.get(client_id, [])
            estimated_duration = len(audio_chunks) * 0.02  # Assuming 20ms chunks
            
            # Simulate processing with realistic timing
            processing_time = self._calculate_processing_time_for_duration(estimated_duration)
            
            # Add queue delay if there are concurrent requests
            queue_delay = self._calculate_queue_delay()
            total_processing_time = processing_time + queue_delay
            
            # Update resource usage simulation
            self._simulate_processing_load(estimated_duration)
            
            # Simulate actual processing delay
            await asyncio.sleep(total_processing_time)
            
            # Generate realistic transcription
            transcription_text = self._generate_realistic_transcription()
            confidence_score = self._generate_confidence_score()
            
            # Update statistics
            self.stats["total_transcriptions"] += 1
            self._update_processing_time_avg(total_processing_time)
            
            # Create transcription result
            result_payload = TranscriptionResultPayload(
                text=transcription_text,
                confidence=confidence_score,
                processing_time=total_processing_time,
                model_used=f"mock-whisper-{self.model_size}",
                language="en",
                audio_duration=estimated_duration,
                is_partial=False
            )
            
            response_header = MessageHeader(
                message_type=MessageType.TRANSCRIPTION_RESULT,
                sequence_id=message.header.sequence_id + 1,
                timestamp=time.time(),
                client_id=client_id,
                session_id=message.header.session_id,
                correlation_id=message.header.correlation_id
            )
            
            response = WebSocketMessage(
                header=response_header, 
                payload=result_payload.to_dict()
            )
            await websocket.send(response.to_json())
            
            logger.info(f"Enhanced transcription sent for client {client_id}: '{transcription_text}' "
                       f"(confidence: {confidence_score:.3f}, time: {total_processing_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"Error in audio buffer processing: {e}")
            await self._send_error(websocket, "BUFFER_PROCESSING_ERROR", str(e), message.header.correlation_id)
    
    async def _send_partial_transcription(self, websocket, message: WebSocketMessage, client_id: str) -> None:
        """Send partial transcription result for streaming simulation"""
        try:
            # Generate partial result (shorter, less confident)
            partial_text = self._generate_partial_transcription()
            partial_confidence = self._generate_confidence_score() * 0.8  # Lower confidence for partial
            
            result_payload = TranscriptionResultPayload(
                text=partial_text,
                confidence=partial_confidence,
                processing_time=0.1,  # Very fast for partial results
                model_used=f"mock-whisper-{self.model_size}",
                language="en",
                audio_duration=0.5,  # Estimated partial duration
                is_partial=True
            )
            
            response_header = MessageHeader(
                message_type=MessageType.TRANSCRIPTION_RESULT,
                sequence_id=message.header.sequence_id + 1,
                timestamp=time.time(),
                client_id=client_id,
                session_id=message.header.session_id,
                correlation_id=message.header.correlation_id
            )
            
            response = WebSocketMessage(
                header=response_header, 
                payload=result_payload.to_dict()
            )
            await websocket.send(response.to_json())
            
            logger.debug(f"Partial transcription sent for client {client_id}: '{partial_text}'")
            
        except Exception as e:
            logger.error(f"Error sending partial transcription: {e}")
    
    def _calculate_processing_time_for_duration(self, audio_duration: float) -> float:
        """Calculate realistic processing time based on audio duration and model characteristics"""
        characteristics = self.model_characteristics.get(self.model_size, self.model_characteristics["tiny.en"])
        
        # Base processing time proportional to audio duration
        base_time = audio_duration * characteristics["base_processing_ratio"]
        
        # Add variance to simulate real-world conditions
        variance = characteristics["processing_variance"]
        variation = random.uniform(-variance, variance) * base_time
        
        # Add minimum processing time and model startup overhead
        min_time = 0.1 + characteristics["startup_time"] * 0.1  # Reduced startup for ongoing processing
        
        final_time = max(min_time, base_time + variation)
        
        # Add small random component for network/system jitter
        jitter = random.uniform(0.01, 0.05)
        
        return final_time + jitter
    
    def _calculate_processing_time_with_queue(self, audio_duration: float, current_queue_size: int) -> float:
        """Calculate processing time including queue delays for concurrent requests"""
        base_time = self._calculate_processing_time_for_duration(audio_duration)
        
        # Add queue delay based on current load (simulating Pi 5 resource constraints)
        queue_delay = current_queue_size * 0.1  # 100ms delay per queued item
        
        return base_time + queue_delay
    
    def _calculate_queue_delay(self) -> float:
        """Calculate current queue delay based on processing load"""
        current_queue_size = len(self.processing_queue)
        
        # Simulate Pi 5 processing limitations
        if current_queue_size == 0:
            return 0.0
        elif current_queue_size <= 2:
            return current_queue_size * 0.05  # 50ms per item for small queue
        else:
            # Exponential backoff for larger queues
            return 0.1 + (current_queue_size - 2) * 0.15
    
    def _generate_realistic_transcription(self) -> str:
        """Generate realistic transcription text with appropriate variety and length"""
        # Choose base transcription
        base_text = random.choice(self.mock_transcriptions)
        
        # Occasionally modify or combine transcriptions for variety
        if random.random() < 0.3:  # 30% chance of modification
            if random.random() < 0.5:
                # Add slight variation
                variations = [
                    f"{base_text} Thank you.",
                    f"Um, {base_text.lower()}",
                    f"{base_text} Please note that.",
                    f"So, {base_text.lower()}"
                ]
                base_text = random.choice(variations)
            else:
                # Combine with another transcription
                second_text = random.choice(self.mock_transcriptions)
                if second_text != base_text:
                    base_text = f"{base_text} {second_text}"
        
        # Ensure reasonable length (trim if too long)
        if len(base_text) > 200:
            base_text = base_text[:197] + "..."
        
        return base_text
    
    def _generate_partial_transcription(self) -> str:
        """Generate partial transcription (typically shorter and incomplete)"""
        full_transcriptions = [
            "Hello this is",
            "Testing the",
            "The quick brown",
            "Voice activity",
            "Real-time audio",
            "Machine learning",
            "Python is an"
        ]
        return random.choice(full_transcriptions)
    
    def _generate_confidence_score(self) -> float:
        """Generate realistic confidence score based on model characteristics"""
        characteristics = self.model_characteristics.get(self.model_size, self.model_characteristics["tiny.en"])
        min_conf, max_conf = characteristics["accuracy_range"]
        
        # Generate score with normal distribution centered in the range
        center = (min_conf + max_conf) / 2
        std_dev = (max_conf - min_conf) / 6  # 99.7% within range
        
        score = random.gauss(center, std_dev)
        
        # Ensure within bounds
        return max(min_conf, min(max_conf, score))
    
    def _should_inject_error(self) -> bool:
        """Determine if an error should be injected based on error rate"""
        return random.random() < self.error_injection_rate
    
    def _simulate_processing_load(self, audio_duration: float) -> None:
        """Simulate processing load effects on system resources"""
        # Increase current processing load
        characteristics = self.model_characteristics.get(self.model_size, self.model_characteristics["tiny.en"])
        
        # Calculate load based on audio duration and model size
        load_increase = min(0.8, audio_duration * 0.2 + characteristics["memory_usage"] / 500.0)
        self._processing_load = min(1.0, self._processing_load + load_increase)
        
        # Update memory usage
        memory_increase = characteristics["memory_usage"] / 10.0  # Convert MB to percentage
        self._current_memory_usage = min(90.0, self._baseline_memory_usage + memory_increase + 
                                       self._processing_load * 20.0)
        
        # Update temperature
        self._simulate_processing_heat()
        
        self._last_processing_time = time.time()
    
    def _simulate_processing_heat(self) -> None:
        """Simulate CPU temperature increase during processing"""
        # Temperature increases with processing load
        temp_increase = self._processing_load * 15.0  # Up to 15°C increase under full load
        target_temp = self._baseline_temperature + temp_increase
        
        # Gradual temperature change (thermal inertia simulation)
        temp_diff = target_temp - self._current_temperature
        self._current_temperature += temp_diff * 0.3  # 30% change per update
        
        # Add some random variation
        self._current_temperature += random.uniform(-1.0, 1.0)
        
        # Ensure reasonable bounds
        self._current_temperature = max(25.0, min(85.0, self._current_temperature))
    
    def _get_simulated_memory_usage(self) -> float:
        """Get current simulated memory usage percentage"""
        # Gradually decay memory usage over time
        time_since_processing = time.time() - self._last_processing_time
        decay_factor = min(1.0, time_since_processing / 30.0)  # 30 second decay
        
        current_usage = self._current_memory_usage - (self._current_memory_usage - self._baseline_memory_usage) * decay_factor
        self._current_memory_usage = max(self._baseline_memory_usage, current_usage)
        
        return self._current_memory_usage
    
    def _get_simulated_temperature(self) -> float:
        """Get current simulated CPU temperature"""
        # Gradually cool down over time
        time_since_processing = time.time() - self._last_processing_time
        cooling_factor = min(1.0, time_since_processing / 60.0)  # 60 second cooling
        
        self._current_temperature -= (self._current_temperature - self._baseline_temperature) * cooling_factor * 0.1
        self._current_temperature = max(self._baseline_temperature, self._current_temperature)
        
        return self._current_temperature
    
    async def _register_test_client(self, client_id: str, websocket) -> None:
        """Register a test client (helper method for testing)"""
        client_info = ClientInfoPayload(
            client_name="Test Client",
            client_version="1.0.0-test",
            platform="test",
            capabilities=["transcription"],
            status=ClientStatus.CONNECTED
        )
        
        self.clients[client_id] = {
            "info": client_info,
            "websocket": websocket,
            "status": ClientStatus.CONNECTED,
            "connected_at": time.time(),
            "last_seen": time.time()
        }
    
    async def _handle_audio_start(self, websocket, message: WebSocketMessage) -> None:
        """Handle audio stream start"""
        client_id = message.header.client_id
        if client_id in self.clients:
            self.clients[client_id]["status"] = ClientStatus.RECORDING
            logger.debug(f"Mock client {client_id} started audio stream")
    
    async def _handle_audio_end(self, websocket, message: WebSocketMessage) -> None:
        """Handle audio stream end"""
        client_id = message.header.client_id
        if client_id in self.clients:
            self.clients[client_id]["status"] = ClientStatus.CONNECTED
            logger.debug(f"Mock client {client_id} ended audio stream")
    
    async def _handle_ping(self, websocket, message: WebSocketMessage) -> None:
        """Handle ping message with pong response"""
        response_header = MessageHeader(
            message_type=MessageType.PONG,
            sequence_id=message.header.sequence_id + 1,
            timestamp=time.time(),
            client_id=message.header.client_id,
            session_id=message.header.session_id,
            correlation_id=message.header.correlation_id
        )
        
        response_payload = {
            "server_time": time.time(),
            "uptime": time.time() - self._start_time
        }
        
        response = WebSocketMessage(header=response_header, payload=response_payload)
        await websocket.send(response.to_json())
        logger.debug("Sent mock pong response")
    
    async def _handle_status_request(self, websocket, message: WebSocketMessage) -> None:
        """Handle server status request"""
        self.stats["uptime_seconds"] = time.time() - self._start_time
        
        response_header = MessageHeader(
            message_type=MessageType.STATUS_RESPONSE,
            sequence_id=message.header.sequence_id + 1,
            timestamp=time.time(),
            client_id=message.header.client_id,
            session_id=message.header.session_id,
            correlation_id=message.header.correlation_id
        )
        
        response_payload = {
            "status": "running",
            "server_type": "mock",
            "model_loaded": self._model_loaded,
            "model_name": f"mock-whisper-{self.model_size}",
            "version": "1.0.0-mock",
            "statistics": self.stats.copy(),
            "active_clients": len(self.clients),
            "supported_capabilities": ["transcription", "audio_processing", "performance_monitoring"]
        }
        
        response = WebSocketMessage(header=response_header, payload=response_payload)
        await websocket.send(response.to_json())
        logger.debug("Sent mock status response")
    
    async def _handle_client_list_request(self, websocket, message: WebSocketMessage) -> None:
        """Handle client list request"""
        client_list = []
        for client_id, client_data in self.clients.items():
            client_info = {
                "client_id": client_id,
                "client_name": client_data["info"].client_name,
                "status": client_data["status"].value,
                "connected_at": client_data["connected_at"],
                "last_seen": client_data["last_seen"]
            }
            client_list.append(client_info)
        
        response_header = MessageHeader(
            message_type=MessageType.CLIENT_LIST_RESPONSE,
            sequence_id=message.header.sequence_id + 1,
            timestamp=time.time(),
            client_id=message.header.client_id,
            session_id=message.header.session_id,
            correlation_id=message.header.correlation_id
        )
        
        response_payload = {
            "clients": client_list,
            "total_clients": len(client_list)
        }
        
        response = WebSocketMessage(header=response_header, payload=response_payload)
        await websocket.send(response.to_json())
        logger.debug(f"Sent mock client list with {len(client_list)} clients")
    
    async def _handle_performance_request(self, websocket, message: WebSocketMessage) -> None:
        """Handle performance metrics request"""
        metrics = self._get_mock_performance_metrics()
        
        response_header = MessageHeader(
            message_type=MessageType.PERFORMANCE_METRICS,
            sequence_id=message.header.sequence_id + 1,
            timestamp=time.time(),
            client_id=message.header.client_id,
            session_id=message.header.session_id,
            correlation_id=message.header.correlation_id
        )
        
        response = WebSocketMessage(header=response_header, payload=metrics.to_dict())
        await websocket.send(response.to_json())
        logger.debug("Sent mock performance metrics")
    
    async def _handle_health_check(self, websocket, message: WebSocketMessage) -> None:
        """Handle health check request"""
        response_header = MessageHeader(
            message_type=MessageType.HEALTH_RESPONSE,
            sequence_id=message.header.sequence_id + 1,
            timestamp=time.time(),
            client_id=message.header.client_id,
            session_id=message.header.session_id,
            correlation_id=message.header.correlation_id
        )
        
        response_payload = {
            "status": "healthy",
            "uptime": time.time() - self._start_time,
            "model_loaded": self._model_loaded,
            "active_connections": self.stats["active_connections"]
        }
        
        response = WebSocketMessage(header=response_header, payload=response_payload)
        await websocket.send(response.to_json())
        logger.debug("Sent mock health check response")
    
    async def _handle_client_disconnect_msg(self, websocket, message: WebSocketMessage) -> None:
        """Handle client disconnect message"""
        client_id = message.header.client_id
        await self._handle_client_disconnect(client_id)
        
        # Send disconnect acknowledgment
        response_header = MessageHeader(
            message_type=MessageType.DISCONNECT,
            sequence_id=message.header.sequence_id + 1,
            timestamp=time.time(),
            client_id=client_id,
            correlation_id=message.header.correlation_id
        )
        
        response_payload = {"status": "disconnected"}
        response = WebSocketMessage(header=response_header, payload=response_payload)
        await websocket.send(response.to_json())
    
    async def _handle_client_disconnect(self, client_id: str) -> None:
        """Handle client disconnection cleanup"""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Mock client {client_id} disconnected and cleaned up")
    
    async def _cleanup_disconnected_client(self, websocket) -> None:
        """Clean up disconnected client by websocket reference"""
        clients_to_remove = []
        for client_id, client_data in self.clients.items():
            if client_data["websocket"] == websocket:
                clients_to_remove.append(client_id)
        
        for client_id in clients_to_remove:
            await self._handle_client_disconnect(client_id)
    
    async def _handle_unknown(self, websocket, message: WebSocketMessage) -> None:
        """Handle unknown message types"""
        logger.warning(f"Received unknown mock message type: {message.header.message_type}")
        await self._send_error(websocket, "UNKNOWN_MESSAGE_TYPE", 
                             f"Mock server does not support message type: {message.header.message_type.value}",
                             message.header.correlation_id)
    
    async def _send_error(self, websocket, error_code: str, error_message: str, 
                         correlation_id: Optional[str] = None) -> None:
        """Send error message to client using proper protocol"""
        try:
            error_payload = ErrorPayload(
                error_code=error_code,
                error_message=error_message,
                recoverable=True,
                suggested_action="Check message format and retry"
            )
            
            error_header = MessageHeader(
                message_type=MessageType.ERROR,
                sequence_id=0,  # Error messages don't need sequence ordering
                timestamp=time.time(),
                client_id="server",
                correlation_id=correlation_id
            )
            
            error_response = WebSocketMessage(header=error_header, payload=error_payload.to_dict())
            await websocket.send(error_response.to_json())
            
            self.stats["error_count"] += 1
            logger.error(f"Sent mock error: {error_code} - {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to send mock error message: {e}")
    
    def _get_mock_processing_time(self) -> float:
        """Get mock processing time based on model size (legacy method for backward compatibility)"""
        # Legacy behavior for test compatibility - exact ranges from original tests
        model_times = {
            "tiny.en": random.uniform(0.1, 0.3),
            "small.en": random.uniform(0.4, 0.7),
            "base.en": random.uniform(0.9, 1.2),
            "medium.en": random.uniform(1.9, 2.2),
            "large": random.uniform(2.9, 3.2)
        }
        return model_times.get(self.model_size, model_times["tiny.en"])
    
    def _generate_mock_transcription(self) -> str:
        """Generate a mock transcription text (legacy method for backward compatibility)"""
        # Legacy behavior - return original corpus items only
        return random.choice(self.mock_transcriptions)
    
    def _update_processing_time_avg(self, processing_time: float) -> None:
        """Update average processing time"""
        total_transcriptions = self.stats["total_transcriptions"]
        if total_transcriptions <= 0:
            # If no transcriptions recorded yet, set this as the first
            self.stats["avg_processing_time"] = processing_time
        elif total_transcriptions == 1:
            self.stats["avg_processing_time"] = processing_time
        else:
            current_avg = self.stats["avg_processing_time"]
            new_avg = ((current_avg * (total_transcriptions - 1)) + processing_time) / total_transcriptions
            self.stats["avg_processing_time"] = new_avg
    
    def _get_mock_performance_metrics(self) -> PerformanceMetricsPayload:
        """Generate enhanced performance metrics using realistic resource simulation"""
        return PerformanceMetricsPayload(
            latency_ms=self.latency_ms + random.uniform(-10, 10),
            throughput_mbps=random.uniform(5.0, 25.0),
            cpu_usage=min(95.0, 15.0 + self._processing_load * 60.0),  # Realistic CPU usage
            memory_usage=self._get_simulated_memory_usage(),  # Use enhanced memory simulation
            temperature=self._get_simulated_temperature(),  # Use enhanced temperature simulation
            network_quality=random.uniform(0.8, 1.0),
            processing_queue_size=len(self.processing_queue),  # Use actual queue size
            error_count=self.stats["error_count"],
            uptime_seconds=time.time() - self._start_time
        )


async def main():
    """Main entry point for the mock server"""
    parser = argparse.ArgumentParser(description="Pi-Whispr Mock WebSocket Server")
    parser.add_argument("--host", default=WEBSOCKET_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=WEBSOCKET_PORT, help="Server port") 
    parser.add_argument("--latency", type=int, default=100, help="Simulated latency in ms")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Mock model size to simulate")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Create configuration for mock server
    from server.mock_configuration import LatencyProfile, ErrorScenario, ResourceConstraints
    
    # Create configuration objects
    latency_profile = LatencyProfile(
        processing_delay_ms=args.latency,
        network_delay_ms=10,  # Small network delay
        variability_factor=0.2
    )
    
    error_scenario = ErrorScenario(
        failure_rate=0.05,  # 5% failure rate for testing
        timeout_rate=0.02,  # 2% timeout rate
        processing_error_rate=0.01,
        error_types=["NETWORK_ERROR", "TIMEOUT_ERROR"]
    )
    
    resource_constraints = ResourceConstraints(
        max_memory_mb=2048,
        max_cpu_percent=80,
        max_concurrent_connections=10,
        memory_pressure_threshold=0.75,
        cpu_throttle_threshold=0.75
    )
    
    # Create mock server config
    config = MockServerConfig(
        host=args.host,
        port=args.port,
        model_size=args.model,
        latency_profile=latency_profile,
        error_scenario=error_scenario,
        resource_constraints=resource_constraints,
        enable_detailed_logging=args.debug
    )
    
    # Create and start mock server
    mock_server = MockWhisperServer(config=config)
    
    try:
        await mock_server.start()
    except KeyboardInterrupt:
        logger.info("Mock server stopped by user")
    except Exception as e:
        logger.error(f"Mock server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 