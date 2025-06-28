"""
Mock WebSocket Server for Docker Testing

This module provides a comprehensive mock implementation of the Pi-Whispr WebSocket server
for development and testing purposes. It implements the complete protocol from task 1
with the same message handling as the real server but without requiring faster-whisper.

Features:
- Complete WebSocket protocol implementation matching the real server
- All message types: connection, audio, transcription, status, error, ping/pong, client management, performance tracking
- Mock transcription responses with realistic delays based on model selection
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
from typing import Dict, Any, Optional, Set
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockWhisperServer:
    """Mock WebSocket server that fully implements the Pi-Whispr protocol"""
    
    def __init__(self, host: str = WEBSOCKET_HOST, port: int = WEBSOCKET_PORT, 
                 latency_ms: int = 100, model_size: str = DEFAULT_MODEL):
        self.host = host
        self.port = port
        self.latency_ms = latency_ms
        self.model_size = model_size
        
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
        
        logger.info(f"MockWhisperServer initialized on {host}:{port} with {latency_ms}ms latency, model: {model_size}")
    
    async def start(self) -> None:
        """Start the mock WebSocket server"""
        logger.info(f"Starting Mock WebSocket server on {self.host}:{self.port}")
        
        try:
            async with websockets.serve(
                self._handle_client_connection,
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
    
    async def _handle_client_connection(self, websocket, path: str) -> None:
        """Handle new client WebSocket connection"""
        client_address = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New mock client connection from {client_address}, path: {path}")
        
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
            
            # Add simulated latency for all message processing
            if self.latency_ms > 0:
                await asyncio.sleep(self.latency_ms / 1000.0)
            
            # Route message based on type
            await self._route_message(websocket, message)
            
        except ValueError as e:
            # This catches structure validation errors from WebSocketMessage.from_json
            await self._send_error(websocket, "INVALID_MESSAGE", f"Invalid message format: {e}")
        except Exception as e:
            logger.error(f"Unexpected error handling mock message: {e}")
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
        """Handle audio data with mock transcription processing"""
        try:
            client_id = message.header.client_id
            
            # Verify client is registered
            if client_id not in self.clients:
                await self._send_error(websocket, "CLIENT_NOT_REGISTERED", 
                                     "Client must register before sending audio data")
                return
            
            # Update client status
            self.clients[client_id]["status"] = ClientStatus.PROCESSING
            self.clients[client_id]["last_seen"] = time.time()
            
            # Extract audio data info
            payload = message.payload
            chunk_index = payload.get("chunk_index", 0)
            is_final = payload.get("is_final", False)
            
            # Only process final chunks for mock transcription
            if is_final:
                self.stats["total_transcriptions"] += 1
                
                # Simulate processing time based on model size
                processing_delay = self._get_mock_processing_time()
                await asyncio.sleep(processing_delay)
                
                # Generate mock transcription
                mock_text = self._generate_mock_transcription()
                
                # Create transcription result
                result_payload = TranscriptionResultPayload(
                    text=mock_text,
                    confidence=random.uniform(0.85, 0.98),
                    processing_time=processing_delay,
                    model_used=f"mock-whisper-{self.model_size}",
                    language="en",
                    audio_duration=random.uniform(1.0, 3.0),
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
                
                # Update processing time average
                self._update_processing_time_avg(processing_delay)
                
                logger.info(f"Mock transcription sent for client {client_id}: '{mock_text}'")
            
            # Update client status back to connected
            self.clients[client_id]["status"] = ClientStatus.CONNECTED
            
        except Exception as e:
            logger.error(f"Error processing mock audio data: {e}")
            await self._send_error(websocket, "AUDIO_PROCESSING_ERROR", str(e), message.header.correlation_id)
    
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
        """Get mock processing time based on model size"""
        base_times = {
            "tiny.en": 0.2,
            "small.en": 0.5,
            "base.en": 1.0,
            "medium.en": 2.0,
            "large": 3.0
        }
        base_time = base_times.get(self.model_size, 0.5)
        # Add some random variation but ensure we stay within bounds
        variation = random.uniform(-0.05, 0.1)  # Reduced variation range
        return max(0.1, base_time + variation)  # Ensure minimum 0.1 seconds
    
    def _generate_mock_transcription(self) -> str:
        """Generate a mock transcription text"""
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
        """Generate mock performance metrics"""
        return PerformanceMetricsPayload(
            latency_ms=self.latency_ms + random.uniform(-10, 10),
            throughput_mbps=random.uniform(5.0, 25.0),
            cpu_usage=random.uniform(10.0, 60.0),
            memory_usage=random.uniform(30.0, 70.0),
            temperature=random.uniform(35.0, 55.0),  # Mock Pi temperature
            network_quality=random.uniform(0.8, 1.0),
            processing_queue_size=random.randint(0, 5),
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
    
    # Create and start mock server
    mock_server = MockWhisperServer(
        host=args.host,
        port=args.port,
        latency_ms=args.latency,
        model_size=args.model
    )
    
    try:
        await mock_server.start()
    except KeyboardInterrupt:
        logger.info("Mock server stopped by user")
    except Exception as e:
        logger.error(f"Mock server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 