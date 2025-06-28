"""
Mock WebSocket Server for Docker Testing

This module provides a lightweight mock implementation of the Pi-Whispr WebSocket server
for development and testing purposes. It simulates the same protocol as the real server
but without requiring faster-whisper or heavy dependencies.

Features:
- Same WebSocket protocol as the real server
- Mock transcription responses with realistic delays
- Client connection management
- Health check endpoint
- Configurable latency simulation
"""

import asyncio
import websockets
import json
import time
import logging
import argparse
from typing import Dict, Any, Optional

# Import shared modules for protocol compatibility
try:
    from shared.protocol import MessageType, MessageBuilder
    from shared.constants import WEBSOCKET_HOST, WEBSOCKET_PORT
except ImportError:
    # Fallback for testing without full shared module
    class MessageType:
        PING = "PING"
        PONG = "PONG"
        CONNECT = "CONNECT"
        DISCONNECT = "DISCONNECT"
        AUDIO_DATA = "AUDIO_DATA"
        TRANSCRIPTION_RESULT = "TRANSCRIPTION_RESULT"
        STATUS_REQUEST = "STATUS_REQUEST"
        STATUS_RESPONSE = "STATUS_RESPONSE"
        ERROR = "ERROR"
    
    WEBSOCKET_HOST = "0.0.0.0"
    WEBSOCKET_PORT = 8765


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockWhisperServer:
    """Mock WebSocket server that simulates the Pi-Whispr protocol"""
    
    def __init__(self, host: str = WEBSOCKET_HOST, port: int = WEBSOCKET_PORT, 
                 latency_ms: int = 100):
        self.host = host
        self.port = port
        self.latency_ms = latency_ms
        
        # Server state
        self._start_time = time.time()
        
        # Client management
        self.clients: Dict[str, Dict[str, Any]] = {}
        
        # Mock statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_transcriptions": 0,
            "uptime_seconds": 0.0
        }
        
        logger.info(f"MockWhisperServer initialized on {host}:{port} with {latency_ms}ms latency")
    
    async def start(self) -> None:
        """Start the mock WebSocket server"""
        logger.info(f"Starting Mock WebSocket server on {self.host}:{self.port}")
        
        # Create a closure that captures self
        async def connection_handler(websocket):
            # websockets.serve expects a handler with (websocket) signature in newer versions
            # Provide a default path since websocket.path may not be available
            path = getattr(websocket, 'path', '/')
            await self._handle_client_connection(websocket, path)
        
        try:
            async with websockets.serve(
                connection_handler,
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
                try:
                    await self._handle_message(websocket, message)
                except Exception as e:
                    logger.error(f"Error processing message from {client_address}: {e}")
                    try:
                        await self._send_error(websocket, "MESSAGE_ERROR", str(e))
                    except:
                        # Connection might be closed, just log and continue
                        logger.error(f"Failed to send error response to {client_address}")
                        break
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Mock client {client_address} disconnected normally")
        except Exception as e:
            logger.error(f"Unexpected error handling mock client {client_address}: {e}")
        finally:
            self.stats["active_connections"] -= 1
            logger.debug(f"Client {client_address} cleanup completed")
    
    async def _handle_message(self, websocket, raw_message: str) -> None:
        """Handle incoming WebSocket message"""
        try:
            message = json.loads(raw_message)
            message_type = message.get("header", {}).get("message_type", "UNKNOWN")
            
            logger.debug(f"Received mock message type: {message_type}")
            
            # Add simulated latency
            if self.latency_ms > 0:
                await asyncio.sleep(self.latency_ms / 1000.0)
            
            # Route message based on type
            await self._route_message(websocket, message)
            
        except json.JSONDecodeError as e:
            await self._send_error(websocket, "INVALID_JSON", f"JSON parsing error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error handling mock message: {e}")
            await self._send_error(websocket, "INTERNAL_ERROR", "Internal server error")
    
    async def _route_message(self, websocket, message: Dict[str, Any]) -> None:
        """Route message to appropriate mock handler"""
        message_type = message.get("header", {}).get("message_type", "UNKNOWN")
        
        handlers = {
            "PING": self._handle_ping,
            "CONNECT": self._handle_connect,
            "DISCONNECT": self._handle_disconnect,
            "AUDIO_DATA": self._handle_audio_data,
            "STATUS_REQUEST": self._handle_status_request,
        }
        
        handler = handlers.get(message_type, self._handle_unknown)
        await handler(websocket, message)
    
    async def _handle_ping(self, websocket, message: Dict[str, Any]) -> None:
        """Handle ping message with pong response"""
        correlation_id = message.get("header", {}).get("correlation_id")
        
        pong_response = {
            "header": {
                "message_type": "PONG",  # Use string instead of enum
                "timestamp": time.time(),
                "sequence_id": message.get("header", {}).get("sequence_id", 0) + 1,
                "correlation_id": correlation_id
            },
            "payload": {
                "server_time": time.time(),
                "uptime": time.time() - self._start_time
            }
        }
        
        await websocket.send(json.dumps(pong_response))
        logger.debug("Sent mock pong response")
    
    async def _handle_connect(self, websocket, message: Dict[str, Any]) -> None:
        """Handle client connection/registration"""
        client_info = message.get("payload", {})
        client_id = client_info.get("client_id", f"mock_client_{len(self.clients)}")
        
        # Register client
        self.clients[client_id] = {
            "websocket": websocket,
            "connected_at": time.time(),
            "client_info": client_info
        }
        
        # Send acknowledgment
        ack_response = {
            "header": {
                "message_type": "CONNECT_ACK",
                "timestamp": time.time(),
                "sequence_id": message.get("header", {}).get("sequence_id", 0) + 1,
                "correlation_id": message.get("header", {}).get("correlation_id")
            },
            "payload": {
                "status": "connected",
                "client_id": client_id,
                "server_capabilities": ["transcription", "audio_processing"]
            }
        }
        
        await websocket.send(json.dumps(ack_response))
        logger.info(f"Mock client {client_id} registered successfully")
    
    async def _handle_disconnect(self, websocket, message: Dict[str, Any]) -> None:
        """Handle client disconnect request"""
        client_id = message.get("payload", {}).get("client_id")
        
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Mock client {client_id} disconnected")
        
        # Send acknowledgment
        disconnect_ack = {
            "header": {
                "message_type": "DISCONNECT_ACK",
                "timestamp": time.time(),
                "sequence_id": message.get("header", {}).get("sequence_id", 0) + 1,
                "correlation_id": message.get("header", {}).get("correlation_id")
            },
            "payload": {
                "status": "disconnected"
            }
        }
        
        await websocket.send(json.dumps(disconnect_ack))
    
    async def _handle_audio_data(self, websocket, message: Dict[str, Any]) -> None:
        """Handle audio data with mock transcription"""
        self.stats["total_transcriptions"] += 1
        
        # Simulate processing time for transcription
        processing_delay = 0.5 + (self.latency_ms / 1000.0)
        await asyncio.sleep(processing_delay)
        
        # Generate mock transcription result
        mock_transcriptions = [
            "Hello, this is a mock transcription.",
            "Testing the WebSocket audio processing.",
            "Mock faster-whisper transcription result.",
            "The quick brown fox jumps over the lazy dog.",
            "This is simulated speech-to-text output."
        ]
        
        mock_text = mock_transcriptions[self.stats["total_transcriptions"] % len(mock_transcriptions)]
        
        transcription_response = {
            "header": {
                "message_type": "TRANSCRIPTION_RESULT",  # Use string instead of enum
                "timestamp": time.time(),
                "sequence_id": message.get("header", {}).get("sequence_id", 0) + 1,
                "correlation_id": message.get("header", {}).get("correlation_id")
            },
            "payload": {
                "text": mock_text,
                "confidence": 0.95,
                "processing_time": processing_delay,
                "language": "en",
                "model": "mock-whisper-tiny"
            }
        }
        
        await websocket.send(json.dumps(transcription_response))
        logger.info(f"Sent mock transcription: '{mock_text}'")
    
    async def _handle_status_request(self, websocket, message: Dict[str, Any]) -> None:
        """Handle server status request"""
        self.stats["uptime_seconds"] = time.time() - self._start_time
        
        status_response = {
            "header": {
                "message_type": "STATUS_RESPONSE",  # Use string instead of enum
                "timestamp": time.time(),
                "sequence_id": message.get("header", {}).get("sequence_id", 0) + 1,
                "correlation_id": message.get("header", {}).get("correlation_id")
            },
            "payload": {
                "status": "running",
                "server_type": "mock",
                "model_loaded": True,
                "model_name": "mock-whisper-tiny",
                "statistics": self.stats,
                "active_clients": len(self.clients)
            }
        }
        
        await websocket.send(json.dumps(status_response))
        logger.debug("Sent mock status response")
    
    async def _handle_unknown(self, websocket, message: Dict[str, Any]) -> None:
        """Handle unknown message types"""
        message_type = message.get("header", {}).get("message_type", "UNKNOWN")
        logger.warning(f"Received unknown mock message type: {message_type}")
        
        await self._send_error(websocket, "UNKNOWN_MESSAGE_TYPE", 
                             f"Mock server does not support message type: {message_type}")
    
    async def _send_error(self, websocket, error_code: str, error_message: str, 
                         correlation_id: Optional[str] = None) -> None:
        """Send error message to client"""
        error_response = {
            "header": {
                "message_type": "ERROR",  # Use string instead of enum
                "timestamp": time.time(),
                "sequence_id": 0,
                "correlation_id": correlation_id
            },
            "payload": {
                "error_code": error_code,
                "error_message": error_message,
                "severity": "error"
            }
        }
        
        try:
            await websocket.send(json.dumps(error_response))
            logger.error(f"Sent mock error: {error_code} - {error_message}")
        except Exception as e:
            logger.error(f"Failed to send mock error message: {e}")


async def main():
    """Main entry point for the mock server"""
    parser = argparse.ArgumentParser(description="Pi-Whispr Mock WebSocket Server")
    parser.add_argument("--host", default=WEBSOCKET_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=WEBSOCKET_PORT, help="Server port") 
    parser.add_argument("--latency", type=int, default=100, help="Simulated latency in ms")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Create and start mock server
    mock_server = MockWhisperServer(
        host=args.host,
        port=args.port,
        latency_ms=args.latency
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