#!/usr/bin/env python3
"""
Pi-Whispr Client Test Runner
Simulates the actual Pi client behavior for testing with mock servers
"""

import asyncio
import logging
import sys
import signal
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from client.websocket_client import EnhancedSpeechClient
from client.hotkey_manager import HotkeyManager
from client.text_inserter import TextInserter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global client instance for cleanup
client = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("Received interrupt signal. Cleaning up...")
    if client:
        client.cleanup()
    sys.exit(0)

async def test_with_mock_server(server_url: str, duration: int = 30):
    """Test client functionality with a mock server"""
    global client
    
    logger.info(f"🎤 Testing Pi-Whispr Client with Mock Server")
    logger.info(f"Server: {server_url}")
    logger.info(f"Test Duration: {duration} seconds")
    logger.info("=" * 50)
    
    try:
        # Create client instance
        client = EnhancedSpeechClient(server_url=server_url)
        
        # Connect to server
        logger.info("🔌 Connecting to mock server...")
        await client.connect()
        logger.info("✅ Connected successfully!")
        
        # Test basic functionality without audio hardware
        logger.info("🧪 Testing basic client operations...")
        
        # Test client registration
        logger.info("📋 Testing client registration...")
        await client._register_client()
        
        # Test status request
        logger.info("📊 Testing status request...")
        await client._request_server_status()
        
        # Test ping/pong
        logger.info("🏓 Testing ping/pong...")
        await client._send_ping()
        
        # Simulate some runtime
        logger.info(f"⏱️ Running for {duration} seconds...")
        for i in range(duration):
            await asyncio.sleep(1)
            if not client.is_connected:
                logger.warning("⚠️ Connection lost! Attempting to reconnect...")
                await client.connect()
            
            # Log every 10 seconds
            if (i + 1) % 10 == 0:
                logger.info(f"   Still running... ({i + 1}/{duration}s)")
        
        logger.info("✅ Mock server test completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error during mock server test: {e}")
        raise
    finally:
        if client:
            client.cleanup()

async def test_with_real_audio(server_url: str):
    """Test client with real audio recording (requires audio hardware)"""
    global client
    
    logger.info(f"🎙️ Testing Pi-Whispr Client with Real Audio")
    logger.info(f"Server: {server_url}")
    logger.info("Controls:")
    logger.info("  - SPACE: Hold to record (push-to-talk)")
    logger.info("  - Fn+SPACE: Toggle recording lock")
    logger.info("  - Ctrl+C: Exit")
    logger.info("=" * 50)
    
    try:
        # Create client instance
        client = EnhancedSpeechClient(server_url=server_url)
        
        # Connect to server
        logger.info("🔌 Connecting to server...")
        await client.connect()
        logger.info("✅ Connected to server successfully!")
        
        # Setup hotkeys (platform-specific)
        logger.info("🎮 Setting up hotkeys...")
        try:
            client._setup_hotkeys()
            logger.info("✅ Hotkeys configured")
        except Exception as e:
            logger.warning(f"⚠️ Hotkey setup failed: {e}")
            logger.info("Continuing without hotkeys...")
        
        # Keep the client running
        logger.info("🚀 Client is ready! Try speaking...")
        
        # Main event loop
        while True:
            await asyncio.sleep(1)
            
            # Check connection status
            if not client.is_connected:
                logger.warning("⚠️ Connection lost! Attempting to reconnect...")
                await client.connect()
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise
    finally:
        if client:
            client.cleanup()

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Pi-Whispr Client Test Runner")
    parser.add_argument(
        "--server-url", 
        default="ws://localhost:8765",
        help="WebSocket server URL (default: ws://localhost:8765)"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "audio"],
        default="mock",
        help="Test mode: 'mock' for mock server testing, 'audio' for real audio testing"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test duration in seconds for mock mode (default: 30)"
    )
    parser.add_argument(
        "--pi5",
        action="store_true",
        help="Use Pi5 simulation server (port 8766)"
    )
    parser.add_argument(
        "--stress",
        action="store_true", 
        help="Use stress testing server (port 8767)"
    )
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Adjust server URL based on flags
    server_url = args.server_url
    if args.pi5:
        server_url = "ws://localhost:8766"
        logger.info("🥧 Using Pi5 simulation server")
    elif args.stress:
        server_url = "ws://localhost:8767"
        logger.info("💪 Using stress testing server")
    
    try:
        if args.mode == "mock":
            asyncio.run(test_with_mock_server(server_url, args.duration))
        else:
            asyncio.run(test_with_real_audio(server_url))
    except KeyboardInterrupt:
        logger.info("Exiting...")
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 