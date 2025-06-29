#!/usr/bin/env python3
"""
Pi-Whispr Connection Test
Quick connection validation for mock servers
"""

import asyncio
import websockets
import json
import logging
import argparse
import sys
import time
from typing import Dict, Any, Optional

# Import shared protocol for proper message formatting
from shared.protocol import (
    MessageBuilder, MessageType, Priority, ClientStatus,
    MessageHeader, WebSocketMessage, ClientInfoPayload
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_connection(server_url: str) -> bool:
    """Test basic WebSocket connection"""
    logger.info(f"🔌 Testing basic connection to {server_url}")
    
    try:
        # Create message builder for proper protocol formatting
        message_builder = MessageBuilder(client_id="connection-test-client")
        
        async with websockets.connect(server_url) as websocket:
            logger.info("✅ Connection established!")
            
            # Send ping using proper protocol format
            ping_message = message_builder.ping_message()
            
            await websocket.send(ping_message.to_json())
            logger.info("📤 Sent ping message")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                # Check for proper protocol response
                if "header" in response_data and response_data["header"].get("type") == "pong":
                    logger.info("✅ Received pong response - Server is responsive!")
                    return True
                else:
                    logger.info(f"✅ Server responded: {response_data}")
                    return True
                    
            except asyncio.TimeoutError:
                logger.warning("⏰ No response within 5 seconds")
                return False
            except json.JSONDecodeError:
                logger.info("✅ Server responded (non-JSON)")
                return True
            
    except ConnectionRefusedError:
        logger.error("❌ Connection refused. Server not running?")
        return False
    except Exception as e:
        logger.error(f"❌ Connection error: {e}")
        return False

async def test_client_registration(server_url: str) -> bool:
    """Test client registration functionality"""
    logger.info(f"🔐 Testing client registration on {server_url}")
    
    try:
        # Create message builder
        message_builder = MessageBuilder(client_id="registration-test-client")
        
        async with websockets.connect(server_url) as websocket:
            # Create client info payload
            client_info = ClientInfoPayload(
                client_name="Pi-Whispr Test Client",
                client_version="1.0.0",
                platform="Test",
                capabilities=["testing", "connection_test"],
                status=ClientStatus.CONNECTED
            )
            
            # Send registration using proper protocol format
            registration_message = message_builder.connect_message(client_info)
            
            await websocket.send(registration_message.to_json())
            logger.info("📤 Sent registration request")
            
            # Wait for confirmation
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            # Check for proper protocol response
            if ("header" in response_data and 
                response_data["header"].get("type") in ["connect_ack", "client_register"]):
                logger.info("✅ Client registration successful!")
                return True
            else:
                logger.warning(f"⚠️ Unexpected response: {response_data}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Registration test failed: {e}")
        return False

async def test_server_status(server_url: str) -> bool:
    """Test server status request"""
    logger.info(f"📊 Testing server status on {server_url}")
    
    try:
        # Create message builder
        message_builder = MessageBuilder(client_id="status-test-client")
        
        async with websockets.connect(server_url) as websocket:
            # Register first if needed
            client_info = ClientInfoPayload(
                client_name="Status Test Client",
                client_version="1.0.0",
                platform="Test",
                capabilities=["testing"],
                status=ClientStatus.CONNECTED
            )
            
            registration_message = message_builder.connect_message(client_info)
            await websocket.send(registration_message.to_json())
            
            # Consume registration response
            try:
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                pass  # May not get immediate response
            
            # Request status using proper protocol format
            status_header = message_builder._create_header(MessageType.STATUS_REQUEST)
            status_message = WebSocketMessage(header=status_header, payload={})
            
            await websocket.send(status_message.to_json())
            logger.info("📤 Sent status request")
            
            # Wait for status response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            if ("header" in response_data and 
                response_data["header"].get("type") == "status_response"):
                status = response_data.get("payload", {})
                logger.info(f"✅ Server status received:")
                logger.info(f"   Model: {status.get('model', 'Unknown')}")
                logger.info(f"   Connections: {status.get('active_connections', 'Unknown')}")
                logger.info(f"   Uptime: {status.get('uptime_seconds', 'Unknown')}s")
                return True
            else:
                logger.warning(f"⚠️ Unexpected response: {response_data}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Status test failed: {e}")
        return False

async def run_connection_tests(server_url: str) -> Dict[str, bool]:
    """Run all connection tests"""
    logger.info(f"🧪 Running connection tests for {server_url}")
    logger.info("=" * 50)
    
    results = {}
    
    # Test basic connection
    results['basic_connection'] = await test_basic_connection(server_url)
    
    # Test client registration
    results['client_registration'] = await test_client_registration(server_url)
    
    # Test server status
    results['server_status'] = await test_server_status(server_url)
    
    return results

def print_test_results(results: Dict[str, bool], server_name: str):
    """Print formatted test results"""
    logger.info(f"\n📊 {server_name} Test Results:")
    logger.info("=" * 30)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nSummary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info(f"🎉 All {server_name} tests passed!")
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} {server_name} tests failed!")

async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Pi-Whispr Connection Test")
    parser.add_argument(
        "--server-url",
        default="ws://localhost:8765",
        help="WebSocket server URL (default: ws://localhost:8765)"
    )
    parser.add_argument(
        "--all-servers",
        action="store_true",
        help="Test all mock server types"
    )
    parser.add_argument(
        "--pi5",
        action="store_true",
        help="Test mock-pi5 server (port 8766)"
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Test mock-stress server (port 8767)"
    )
    
    args = parser.parse_args()
    
    logger.info("🧪 Pi-Whispr Connection Test Suite")
    logger.info("=" * 40)
    
    overall_success = True
    
    if args.all_servers:
        # Test all server types
        servers = [
            ("Mock Server", "ws://localhost:8765"),
            ("Mock Pi5", "ws://localhost:8766"),
            ("Mock Stress", "ws://localhost:8767")
        ]
        
        for server_name, server_url in servers:
            logger.info(f"\n🔄 Testing {server_name}...")
            results = await run_connection_tests(server_url)
            print_test_results(results, server_name)
            
            if not all(results.values()):
                overall_success = False
                
    else:
        # Test single server
        server_url = args.server_url
        server_name = "Default Server"
        
        if args.pi5:
            server_url = "ws://localhost:8766"
            server_name = "Mock Pi5"
        elif args.stress:
            server_url = "ws://localhost:8767"
            server_name = "Mock Stress"
        
        results = await run_connection_tests(server_url)
        print_test_results(results, server_name)
        overall_success = all(results.values())
    
    if overall_success:
        logger.info("\n🎉 All connection tests passed! Servers are ready.")
        return 0
    else:
        logger.error("\n❌ Some connection tests failed. Check server status.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main()) 