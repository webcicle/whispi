#!/usr/bin/env python3
"""
Comprehensive Pi-Whispr Testing Environment
Orchestrates testing in mock environment before Pi deployment
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
import websockets
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import docker
import signal
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestEnvironment:
    """Manages the complete testing environment for Pi-Whispr"""
    
    def __init__(self):
        self.docker_client = None
        self.running_containers = []
        self.test_results = {
            'connection_tests': {},
            'functionality_tests': {},
            'performance_tests': {},
            'pi_simulation_tests': {}
        }
        
    async def initialize(self):
        """Initialize Docker client and environment"""
        try:
            self.docker_client = docker.from_env()
            logger.info("✅ Docker client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Docker client: {e}")
            return False
        return True
    
    def cleanup(self):
        """Clean up any running containers"""
        logger.info("🧹 Cleaning up test environment...")
        
        # Stop any running containers
        for container_name in self.running_containers:
            try:
                container = self.docker_client.containers.get(container_name)
                if container.status == 'running':
                    logger.info(f"Stopping container: {container_name}")
                    container.stop(timeout=10)
            except Exception as e:
                logger.warning(f"Error stopping container {container_name}: {e}")
        
        self.running_containers.clear()
    
    async def start_mock_server(self, server_type: str = "mock-server") -> bool:
        """Start a specific mock server configuration"""
        logger.info(f"🚀 Starting {server_type}...")
        
        try:
            # Use docker-compose to start the specific service
            cmd = ["docker-compose", "up", "-d", server_type]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"Failed to start {server_type}: {result.stderr}")
                return False
            
            self.running_containers.append(server_type)
            
            # Wait for server to be ready
            port = self._get_server_port(server_type)
            if await self._wait_for_server(port):
                logger.info(f"✅ {server_type} is ready on port {port}")
                return True
            else:
                logger.error(f"❌ {server_type} failed to start properly")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout starting {server_type}")
            return False
        except Exception as e:
            logger.error(f"❌ Error starting {server_type}: {e}")
            return False
    
    def _get_server_port(self, server_type: str) -> int:
        """Get the port for a specific server type"""
        port_map = {
            "mock-server": 8765,
            "mock-pi5": 8766,
            "mock-stress": 8767
        }
        return port_map.get(server_type, 8765)
    
    async def _wait_for_server(self, port: int, timeout: int = 30) -> bool:
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with websockets.connect(f"ws://localhost:{port}") as websocket:
                    # Send a ping to verify it's responding
                    ping_msg = {"type": "ping", "timestamp": time.time()}
                    await websocket.send(json.dumps(ping_msg))
                    
                    # Wait for response
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    logger.debug(f"Server responded: {response}")
                    return True
                    
            except Exception as e:
                logger.debug(f"Server not ready yet: {e}")
                await asyncio.sleep(2)
        
        return False
    
    async def test_basic_connection(self, port: int = 8765) -> bool:
        """Test basic WebSocket connection"""
        logger.info(f"🔌 Testing basic connection on port {port}")
        
        try:
            async with websockets.connect(f"ws://localhost:{port}") as websocket:
                # Send ping
                ping_msg = {
                    "type": "ping",
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(ping_msg))
                
                # Wait for pong
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                if response_data.get("type") == "pong":
                    logger.info("✅ Basic connection test passed")
                    self.test_results['connection_tests']['basic'] = True
                    return True
                else:
                    logger.warning(f"Unexpected response: {response_data}")
                    self.test_results['connection_tests']['basic'] = False
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Basic connection test failed: {e}")
            self.test_results['connection_tests']['basic'] = False
            return False
    
    async def test_client_registration(self, port: int = 8765) -> bool:
        """Test client registration functionality"""
        logger.info(f"🔐 Testing client registration on port {port}")
        
        try:
            async with websockets.connect(f"ws://localhost:{port}") as websocket:
                # Send client registration
                registration_msg = {
                    "type": "client_registration",
                    "client_id": "test-client-001",
                    "client_info": {
                        "platform": "macOS",
                        "version": "1.0.0",
                        "capabilities": ["audio_streaming", "hotkeys"]
                    }
                }
                await websocket.send(json.dumps(registration_msg))
                
                # Wait for confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                if response_data.get("type") == "registration_confirmed":
                    logger.info("✅ Client registration test passed")
                    self.test_results['connection_tests']['registration'] = True
                    return True
                else:
                    logger.warning(f"Registration failed: {response_data}")
                    self.test_results['connection_tests']['registration'] = False
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Client registration test failed: {e}")
            self.test_results['connection_tests']['registration'] = False
            return False
    
    async def test_audio_streaming(self, port: int = 8765) -> bool:
        """Test audio streaming functionality"""
        logger.info(f"🎵 Testing audio streaming on port {port}")
        
        try:
            async with websockets.connect(f"ws://localhost:{port}") as websocket:
                # Register client first
                registration_msg = {
                    "type": "client_registration",
                    "client_id": "test-audio-client",
                    "client_info": {
                        "platform": "macOS",
                        "version": "1.0.0"
                    }
                }
                await websocket.send(json.dumps(registration_msg))
                await websocket.recv()  # Consume registration response
                
                # Send audio start
                audio_start_msg = {
                    "type": "audio_start",
                    "client_id": "test-audio-client",
                    "audio_config": {
                        "sample_rate": 16000,
                        "channels": 1,
                        "format": "pcm"
                    }
                }
                await websocket.send(json.dumps(audio_start_msg))
                
                # Send mock audio data
                mock_audio_data = b"mock_audio_bytes_here" * 100  # Simulate audio
                audio_data_msg = {
                    "type": "audio_data",
                    "client_id": "test-audio-client",
                    "audio_data": mock_audio_data.hex(),
                    "sequence_number": 1,
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(audio_data_msg))
                
                # Send audio end
                audio_end_msg = {
                    "type": "audio_end",
                    "client_id": "test-audio-client"
                }
                await websocket.send(json.dumps(audio_end_msg))
                
                # Wait for transcription result
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                response_data = json.loads(response)
                
                if response_data.get("type") == "transcription_result":
                    logger.info("✅ Audio streaming test passed")
                    logger.info(f"Mock transcription: {response_data.get('text', 'N/A')}")
                    self.test_results['functionality_tests']['audio_streaming'] = True
                    return True
                else:
                    logger.warning(f"No transcription received: {response_data}")
                    self.test_results['functionality_tests']['audio_streaming'] = False
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Audio streaming test failed: {e}")
            self.test_results['functionality_tests']['audio_streaming'] = False
            return False
    
    async def test_pi_simulation(self) -> bool:
        """Test Pi5 simulation environment"""
        logger.info("🥧 Testing Pi5 simulation environment")
        
        # Start Pi5 simulation server
        if not await self.start_mock_server("mock-pi5"):
            return False
        
        try:
            # Test with Pi5 simulation specific checks
            pi5_port = 8766
            
            # Basic connection test
            connection_ok = await self.test_basic_connection(pi5_port)
            
            # Registration test
            registration_ok = await self.test_client_registration(pi5_port)
            
            # Audio streaming with Pi5 characteristics
            audio_ok = await self.test_audio_streaming(pi5_port)
            
            # Test Pi5 specific metrics
            performance_ok = await self.test_pi5_performance_metrics(pi5_port)
            
            all_passed = all([connection_ok, registration_ok, audio_ok, performance_ok])
            
            if all_passed:
                logger.info("✅ Pi5 simulation tests passed")
                self.test_results['pi_simulation_tests']['overall'] = True
            else:
                logger.warning("⚠️ Some Pi5 simulation tests failed")
                self.test_results['pi_simulation_tests']['overall'] = False
                
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Pi5 simulation test failed: {e}")
            self.test_results['pi_simulation_tests']['overall'] = False
            return False
    
    async def test_pi5_performance_metrics(self, port: int) -> bool:
        """Test Pi5 specific performance metrics"""
        logger.info("📊 Testing Pi5 performance metrics")
        
        try:
            async with websockets.connect(f"ws://localhost:{port}") as websocket:
                # Register client
                registration_msg = {
                    "type": "client_registration",
                    "client_id": "pi5-perf-test",
                    "client_info": {"platform": "Pi5Test"}
                }
                await websocket.send(json.dumps(registration_msg))
                await websocket.recv()  # Consume response
                
                # Request performance metrics
                perf_request = {
                    "type": "performance_request",
                    "client_id": "pi5-perf-test"
                }
                await websocket.send(json.dumps(perf_request))
                
                # Get performance response
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                if response_data.get("type") == "performance_metrics":
                    metrics = response_data.get("metrics", {})
                    logger.info(f"Pi5 Metrics - Memory: {metrics.get('memory_usage_mb', 'N/A')}MB, "
                              f"CPU: {metrics.get('cpu_usage_percent', 'N/A')}%, "
                              f"Temp: {metrics.get('temperature_celsius', 'N/A')}°C")
                    
                    self.test_results['pi_simulation_tests']['performance'] = True
                    return True
                else:
                    logger.warning(f"No performance metrics received: {response_data}")
                    self.test_results['pi_simulation_tests']['performance'] = False
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Pi5 performance metrics test failed: {e}")
            self.test_results['pi_simulation_tests']['performance'] = False
            return False
    
    async def run_full_test_suite(self) -> bool:
        """Run the complete test suite"""
        logger.info("🧪 Starting full Pi-Whispr test suite")
        logger.info("=" * 60)
        
        try:
            # Initialize
            if not await self.initialize():
                return False
            
            # Start basic mock server
            if not await self.start_mock_server("mock-server"):
                return False
            
            # Run basic tests
            logger.info("\n📋 Running basic functionality tests...")
            basic_tests = [
                await self.test_basic_connection(),
                await self.test_client_registration(),
                await self.test_audio_streaming()
            ]
            
            basic_passed = all(basic_tests)
            logger.info(f"Basic tests: {'✅ PASSED' if basic_passed else '❌ FAILED'}")
            
            # Run Pi5 simulation tests
            logger.info("\n🥧 Running Pi5 simulation tests...")
            pi5_passed = await self.test_pi_simulation()
            logger.info(f"Pi5 simulation: {'✅ PASSED' if pi5_passed else '❌ FAILED'}")
            
            # Generate test report
            self.generate_test_report()
            
            overall_success = basic_passed and pi5_passed
            
            if overall_success:
                logger.info("\n🎉 ALL TESTS PASSED! Ready for Pi deployment.")
            else:
                logger.error("\n❌ Some tests failed. Check issues before Pi deployment.")
            
            return overall_success
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            return False
        finally:
            self.cleanup()
    
    def generate_test_report(self):
        """Generate a comprehensive test report"""
        logger.info("\n📊 TEST REPORT")
        logger.info("=" * 40)
        
        # Connection tests
        logger.info("🔌 Connection Tests:")
        for test_name, result in self.test_results['connection_tests'].items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        # Functionality tests
        logger.info("⚙️ Functionality Tests:")
        for test_name, result in self.test_results['functionality_tests'].items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        # Pi simulation tests
        logger.info("🥧 Pi5 Simulation Tests:")
        for test_name, result in self.test_results['pi_simulation_tests'].items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        # Save report to file
        report_path = Path("test_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"\n📄 Detailed report saved to: {report_path}")

async def main():
    """Main test runner"""
    test_env = TestEnvironment()
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        logger.info("Received interrupt signal. Cleaning up...")
        test_env.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        success = await test_env.run_full_test_suite()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 1
    finally:
        test_env.cleanup()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 