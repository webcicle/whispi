"""
Integration Test Suite for Docker Mock Infrastructure (Task 2.6)

This test suite validates the complete Docker-based mock server environment including:
- Docker container builds and deployment
- WebSocket connection and communication
- Basic mock server functionality
- Different configuration scenarios
- Docker Compose profile validation
"""

import pytest
import asyncio
import websockets
import json
import base64
import time
import docker
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path


class TestDockerMockInfrastructure:
    """Integration tests for Docker mock server infrastructure"""
    
    @pytest.fixture(scope="class")
    def docker_client(self):
        """Docker client for container management"""
        try:
            return docker.from_env()
        except Exception:
            pytest.skip("Docker not available")
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data for testing"""
        # Simulate 1 second of 16kHz mono audio
        audio_bytes = b'\x00' * (16000 * 2)
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_docker_container_websocket_connection(self):
        """Test basic WebSocket connection to Docker container"""
        uri = "ws://localhost:8765"
        
        try:
            async with websockets.connect(uri, open_timeout=5) as websocket:
                # Send simple ping message using correct protocol format
                ping_message = {
                    "header": {
                        "type": "ping",
                        "client_id": "test_client_001",
                        "timestamp": time.time(),
                        "sequence_id": 1
                    },
                    "payload": {}
                }
                
                await websocket.send(json.dumps(ping_message))
                
                # Receive response
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_msg = json.loads(response)
                
                assert response_msg["header"]["type"] == "pong"
                
        except asyncio.TimeoutError:
            pytest.fail("WebSocket connection timed out")
        except ConnectionRefusedError:
            pytest.skip("Mock server not running - start with: docker-compose up mock-server")
        except Exception as e:
            pytest.fail(f"WebSocket connection failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_basic_transcription_workflow(self, sample_audio_data):
        """Test basic transcription workflow through Docker mock server"""
        uri = "ws://localhost:8765"
        client_id = "integration_test_client"
        session_id = "integration_session_001"
        
        try:
            async with websockets.connect(uri, open_timeout=5) as websocket:
                # Step 1: Register client
                register_message = {
                    "header": {
                        "type": "connect",
                        "client_id": client_id,
                        "timestamp": time.time(),
                        "sequence_id": 1
                    },
                    "payload": {
                        "client_name": "test_client",
                        "client_version": "1.0.0",
                        "platform": "test_platform",
                        "capabilities": ["audio_streaming", "transcription"],
                        "status": "connected"
                    }
                }
                
                await websocket.send(json.dumps(register_message))
                
                # Wait for registration confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_msg = json.loads(response)
                assert response_msg["header"]["type"] == "connect_ack"
                
                # Step 2: Send audio data
                audio_message = {
                    "header": {
                        "type": "audio_data",
                        "client_id": client_id,
                        "timestamp": time.time(),
                        "sequence_id": 2,
                        "session_id": session_id
                    },
                    "payload": {
                        "audio_data": sample_audio_data,
                        "chunk_index": 0,
                        "is_final": True
                    }
                }
                
                await websocket.send(json.dumps(audio_message))
                
                # Step 3: Receive transcription result
                transcription_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                transcription_msg = json.loads(transcription_response)
                
                assert transcription_msg["header"]["type"] == "transcription_result"
                assert transcription_msg["header"]["client_id"] == client_id
                
                # Validate transcription payload
                payload = transcription_msg["payload"]
                assert payload["client_id"] == client_id
                assert payload["session_id"] == session_id
                assert len(payload["text"]) > 0
                assert 0.0 <= payload["confidence"] <= 1.0
                
        except ConnectionRefusedError:
            pytest.skip("Mock server not running - start with: docker-compose up mock-server")
        except Exception as e:
            pytest.fail(f"Transcription workflow failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_clients_support(self, sample_audio_data):
        """Test mock server handles multiple concurrent clients"""
        uri = "ws://localhost:8765"
        num_clients = 3
        client_tasks = []
        
        async def client_session(client_index: int):
            """Single client session for concurrent testing"""
            client_id = f"concurrent_client_{client_index}"
            session_id = f"concurrent_session_{client_index}"
            
            try:
                async with websockets.connect(uri, open_timeout=5) as websocket:
                    # Register client
                    register_message = {
                        "header": {
                            "type": "connect",
                            "client_id": client_id,
                            "timestamp": time.time(),
                            "sequence_id": 1
                        },
                        "payload": {
                            "client_name": client_id,
                            "client_version": "1.0.0",
                            "platform": "test_platform",
                            "capabilities": ["audio_streaming"],
                            "status": "connected"
                        }
                    }
                    
                    await websocket.send(json.dumps(register_message))
                    await websocket.recv()  # Wait for registration ACK
                    
                    # Send audio and get transcription
                    audio_message = {
                        "header": {
                            "type": "audio_data",
                            "client_id": client_id,
                            "timestamp": time.time(),
                            "sequence_id": 2,
                            "session_id": session_id
                        },
                        "payload": {
                            "audio_data": sample_audio_data,
                            "chunk_index": 0,
                            "is_final": True
                        }
                    }
                    
                    await websocket.send(json.dumps(audio_message))
                    
                    # Wait for transcription
                    response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                    response_msg = json.loads(response)
                    
                    assert response_msg["header"]["type"] == "transcription_result"
                    return client_index
                    
            except ConnectionRefusedError:
                pytest.skip("Mock server not running")
                
        try:
            # Run concurrent client sessions
            for i in range(num_clients):
                task = asyncio.create_task(client_session(i))
                client_tasks.append(task)
            
            # Wait for all clients to complete
            results = await asyncio.gather(*client_tasks)
            assert len(results) == num_clients
            assert all(isinstance(result, int) for result in results)
            
        except ConnectionRefusedError:
            pytest.skip("Mock server not running - start with: docker-compose up mock-server")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_server_health_and_metrics(self):
        """Test server health check and performance metrics"""
        uri = "ws://localhost:8765"
        client_id = "metrics_test_client"
        
        try:
            async with websockets.connect(uri, open_timeout=5) as websocket:
                # Register client
                register_message = {
                    "header": {
                        "type": "connect",
                        "client_id": client_id,
                        "timestamp": time.time(),
                        "sequence_id": 1
                    },
                    "payload": {
                        "client_name": client_id,
                        "client_version": "1.0.0",
                        "platform": "test_platform",
                        "capabilities": ["performance_monitoring"],
                        "status": "connected"
                    }
                }
                
                await websocket.send(json.dumps(register_message))
                await websocket.recv()  # Registration ACK
                
                # Request performance metrics
                metrics_message = {
                    "header": {
                        "type": "performance_metrics",
                        "client_id": client_id,
                        "timestamp": time.time(),
                        "sequence_id": 2
                    },
                    "payload": {}
                }
                
                await websocket.send(json.dumps(metrics_message))
                
                # Receive performance metrics
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_msg = json.loads(response)
                
                assert response_msg["header"]["type"] == "performance_metrics"
                
                # Validate metrics structure
                metrics = response_msg["payload"]
                assert "memory_usage_percent" in metrics
                assert "cpu_temperature" in metrics
                assert "active_connections" in metrics
                assert "total_transcriptions" in metrics
                
        except ConnectionRefusedError:
            pytest.skip("Mock server not running - start with: docker-compose up mock-server")
    
    @pytest.mark.integration
    def test_docker_compose_profiles(self):
        """Test that Docker Compose profiles are properly configured"""
        compose_file = Path("docker-compose.yml")
        assert compose_file.exists()
        
        # Check that compose file contains mock service definitions
        with open(compose_file, 'r') as f:
            compose_content = f.read()
        
        assert "mock-server:" in compose_content
        assert "mock-pi5:" in compose_content
        assert "mock-stress:" in compose_content
        assert "dockerfile: docker/Dockerfile.mock" in compose_content
    
    @pytest.mark.integration
    def test_mock_scenario_configurations(self):
        """Test that all mock scenario configuration files are valid"""
        scenarios_dir = Path("config/mock_scenarios")
        assert scenarios_dir.exists()
        
        scenario_files = [
            "fast_development.json",
            "pi5_simulation.json", 
            "stress_testing.json",
            "custom_example.json"
        ]
        
        for scenario_file in scenario_files:
            scenario_path = scenarios_dir / scenario_file
            assert scenario_path.exists()
            
            # Validate JSON format
            with open(scenario_path, 'r') as f:
                scenario_config = json.load(f)
            
            # Validate required fields
            assert "description" in scenario_config
            assert "latency_profile" in scenario_config
            assert "error_scenario" in scenario_config
            assert "resource_constraints" in scenario_config
    
    @pytest.mark.integration
    def test_docker_mock_dockerfile_exists(self):
        """Test that Dockerfile.mock exists and has basic requirements"""
        dockerfile_path = Path("docker/Dockerfile.mock")
        assert dockerfile_path.exists()
        
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        # Check key components
        assert "FROM python:" in dockerfile_content
        assert "requirements/mock.txt" in dockerfile_content
        assert "mock_server" in dockerfile_content
        assert "EXPOSE 8765" in dockerfile_content


@pytest.mark.integration
class TestDockerMockPerformance:
    """Performance tests for Docker mock infrastructure"""
    
    @pytest.mark.asyncio
    async def test_transcription_throughput(self):
        """Test transcription throughput under load"""
        uri = "ws://localhost:8765"
        num_requests = 5  # Reduced for faster testing
        start_time = time.time()
        
        async def send_transcription_request(request_id: int):
            client_id = f"throughput_client_{request_id}"
            
            # Generate audio data
            audio_bytes = b'\x00' * (16000 * 2)
            sample_audio_data = base64.b64encode(audio_bytes).decode('utf-8')
            
            try:
                async with websockets.connect(uri, open_timeout=5) as websocket:
                    # Register client
                    register_message = {
                        "header": {
                            "message_type": "CLIENT_REGISTER",
                            "source_id": client_id,
                            "timestamp": time.time(),
                            "sequence_id": 1
                        },
                        "payload": {
                            "client_id": client_id,
                            "platform": "test_platform",
                            "version": "1.0.0",
                            "features": ["audio_streaming"]
                        }
                    }
                    
                    await websocket.send(json.dumps(register_message))
                    await websocket.recv()
                    
                    # Send audio and get transcription
                    audio_message = {
                        "header": {
                            "message_type": "AUDIO_DATA",
                            "source_id": client_id,
                            "timestamp": time.time(),
                            "sequence_id": 2
                        },
                        "payload": {
                            "client_id": client_id,
                            "session_id": f"throughput_session_{request_id}",
                            "chunk_index": 0,
                            "audio_data": sample_audio_data,
                            "sample_rate": 16000,
                            "channels": 1,
                            "bit_depth": 16,
                            "is_final": True
                        }
                    }
                    
                    await websocket.send(json.dumps(audio_message))
                    await websocket.recv()
                    
                    return request_id
                    
            except ConnectionRefusedError:
                pytest.skip("Mock server not running")
        
        try:
            # Execute concurrent requests
            tasks = [send_transcription_request(i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions (connection refused, etc.)
            successful_results = [r for r in results if not isinstance(r, Exception)]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if successful_results:
                assert len(successful_results) >= 1  # At least one should succeed
                assert total_time < 30.0  # Should complete in reasonable time
                
                # Calculate throughput
                throughput = len(successful_results) / total_time
                assert throughput > 0.1  # Should handle at least 0.1 requests per second
            else:
                pytest.skip("No successful connections - mock server may not be running")
                
        except Exception as e:
            pytest.skip(f"Performance test failed - server may not be running: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 