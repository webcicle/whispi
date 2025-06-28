"""
Test suite for Docker Mock Infrastructure (Task 2.1)

This module tests the Docker container infrastructure for the mock server,
validating that it builds correctly, runs the WebSocket server, and handles
client connections as specified in the task requirements.
"""

import asyncio
import json
import time
import subprocess
import tempfile
import os
import pytest
import docker
import websockets
from pathlib import Path


class TestDockerMockInfrastructure:
    """Test Docker container infrastructure for mock server"""
    
    @pytest.fixture(scope="class")
    def docker_client(self):
        """Provide Docker client for tests"""
        return docker.from_env()
    
    @pytest.fixture(scope="class") 
    def project_root(self):
        """Get project root directory"""
        return Path(__file__).parent.parent
    
    def test_dockerfile_exists(self, project_root):
        """Test that the mock server Dockerfile exists"""
        dockerfile_path = project_root / "docker" / "Dockerfile.mock"
        assert dockerfile_path.exists(), "Dockerfile.mock should exist"
        
        # Verify it has the required content
        content = dockerfile_path.read_text()
        assert "FROM python:3.11-slim" in content, "Should use Python 3.11+ base image"
        assert "COPY requirements/mock.txt" in content, "Should use mock requirements"
        assert "CMD" in content and "mock_server" in content, "Should run mock server"
    
    def test_mock_requirements_exists(self, project_root):
        """Test that mock requirements file exists with appropriate dependencies"""
        req_path = project_root / "requirements" / "mock.txt"
        assert req_path.exists(), "mock.txt requirements should exist"
        
        content = req_path.read_text()
        assert "websockets" in content, "Should include websockets"
        assert "pytest" in content, "Should include pytest"
        assert "faster-whisper" not in content, "Should NOT include faster-whisper for lightweight image"
    
    def test_docker_image_builds_successfully(self, docker_client, project_root):
        """Test that the Docker image builds without errors"""
        try:
            # Build the image
            image, build_logs = docker_client.images.build(
                path=str(project_root),
                dockerfile="docker/Dockerfile.mock",
                tag="pi-whispr-mock:test",
                rm=True
            )
            
            # Verify image was created
            assert image is not None, "Docker image should be created"
            assert any("pi-whispr-mock:test" in tag for tag in image.tags), "Image should be tagged correctly"
            
            # Check image has Python 3.11+
            result = docker_client.containers.run(
                image=image,
                command=["python", "--version"],
                remove=True,
                stderr=True,
                stdout=True
            )
            output = result.decode('utf-8') if isinstance(result, bytes) else str(result)
            assert "Python 3.11" in output or "Python 3.12" in output or "Python 3.13" in output, "Should use Python 3.11+"
            
        except docker.errors.BuildError as e:
            pytest.fail(f"Docker build failed: {e}")
    
    def test_required_dependencies_installed(self, docker_client, project_root):
        """Test that required dependencies are installed in the container"""
        try:
            # Build image first
            image, _ = docker_client.images.build(
                path=str(project_root),
                dockerfile="docker/Dockerfile.mock", 
                tag="pi-whispr-mock:deps-test",
                rm=True
            )
            
            # Test key dependencies are available
            required_packages = ["websockets", "pytest", "pytest_asyncio", "pydantic", "structlog"]
            
            for package in required_packages:
                result = docker_client.containers.run(
                    image=image,
                    command=["python", "-c", f"import {package}; print(f'{package} imported successfully')"],
                    remove=True,
                    stderr=True,
                    stdout=True
                )
                output = result.decode('utf-8') if isinstance(result, bytes) else str(result)
                assert "imported successfully" in output, f"Package {package} should be importable"
                
        except docker.errors.BuildError as e:
            pytest.fail(f"Docker build failed: {e}")
            
    def test_container_starts_and_runs(self, docker_client, project_root):
        """Test that the container starts and runs without crashing"""
        try:
            # Build image first
            image, _ = docker_client.images.build(
                path=str(project_root),
                dockerfile="docker/Dockerfile.mock",
                tag="pi-whispr-mock:runtime-test", 
                rm=True
            )
            
            # Start container with unique port to avoid conflicts
            container = docker_client.containers.run(
                image=image,
                ports={"8765/tcp": 8771},  # Use port 8771 to avoid conflicts
                detach=True,
                remove=True,
                name="mock-test-runtime"
            )
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Check container is still running
            container.reload()
            assert container.status == "running", "Container should be running"
            
            # Check logs for successful startup
            logs = container.logs().decode('utf-8')
            assert "Mock WebSocket server started successfully" in logs or "MockWhisperServer initialized" in logs, "Server should start successfully"
            
            # Clean up
            container.stop()
            
        except docker.errors.ContainerError as e:
            pytest.fail(f"Container failed to start: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error testing container: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_server_responds(self, docker_client, project_root):
        """Test that the WebSocket server in the container responds to connections"""
        container = None
        try:
            # Build image
            image, _ = docker_client.images.build(
                path=str(project_root),
                dockerfile="docker/Dockerfile.mock",
                tag="pi-whispr-mock:websocket-test",
                rm=True
            )
            
            # Start container with unique port
            container = docker_client.containers.run(
                image=image,
                ports={"8765/tcp": 8772},  # Use port 8772 to avoid conflicts
                detach=True,
                name="mock-test-websocket"
            )
            
            # Wait for server to start
            await asyncio.sleep(5)
            
            # Test WebSocket connection
            uri = "ws://localhost:8772"
            
            try:
                async with websockets.connect(uri, open_timeout=10) as websocket:
                    # Send a PING message
                    ping_message = {
                        "header": {
                            "message_type": "PING",
                            "timestamp": time.time(),
                            "sequence_id": 1,
                            "correlation_id": "test-123"
                        },
                        "payload": {}
                    }
                    
                    await websocket.send(json.dumps(ping_message))
                    
                    # Wait for PONG response
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    
                    # Verify response
                    assert response_data["header"]["message_type"] == "PONG", "Should receive PONG response"
                    assert response_data["header"]["correlation_id"] == "test-123", "Should maintain correlation ID"
                    
            except asyncio.TimeoutError:
                pytest.fail("WebSocket connection or response timed out")
            except websockets.exceptions.ConnectionRefused:
                pytest.fail("Could not connect to WebSocket server")
                
        except Exception as e:
            pytest.fail(f"WebSocket test failed: {e}")
        finally:
            if container:
                try:
                    container.stop()
                    container.remove()
                except:
                    pass  # Container might already be stopped
    
    def test_health_check_functionality(self, docker_client, project_root):
        """Test that the health check works correctly"""
        try:
            # Build image
            image, _ = docker_client.images.build(
                path=str(project_root),
                dockerfile="docker/Dockerfile.mock",
                tag="pi-whispr-mock:health-test",
                rm=True
            )
            
            # Start container with unique port
            container = docker_client.containers.run(
                image=image,
                ports={"8765/tcp": 8773},  # Use port 8773 to avoid conflicts
                detach=True,
                name="mock-test-health"
            )
            
            # Wait for startup and health check to run
            time.sleep(10)
            
            # Check health status
            container.reload()
            health_status = container.attrs.get("State", {}).get("Health", {}).get("Status")
            
            # Health check should be healthy or starting (it takes time to be marked healthy)
            assert health_status in ["healthy", "starting"], f"Health check should be healthy or starting, got: {health_status}"
            
            # Clean up
            container.stop()
            container.remove()
            
        except Exception as e:
            pytest.fail(f"Health check test failed: {e}")
    
    def test_environment_variables_set(self, docker_client, project_root):
        """Test that required environment variables are set correctly"""
        try:
            # Build image
            image, _ = docker_client.images.build(
                path=str(project_root),
                dockerfile="docker/Dockerfile.mock",
                tag="pi-whispr-mock:env-test",
                rm=True
            )
            
            # Check PYTHONPATH is set
            result = docker_client.containers.run(
                image=image,
                command=["sh", "-c", "echo $PYTHONPATH"],
                remove=True,
                stderr=True,
                stdout=True
            )
            output = result.decode('utf-8') if isinstance(result, bytes) else str(result)
            assert "/app" in output, "PYTHONPATH should be set to /app"
            
        except Exception as e:
            pytest.fail(f"Environment variable test failed: {e}")
            
    def test_working_directory_correct(self, docker_client, project_root):
        """Test that working directory is set correctly"""
        try:
            # Build image
            image, _ = docker_client.images.build(
                path=str(project_root),
                dockerfile="docker/Dockerfile.mock",
                tag="pi-whispr-mock:workdir-test",
                rm=True
            )
            
            # Check working directory
            result = docker_client.containers.run(
                image=image,
                command=["pwd"],
                remove=True,
                stderr=True,
                stdout=True
            )
            output = result.decode('utf-8') if isinstance(result, bytes) else str(result)
            assert "/app" in output.strip(), "Working directory should be /app"
            
        except Exception as e:
            pytest.fail(f"Working directory test failed: {e}")
    
    def test_file_structure_in_container(self, docker_client, project_root):
        """Test that all required files are present in the container"""
        try:
            # Build image  
            image, _ = docker_client.images.build(
                path=str(project_root),
                dockerfile="docker/Dockerfile.mock",
                tag="pi-whispr-mock:files-test",
                rm=True
            )
            
            # Check required files exist
            required_paths = [
                "/app/server/mock_server.py",
                "/app/shared",
                "/app/healthcheck.py"
            ]
            
            for path in required_paths:
                command = ["sh", "-c", f"test -e {path} && echo 'EXISTS' || echo 'MISSING'"]
                result = docker_client.containers.run(
                    image=image,
                    command=command,
                    remove=True,
                    stderr=True,
                    stdout=True
                )
                output = result.decode('utf-8') if isinstance(result, bytes) else str(result)
                assert "EXISTS" in output, f"Required path {path} should exist in container"
                
        except Exception as e:
            pytest.fail(f"File structure test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 