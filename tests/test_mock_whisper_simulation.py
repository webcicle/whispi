"""
Test suite for Mock faster-whisper Simulation Logic (Task 2.3)

Tests the enhanced mock transcription system that simulates faster-whisper
processing with realistic timing and responses based on model selection.
"""

import pytest
import asyncio
import json
import time
import base64
from unittest.mock import Mock, patch
from server.mock_server import MockWhisperServer
from shared.protocol import (
    MessageType, MessageHeader, WebSocketMessage,
    AudioDataPayload, TranscriptionResultPayload
)
from shared.constants import DEFAULT_MODEL


class TestMockWhisperSimulation:
    """Test enhanced faster-whisper simulation functionality"""
    
    @pytest.fixture
    def mock_server(self):
        """Create mock server instance for testing"""
        from server.mock_configuration import MockServerConfig, LatencyProfile, ErrorScenario, ResourceConstraints
        
        # Create minimal configuration for testing
        latency_profile = LatencyProfile(
            processing_delay_ms=0,  # No extra latency for testing
            network_delay_ms=0,
            variability_factor=0.0
        )
        
        error_scenario = ErrorScenario(
            failure_rate=0.0,  # No errors for basic testing
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
        
        config = MockServerConfig(
            host="localhost",
            port=8765,
            model_size="tiny.en",
            latency_profile=latency_profile,
            error_scenario=error_scenario,
            resource_constraints=resource_constraints,
            enable_detailed_logging=False
        )
        
        return MockWhisperServer(config=config)
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock websocket for testing"""
        websocket = Mock()
        # Create a proper async mock for sending
        async def mock_send(data):
            pass
        websocket.send = Mock(side_effect=mock_send)
        websocket.remote_address = ("127.0.0.1", 12345)
        return websocket
    
    @pytest.fixture
    def sample_audio_data(self):
        """Create sample base64-encoded audio data for testing"""
        # Simulate 1 second of 16kHz mono audio (16000 samples * 2 bytes)
        audio_bytes = b'\x00' * (16000 * 2)
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    @pytest.mark.asyncio
    async def test_model_specific_processing_times(self, mock_server):
        """Test that different models have appropriate processing times"""
        # Test tiny.en model (should be fastest)
        mock_server.model_size = "tiny.en"
        tiny_time = mock_server._get_mock_processing_time()
        
        # Test small.en model (should be slower)
        mock_server.model_size = "small.en"
        small_time = mock_server._get_mock_processing_time()
        
        # Test base.en model (should be slowest of these three)
        mock_server.model_size = "base.en"
        base_time = mock_server._get_mock_processing_time()
        
        # Verify relative timing relationships
        assert tiny_time < small_time < base_time
        assert tiny_time >= 0.1  # Minimum processing time
        assert base_time <= 2.0  # Maximum reasonable time for test
    
    @pytest.mark.asyncio
    async def test_audio_duration_based_processing(self, mock_server):
        """Test that processing time scales with audio duration"""
        # Short audio (0.5 seconds)
        short_duration = 0.5
        short_time = mock_server._calculate_processing_time_for_duration(short_duration)
        
        # Medium audio (2.0 seconds)
        medium_duration = 2.0
        medium_time = mock_server._calculate_processing_time_for_duration(medium_duration)
        
        # Long audio (5.0 seconds)
        long_duration = 5.0
        long_time = mock_server._calculate_processing_time_for_duration(long_duration)
        
        # Processing time should scale with audio duration
        assert short_time < medium_time < long_time
        
        # Check realistic ratios (faster-whisper typically processes faster than real-time)
        assert short_time <= short_duration  # Should be faster than real-time
        assert medium_time <= medium_duration
    
    @pytest.mark.asyncio
    async def test_realistic_transcription_generation(self, mock_server):
        """Test generation of realistic transcription responses"""
        # Test multiple generations to ensure variety
        transcriptions = []
        for _ in range(10):
            text = mock_server._generate_realistic_transcription()
            transcriptions.append(text)
            
            # Basic validation
            assert isinstance(text, str)
            assert len(text) > 0
            assert len(text) <= 200  # Reasonable length limit
        
        # Should have some variety (not all the same)
        unique_transcriptions = set(transcriptions)
        assert len(unique_transcriptions) > 1
    
    @pytest.mark.asyncio
    async def test_confidence_score_generation(self, mock_server):
        """Test realistic confidence score generation"""
        scores = []
        for _ in range(20):
            score = mock_server._generate_confidence_score()
            scores.append(score)
            
            # Validate confidence score range
            assert 0.0 <= score <= 1.0
            assert score >= 0.7  # Should be realistic (not too low)
        
        # Should have variety in scores
        assert len(set(scores)) > 1
        # Average should be in reasonable range
        avg_score = sum(scores) / len(scores)
        assert 0.8 <= avg_score <= 0.95
    
    @pytest.mark.asyncio
    async def test_audio_chunk_accumulation(self, mock_server, mock_websocket, sample_audio_data):
        """Test accumulation of audio chunks before processing"""
        client_id = "test_client_123"
        session_id = "session_456"
        
        # Register client first
        await mock_server._register_test_client(client_id, mock_websocket)
        
        # Send multiple audio chunks but don't send final chunk yet
        for chunk_index in range(2):  # Send 2 non-final chunks
            audio_message = self._create_audio_message(
                client_id, session_id, chunk_index, 
                sample_audio_data, is_final=False
            )
            
            await mock_server._handle_audio_data(mock_websocket, audio_message)
        
        # Verify audio chunks were accumulated (before final processing)
        assert client_id in mock_server.audio_buffers
        assert len(mock_server.audio_buffers[client_id]) == 2
        
        # Now send final chunk
        final_message = self._create_audio_message(
            client_id, session_id, 2, 
            sample_audio_data, is_final=True
        )
        await mock_server._handle_audio_data(mock_websocket, final_message)
        
        # Check that final chunk triggered processing and websocket.send was called
        mock_websocket.send.assert_called()
    
    @pytest.mark.asyncio
    async def test_partial_transcription_simulation(self, mock_server, mock_websocket, sample_audio_data):
        """Test simulation of partial transcription results"""
        client_id = "test_client_partial"
        session_id = "session_partial"
        
        # Enable partial results for this test
        mock_server.enable_partial_results = True
        
        await mock_server._register_test_client(client_id, mock_websocket)
        
        # Send audio chunk that should trigger partial result
        audio_message = self._create_audio_message(
            client_id, session_id, 0, sample_audio_data, 
            is_final=False, trigger_partial=True
        )
        
        await mock_server._handle_audio_data(mock_websocket, audio_message)
        
        # Should have sent partial result
        mock_websocket.send.assert_called()
        # Get the call arguments to verify it's a partial result
        call_args = mock_websocket.send.call_args[0]
        response_json = call_args[0]
        response = json.loads(response_json)
        
        assert response['header']['type'] == 'transcription_result'
        assert response['payload']['is_partial'] is True
    
    @pytest.mark.asyncio
    async def test_processing_queue_simulation(self, mock_server, mock_websocket, sample_audio_data):
        """Test simulation of processing queue and concurrent requests"""
        # Create multiple concurrent transcription requests
        tasks = []
        client_ids = []
        
        for i in range(3):
            client_id = f"client_{i}"
            client_ids.append(client_id)
            await mock_server._register_test_client(client_id, mock_websocket)
            
            audio_message = self._create_audio_message(
                client_id, f"session_{i}", 0, sample_audio_data, is_final=True
            )
            
            # Submit concurrent processing requests
            task = asyncio.create_task(
                mock_server._handle_audio_data(mock_websocket, audio_message)
            )
            tasks.append(task)
        
        # Wait for all to complete
        await asyncio.gather(*tasks)
        
        # Verify all were processed (each call should have resulted in a send)
        assert mock_websocket.send.call_count == 3
        
        # Check processing queue was managed
        assert mock_server.stats["total_transcriptions"] == 3
    
    @pytest.mark.asyncio
    async def test_error_simulation_scenarios(self, mock_server, mock_websocket, sample_audio_data):
        """Test simulation of various error scenarios"""
        client_id = "error_test_client"
        session_id = "error_session"
        
        await mock_server._register_test_client(client_id, mock_websocket)
        
        # Test with error injection enabled
        mock_server.error_injection_rate = 0.5  # 50% error rate for testing
        
        # Send multiple requests to trigger errors
        error_count = 0
        success_count = 0
        
        for i in range(10):
            mock_websocket.reset_mock()  # Reset for each iteration
            
            audio_message = self._create_audio_message(
                client_id, session_id, i, sample_audio_data, is_final=True
            )
            
            await mock_server._handle_audio_data(mock_websocket, audio_message)
            
            if mock_websocket.send.called:
                sent_args = mock_websocket.send.call_args[0]
                response_json = sent_args[0]
                response = json.loads(response_json)
                
                if response['header']['type'] == 'error':
                    error_count += 1
                else:
                    success_count += 1
        
        # Should have both successes and errors
        assert error_count > 0
        assert success_count > 0
    
    @pytest.mark.asyncio
    async def test_memory_usage_simulation(self, mock_server):
        """Test memory usage simulation during processing"""
        # Get baseline memory usage
        baseline_memory = mock_server._get_simulated_memory_usage()
        
        # Simulate processing load (use audio_duration parameter)
        mock_server._simulate_processing_load(2.0)  # 2 seconds of audio
        
        # Memory usage should increase during processing
        processing_memory = mock_server._get_simulated_memory_usage()
        assert processing_memory > baseline_memory
        
        # Wait for processing to complete
        await asyncio.sleep(0.1)
        
        # Memory should return closer to baseline
        post_processing_memory = mock_server._get_simulated_memory_usage()
        assert post_processing_memory < processing_memory
    
    @pytest.mark.asyncio
    async def test_temperature_simulation(self, mock_server):
        """Test CPU temperature simulation during sustained processing"""
        # Get baseline temperature
        baseline_temp = mock_server._get_simulated_temperature()
        assert 30.0 <= baseline_temp <= 45.0  # Reasonable idle temperature
        
        # Simulate sustained processing by increasing processing load
        for i in range(5):
            mock_server._simulate_processing_load(2.0)  # Simulate 2 seconds of processing each time
            mock_server._simulate_processing_heat()
        
        # Temperature should increase
        processing_temp = mock_server._get_simulated_temperature()
        assert processing_temp > baseline_temp
        assert processing_temp <= 80.0  # Should not exceed reasonable limits
    
    def _create_audio_message(self, client_id: str, session_id: str, 
                             chunk_index: int, audio_data: str, is_final: bool,
                             trigger_partial: bool = False) -> WebSocketMessage:
        """Helper to create audio data messages"""
        header = MessageHeader(
            message_type=MessageType.AUDIO_DATA,
            sequence_id=chunk_index + 1,
            timestamp=time.time(),
            client_id=client_id,
            session_id=session_id,
            correlation_id=f"corr_{chunk_index}"
        )
        
        payload = {
            "audio_data": audio_data,
            "chunk_index": chunk_index,
            "is_final": is_final,
            "sample_rate": 16000,
            "channels": 1,
            "format": "int16",
            "trigger_partial": trigger_partial
        }
        
        return WebSocketMessage(header=header, payload=payload)


class TestRealisticTimingBenchmarks:
    """Test realistic timing benchmarks for different scenarios"""
    
    @pytest.mark.asyncio
    async def test_tiny_en_model_performance(self):
        """Test tiny.en model performance simulation"""
        from server.mock_configuration import MockServerConfig, LatencyProfile, ErrorScenario, ResourceConstraints
        
        config = MockServerConfig(
            host="localhost",
            port=8765,
            model_size="tiny.en",
            latency_profile=LatencyProfile(0, 0, 0.0),
            error_scenario=ErrorScenario(0.0, 0.0, 0.0, []),
            resource_constraints=ResourceConstraints(2048, 80, 10, 0.75, 0.75),
            enable_detailed_logging=False
        )
        server = MockWhisperServer(config=config)
        
        # Test various audio durations
        durations = [0.5, 1.0, 2.0, 5.0]
        for duration in durations:
            processing_time = server._calculate_processing_time_for_duration(duration)
            
            # tiny.en should be very fast (much faster than real-time)
            # For very short audio clips, there's a minimum processing overhead
            if duration >= 1.0:
                assert processing_time <= duration * 0.3  # At most 30% of audio duration
            else:
                # For short clips, allow more overhead due to model loading/initialization
                assert processing_time <= duration * 0.5  # At most 50% for short clips
            assert processing_time >= 0.1  # Minimum processing time
    
    @pytest.mark.asyncio
    async def test_small_en_model_performance(self):
        """Test small.en model performance simulation"""
        from server.mock_configuration import MockServerConfig, LatencyProfile, ErrorScenario, ResourceConstraints
        
        config = MockServerConfig(
            host="localhost",
            port=8765,
            model_size="small.en",
            latency_profile=LatencyProfile(0, 0, 0.0),
            error_scenario=ErrorScenario(0.0, 0.0, 0.0, []),
            resource_constraints=ResourceConstraints(2048, 80, 10, 0.75, 0.75),
            enable_detailed_logging=False
        )
        server = MockWhisperServer(config=config)
        
        # Test various audio durations
        durations = [0.5, 1.0, 2.0, 5.0]
        for duration in durations:
            processing_time = server._calculate_processing_time_for_duration(duration)
            
            # small.en should be slower than tiny but still faster than real-time
            assert processing_time <= duration * 0.6  # At most 60% of audio duration
            assert processing_time >= 0.15  # Minimum processing time for small model
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_limits(self):
        """Test simulation of Pi 5 concurrent processing limits"""
        from server.mock_configuration import MockServerConfig, LatencyProfile, ErrorScenario, ResourceConstraints
        
        config = MockServerConfig(
            host="localhost",
            port=8765,
            model_size="small.en",
            latency_profile=LatencyProfile(0, 0, 0.0),
            error_scenario=ErrorScenario(0.0, 0.0, 0.0, []),
            resource_constraints=ResourceConstraints(2048, 80, 10, 0.75, 0.75),
            enable_detailed_logging=False
        )
        server = MockWhisperServer(config=config)
        
        # Simulate multiple concurrent requests
        start_time = time.time()
        
        # Process multiple requests concurrently
        durations = [1.0, 1.5, 2.0]
        processing_times = []
        
        for duration in durations:
            # Simulate processing with queue delays
            processing_time = server._calculate_processing_time_with_queue(
                duration, current_queue_size=len(processing_times)
            )
            processing_times.append(processing_time)
        
        # Later requests should take longer due to queue
        assert processing_times[1] >= processing_times[0]
        assert processing_times[2] >= processing_times[1] 