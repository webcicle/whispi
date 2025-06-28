"""
Test suite for Audio Stream Buffer System (Task 5.1)

Tests for:
- Audio chunk buffering and management
- Compression functionality 
- Buffer overflow handling
- Reconnection buffering
- Performance monitoring
- Thread safety
"""

import pytest
import time
import threading
import zlib
from client.audio_stream_buffer import (
    AudioStreamBuffer, 
    ReconnectionBuffer, 
    AudioChunkData, 
    BufferState
)


class TestAudioStreamBuffer:
    """Test AudioStreamBuffer functionality"""
    
    @pytest.fixture
    def audio_buffer(self):
        """Create audio buffer for testing"""
        # Disable compression for basic tests to avoid unexpected compression
        return AudioStreamBuffer(max_buffer_size=10, compression_enabled=False)
    
    @pytest.fixture
    def sample_audio_data(self):
        """Create sample audio data for testing"""
        return b'\x00\x01' * 320  # 640 bytes - standard 20ms chunk
    
    def test_buffer_initialization(self, audio_buffer):
        """Test buffer initializes correctly"""
        assert len(audio_buffer.buffer) == 0
        assert audio_buffer.state == BufferState.IDLE
        assert audio_buffer.sequence_counter == 0
        assert audio_buffer.max_buffer_size == 10
    
    def test_add_chunk_basic(self, audio_buffer, sample_audio_data):
        """Test basic chunk addition"""
        timestamp = time.time()
        result = audio_buffer.add_chunk(sample_audio_data, 0, timestamp)
        
        assert result == True
        assert len(audio_buffer.buffer) == 1
        assert audio_buffer.state == BufferState.BUFFERING
        assert audio_buffer.sequence_counter == 1
    
    def test_get_chunk_basic(self, audio_buffer, sample_audio_data):
        """Test basic chunk retrieval"""
        timestamp = time.time()
        audio_buffer.add_chunk(sample_audio_data, 0, timestamp)
        
        chunk = audio_buffer.get_next_chunk()
        
        assert chunk is not None
        assert chunk.data == sample_audio_data
        assert chunk.chunk_index == 0
        assert chunk.timestamp == timestamp
        assert chunk.sequence_id == 0
        assert len(audio_buffer.buffer) == 0
        # After getting all chunks, buffer should return to IDLE
        assert audio_buffer.state == BufferState.IDLE
    
    def test_chunk_sequencing(self, audio_buffer, sample_audio_data):
        """Test chunks are properly sequenced"""
        timestamps = [time.time() + i * 0.02 for i in range(3)]
        
        # Add multiple chunks
        for i, ts in enumerate(timestamps):
            audio_buffer.add_chunk(sample_audio_data, i, ts)
        
        # Retrieve and verify sequence
        for i in range(3):
            chunk = audio_buffer.get_next_chunk()
            assert chunk.chunk_index == i
            assert chunk.sequence_id == i
            assert chunk.timestamp == timestamps[i]
    
    def test_compression_functionality(self):
        """Test audio compression"""
        # Create compressible data (repeated pattern)
        large_data = b'\x00\x01' * 1000  # 2KB
        
        buffer = AudioStreamBuffer(compression_enabled=True, compression_threshold=500)
        timestamp = time.time()
        
        result = buffer.add_chunk(large_data, 0, timestamp)
        assert result == True
        
        chunk = buffer.get_next_chunk()
        assert chunk.compressed == True
        assert len(chunk.data) < len(large_data)  # Should be compressed
        
        # Verify decompression works
        decompressed = zlib.decompress(chunk.data)
        assert decompressed == large_data
    
    def test_compression_threshold(self):
        """Test compression threshold"""
        small_data = b'\x00\x01' * 100  # 200 bytes
        
        buffer = AudioStreamBuffer(compression_enabled=True, compression_threshold=500)
        buffer.add_chunk(small_data, 0, time.time())
        
        chunk = buffer.get_next_chunk()
        assert chunk.compressed == False
        assert chunk.data == small_data
    
    def test_buffer_overflow_drop_oldest(self, sample_audio_data):
        """Test buffer overflow with drop_oldest strategy"""
        buffer = AudioStreamBuffer(max_buffer_size=3, overflow_strategy="drop_oldest")
        
        # Fill buffer beyond capacity
        for i in range(5):
            buffer.add_chunk(sample_audio_data, i, time.time() + i * 0.02)
        
        # Should only have 3 chunks (newest ones)
        assert len(buffer.buffer) == 3
        assert buffer.buffer_overflows == 2
        
        # First chunk should be index 2 (oldest 0,1 were dropped)
        first_chunk = buffer.get_next_chunk()
        assert first_chunk.chunk_index == 2
    
    def test_buffer_overflow_drop_newest(self, sample_audio_data):
        """Test buffer overflow with drop_newest strategy"""
        buffer = AudioStreamBuffer(max_buffer_size=3, overflow_strategy="drop_newest")
        
        # Fill buffer to capacity
        for i in range(3):
            buffer.add_chunk(sample_audio_data, i, time.time() + i * 0.02)
        
        # Try to add more (should be rejected)
        result = buffer.add_chunk(sample_audio_data, 3, time.time() + 3 * 0.02)
        assert result == False
        assert len(buffer.buffer) == 3
        
        # Should still have original chunks 0, 1, 2
        first_chunk = buffer.get_next_chunk()
        assert first_chunk.chunk_index == 0
    
    def test_batch_retrieval(self, audio_buffer, sample_audio_data):
        """Test batch chunk retrieval"""
        # Add 5 chunks
        for i in range(5):
            audio_buffer.add_chunk(sample_audio_data, i, time.time() + i * 0.02)
        
        # Get batch of 3
        chunks = audio_buffer.get_chunks_batch(max_chunks=3)
        
        assert len(chunks) == 3
        assert len(audio_buffer.buffer) == 2  # 2 remaining
        
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_requeue_chunk(self, audio_buffer, sample_audio_data):
        """Test chunk requeuing functionality"""
        timestamp = time.time()
        audio_buffer.add_chunk(sample_audio_data, 0, timestamp)
        
        chunk = audio_buffer.get_next_chunk()
        assert len(audio_buffer.buffer) == 0
        
        # Requeue the chunk
        result = audio_buffer.requeue_chunk(chunk)
        assert result == True
        assert len(audio_buffer.buffer) == 1
        assert chunk.attempts == 1
        
        # Verify requeued chunk is at front
        requeued = audio_buffer.get_next_chunk()
        assert requeued.chunk_index == 0
        assert requeued.attempts == 1
    
    def test_requeue_max_attempts(self, audio_buffer, sample_audio_data):
        """Test max attempts limit for requeuing"""
        chunk_data = AudioChunkData(
            data=sample_audio_data,
            chunk_index=0,
            timestamp=time.time(),
            sequence_id=0,
            max_attempts=3  # Allow 3 attempts total
        )
        
        # First requeue (attempts = 1) - should succeed
        assert audio_buffer.requeue_chunk(chunk_data) == True  # attempts = 1
        # Second requeue (attempts = 2) - should succeed
        assert audio_buffer.requeue_chunk(chunk_data) == True  # attempts = 2  
        # Third requeue (attempts = 3) - should succeed but at limit
        assert audio_buffer.requeue_chunk(chunk_data) == True  # attempts = 3
        # Fourth requeue (attempts = 4) - should fail, exceeds max
        assert audio_buffer.requeue_chunk(chunk_data) == False  # attempts = 4, exceeds max
    
    def test_buffer_stats(self, audio_buffer, sample_audio_data):
        """Test buffer statistics"""
        # Add some chunks
        for i in range(3):
            audio_buffer.add_chunk(sample_audio_data, i, time.time())
        
        # State should be BUFFERING when chunks are in buffer but none sent yet
        stats = audio_buffer.get_buffer_stats()
        assert stats["state"] == BufferState.BUFFERING.value
        
        # Retrieve one
        audio_buffer.get_next_chunk()
        
        stats = audio_buffer.get_buffer_stats()
        
        assert stats["total_chunks_added"] == 3
        assert stats["total_chunks_sent"] == 1
        assert stats["buffer_size"] == 2
        # After sending chunks, should be in STREAMING state
        assert stats["state"] == BufferState.STREAMING.value
        assert stats["buffer_utilization"] == 0.2  # 2/10
    
    def test_clear_buffer(self, audio_buffer, sample_audio_data):
        """Test buffer clearing"""
        # Add chunks
        for i in range(5):
            audio_buffer.add_chunk(sample_audio_data, i, time.time())
        
        # Clear keeping 2 recent
        cleared = audio_buffer.clear_buffer(keep_recent=2)
        
        assert cleared == 3
        assert len(audio_buffer.buffer) == 2
        
        # Verify we kept the most recent chunks (3, 4)
        chunk1 = audio_buffer.get_next_chunk()
        chunk2 = audio_buffer.get_next_chunk()
        assert chunk1.chunk_index == 3
        assert chunk2.chunk_index == 4
    
    def test_thread_safety(self, sample_audio_data):
        """Test thread safety of buffer operations"""
        buffer = AudioStreamBuffer(max_buffer_size=100)
        results = []
        errors = []
        
        def add_chunks(start_index, count):
            try:
                for i in range(count):
                    buffer.add_chunk(sample_audio_data, start_index + i, time.time())
                results.append(f"Added {count} chunks starting from {start_index}")
            except Exception as e:
                errors.append(e)
        
        def get_chunks(count):
            try:
                retrieved = 0
                while retrieved < count:
                    chunk = buffer.get_next_chunk()
                    if chunk:
                        retrieved += 1
                    else:
                        time.sleep(0.001)  # Small delay if buffer empty
                results.append(f"Retrieved {retrieved} chunks")
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads - fix the args parameter
        threads = [
            threading.Thread(target=add_chunks, args=(0, 20)),
            threading.Thread(target=add_chunks, args=(20, 20)),
            threading.Thread(target=get_chunks, args=(15,)),  # Fixed: added comma for tuple
            threading.Thread(target=get_chunks, args=(15,))   # Fixed: added comma for tuple
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0
        assert len(results) == 4


class TestReconnectionBuffer:
    """Test ReconnectionBuffer functionality"""
    
    @pytest.fixture
    def reconnection_buffer(self):
        """Create reconnection buffer for testing"""
        return ReconnectionBuffer(max_buffer_time_seconds=5.0)
    
    @pytest.fixture
    def sample_chunk(self):
        """Create sample audio chunk"""
        return AudioChunkData(
            data=b'\x00\x01' * 320,
            chunk_index=0,
            timestamp=time.time(),
            sequence_id=0
        )
    
    def test_add_chunk(self, reconnection_buffer, sample_chunk):
        """Test adding chunk to reconnection buffer"""
        result = reconnection_buffer.add_chunk(sample_chunk)
        
        assert result == True
        assert len(reconnection_buffer.chunks) == 1
    
    def test_get_all_chunks(self, reconnection_buffer, sample_chunk):
        """Test retrieving all chunks"""
        reconnection_buffer.add_chunk(sample_chunk)
        
        chunks = reconnection_buffer.get_all_chunks()
        
        assert len(chunks) == 1
        assert chunks[0] == sample_chunk
        assert len(reconnection_buffer.chunks) == 0  # Should be cleared
    
    def test_time_based_cleanup(self, reconnection_buffer):
        """Test automatic cleanup of old chunks"""
        # Add old chunk
        old_chunk = AudioChunkData(
            data=b'\x00\x01' * 320,
            chunk_index=0,
            timestamp=time.time(),
            sequence_id=0
        )
        old_chunk.buffered_at = time.time() - 10.0  # 10 seconds ago
        reconnection_buffer.chunks.append(old_chunk)
        
        # Add new chunk (this should trigger cleanup)
        new_chunk = AudioChunkData(
            data=b'\x00\x01' * 320,
            chunk_index=1,
            timestamp=time.time(),
            sequence_id=1
        )
        
        reconnection_buffer.add_chunk(new_chunk)
        
        # Old chunk should be removed, only new one remains
        assert len(reconnection_buffer.chunks) == 1
        assert reconnection_buffer.chunks[0].chunk_index == 1
    
    def test_buffer_info(self, reconnection_buffer):
        """Test buffer information"""
        # Empty buffer
        info = reconnection_buffer.get_buffer_info()
        assert info["chunk_count"] == 0
        
        # Add chunks with different timestamps
        chunk1 = AudioChunkData(
            data=b'\x00\x01' * 320,
            chunk_index=0,
            timestamp=100.0,
            sequence_id=0
        )
        chunk2 = AudioChunkData(
            data=b'\x00\x01' * 320,
            chunk_index=1,
            timestamp=102.0,
            sequence_id=1
        )
        
        reconnection_buffer.add_chunk(chunk1)
        reconnection_buffer.add_chunk(chunk2)
        
        info = reconnection_buffer.get_buffer_info()
        assert info["chunk_count"] == 2
        assert info["time_span"] == 2.0  # 102.0 - 100.0
        assert info["oldest_chunk"] == 0
        assert info["newest_chunk"] == 1


class TestBufferIntegration:
    """Integration tests for buffer system"""
    
    def test_end_to_end_buffering(self):
        """Test complete buffering workflow"""
        buffer = AudioStreamBuffer(max_buffer_size=10, compression_enabled=True)
        sample_data = b'\x00\x01' * 500  # Compressible data
        
        # Add chunks
        for i in range(5):
            buffer.add_chunk(sample_data, i, time.time() + i * 0.02)
        
        # Verify buffering
        assert len(buffer.buffer) == 5
        assert buffer.state == BufferState.BUFFERING
        
        # Start streaming
        chunks = buffer.get_chunks_batch(3)
        assert len(chunks) == 3
        assert buffer.state == BufferState.STREAMING
        
        # Verify compression was applied
        compressed_chunks = [c for c in chunks if c.compressed]
        assert len(compressed_chunks) > 0
        
        # Get remaining chunks
        remaining = buffer.get_chunks_batch(5)
        assert len(remaining) == 2
        
        # Buffer should be empty and idle now
        assert len(buffer.buffer) == 0
        # After getting all chunks, should transition back to IDLE
        next_chunk = buffer.get_next_chunk()
        assert next_chunk is None
        assert buffer.state == BufferState.IDLE
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality"""
        buffer = AudioStreamBuffer(max_buffer_size=20)
        sample_data = b'\x00\x01' * 320
        
        start_time = time.time()
        
        # Add and process chunks over time
        for i in range(10):
            buffer.add_chunk(sample_data, i, time.time())
            if i % 2 == 0:  # Retrieve every other chunk
                buffer.get_next_chunk()
            time.sleep(0.001)  # Small delay
        
        stats = buffer.get_buffer_stats()
        
        assert stats["total_chunks_added"] == 10
        assert stats["total_chunks_sent"] == 5
        assert stats["throughput_chunks_per_sec"] > 0
        assert stats["buffer_utilization"] > 0
        assert stats["uptime_seconds"] > 0 