"""
Audio Streaming Buffer System for Pi-Whispr (Task 5.1)

This module implements a sophisticated buffering system for audio streaming that:
- Manages audio chunks before transmission to handle network latency
- Provides automatic buffer overflow protection
- Supports compression to reduce bandwidth usage
- Handles sequence numbering and timestamps consistently
- Provides reconnection buffering during connection issues
- Offers performance monitoring and optimization
"""

import asyncio
import time
import threading
import zlib
import base64
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class BufferState(Enum):
    """Audio buffer state enumeration"""
    IDLE = "idle"
    BUFFERING = "buffering"
    STREAMING = "streaming"
    OVERFLOW = "overflow"
    ERROR = "error"


@dataclass
class AudioChunkData:
    """Enhanced audio chunk with metadata for streaming"""
    data: bytes
    chunk_index: int
    timestamp: float
    sequence_id: int
    is_final: bool = False
    energy_level: Optional[float] = None
    compressed: bool = False
    buffered_at: float = field(default_factory=time.time)
    attempts: int = 0
    max_attempts: int = 3


class AudioStreamBuffer:
    """
    Advanced audio streaming buffer for managing chunks before transmission
    
    Features:
    - Automatic overflow protection with configurable max size
    - Compression support with adaptive threshold
    - Sequence numbering and timing consistency
    - Performance monitoring and adaptive behavior
    - Thread-safe operations for real-time audio processing
    """
    
    def __init__(self, 
                 max_buffer_size: int = 100,
                 compression_enabled: bool = True,
                 compression_threshold: int = 500,  # bytes
                 overflow_strategy: str = "drop_oldest"):
        
        # Buffer configuration
        self.max_buffer_size = max_buffer_size
        self.compression_enabled = compression_enabled
        self.compression_threshold = compression_threshold
        self.overflow_strategy = overflow_strategy  # "drop_oldest", "drop_newest", "compress_all"
        
        # Buffer state
        self.buffer: deque = deque()
        self.state = BufferState.IDLE
        self.sequence_counter = 0
        self.chunk_counter = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Performance tracking
        self.total_chunks_added = 0
        self.total_chunks_sent = 0
        self.total_bytes_buffered = 0
        self.compression_savings = 0
        self.buffer_overflows = 0
        self.start_time = time.time()
        
        # Callbacks
        self.overflow_callback: Optional[Callable[[int], None]] = None
        self.compression_callback: Optional[Callable[[int, int], None]] = None
    
    def add_chunk(self, 
                  audio_data: bytes, 
                  chunk_index: int, 
                  timestamp: float,
                  is_final: bool = False,
                  energy_level: Optional[float] = None) -> bool:
        """
        Add audio chunk to buffer with automatic compression and overflow handling
        
        Args:
            audio_data: Raw audio data
            chunk_index: Sequential chunk index
            timestamp: Chunk timestamp
            is_final: Whether this is the final chunk
            energy_level: Audio energy level for VAD
            
        Returns:
            bool: True if chunk was successfully added
        """
        with self.lock:
            try:
                # Check for buffer overflow
                if len(self.buffer) >= self.max_buffer_size:
                    if not self._handle_overflow():
                        logger.warning(f"Buffer overflow - failed to add chunk {chunk_index}")
                        return False
                
                # Decide whether to compress
                should_compress = (
                    self.compression_enabled and 
                    len(audio_data) > self.compression_threshold
                )
                
                # Compress if needed
                final_data = audio_data
                compressed = False
                if should_compress:
                    try:
                        compressed_data = zlib.compress(audio_data, level=6)
                        if len(compressed_data) < len(audio_data):
                            final_data = compressed_data
                            compressed = True
                            self.compression_savings += len(audio_data) - len(compressed_data)
                            
                            if self.compression_callback:
                                self.compression_callback(len(audio_data), len(compressed_data))
                    except Exception as e:
                        logger.warning(f"Compression failed for chunk {chunk_index}: {e}")
                
                # Create chunk data
                chunk_data = AudioChunkData(
                    data=final_data,
                    chunk_index=chunk_index,
                    timestamp=timestamp,
                    sequence_id=self.sequence_counter,
                    is_final=is_final,
                    energy_level=energy_level,
                    compressed=compressed
                )
                
                # Add to buffer
                self.buffer.append(chunk_data)
                self.sequence_counter += 1
                self.chunk_counter += 1
                self.total_chunks_added += 1
                self.total_bytes_buffered += len(final_data)
                
                # Update state
                if self.state == BufferState.IDLE:
                    self.state = BufferState.BUFFERING
                
                return True
                
            except Exception as e:
                logger.error(f"Error adding chunk to buffer: {e}")
                self.state = BufferState.ERROR
                return False
    
    def get_next_chunk(self) -> Optional[AudioChunkData]:
        """
        Get the next chunk from buffer for transmission
        
        Returns:
            AudioChunkData or None if buffer is empty
        """
        with self.lock:
            if not self.buffer:
                if self.state == BufferState.STREAMING:
                    self.state = BufferState.IDLE
                return None
            
            chunk = self.buffer.popleft()
            self.total_chunks_sent += 1
            
            if self.state != BufferState.STREAMING:
                self.state = BufferState.STREAMING
            
            # Check if buffer is now empty and transition to IDLE
            if not self.buffer and self.state == BufferState.STREAMING:
                self.state = BufferState.IDLE
            
            return chunk
    
    def peek_next_chunk(self) -> Optional[AudioChunkData]:
        """Peek at next chunk without removing it from buffer"""
        with self.lock:
            return self.buffer[0] if self.buffer else None
    
    def get_chunks_batch(self, max_chunks: int = 5) -> List[AudioChunkData]:
        """
        Get multiple chunks for batch transmission
        
        Args:
            max_chunks: Maximum number of chunks to return
            
        Returns:
            List of AudioChunkData
        """
        with self.lock:
            chunks = []
            for _ in range(min(max_chunks, len(self.buffer))):
                if self.buffer:
                    chunks.append(self.buffer.popleft())
                    self.total_chunks_sent += 1
            
            if chunks and self.state != BufferState.STREAMING:
                self.state = BufferState.STREAMING
            elif not self.buffer and self.state == BufferState.STREAMING:
                self.state = BufferState.IDLE
            
            return chunks
    
    def requeue_chunk(self, chunk: AudioChunkData, increment_attempts: bool = True) -> bool:
        """
        Requeue a chunk that failed to send (for reconnection scenarios)
        
        Args:
            chunk: The chunk to requeue
            increment_attempts: Whether to increment the attempt counter
            
        Returns:
            bool: True if successfully requeued
        """
        with self.lock:
            if increment_attempts:
                chunk.attempts += 1
                
            # Check if we've exceeded max attempts
            if chunk.attempts > chunk.max_attempts:  # Changed from >= to >
                logger.warning(f"Dropping chunk {chunk.chunk_index} after {chunk.attempts} failed attempts")
                return False
            
            # Add back to front of queue (LIFO for failed chunks)
            self.buffer.appendleft(chunk)
            return True
    
    def clear_buffer(self, keep_recent: int = 0) -> int:
        """
        Clear buffer, optionally keeping recent chunks
        
        Args:
            keep_recent: Number of most recent chunks to keep
            
        Returns:
            int: Number of chunks cleared
        """
        with self.lock:
            cleared_count = len(self.buffer) - keep_recent
            
            if keep_recent > 0 and len(self.buffer) > keep_recent:
                # Keep only the most recent chunks
                recent_chunks = []
                for _ in range(keep_recent):
                    if self.buffer:
                        recent_chunks.append(self.buffer.pop())
                
                self.buffer.clear()
                self.buffer.extend(reversed(recent_chunks))
            else:
                self.buffer.clear()
            
            if cleared_count > 0:
                self.state = BufferState.IDLE if not self.buffer else BufferState.BUFFERING
            
            return max(0, cleared_count)
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            return {
                "state": self.state.value,
                "buffer_size": len(self.buffer),
                "max_buffer_size": self.max_buffer_size,
                "sequence_counter": self.sequence_counter,
                "chunk_counter": self.chunk_counter,
                "total_chunks_added": self.total_chunks_added,
                "total_chunks_sent": self.total_chunks_sent,
                "total_bytes_buffered": self.total_bytes_buffered,
                "compression_enabled": self.compression_enabled,
                "compression_savings_bytes": self.compression_savings,
                "compression_ratio": self.compression_savings / max(1, self.total_bytes_buffered),
                "buffer_overflows": self.buffer_overflows,
                "throughput_chunks_per_sec": self.total_chunks_sent / max(1, uptime),
                "pending_chunks": len(self.buffer),
                "buffer_utilization": len(self.buffer) / self.max_buffer_size,
                "uptime_seconds": uptime
            }
    
    def _handle_overflow(self) -> bool:
        """
        Handle buffer overflow based on configured strategy
        
        Returns:
            bool: True if overflow was handled successfully
        """
        self.buffer_overflows += 1
        
        if self.overflow_callback:
            self.overflow_callback(len(self.buffer))
        
        if self.overflow_strategy == "drop_oldest":
            # Remove oldest chunk
            if self.buffer:
                dropped = self.buffer.popleft()
                logger.debug(f"Dropped oldest chunk {dropped.chunk_index} due to overflow")
                return True
        
        elif self.overflow_strategy == "drop_newest":
            # Don't add the new chunk
            logger.debug("Dropping newest chunk due to overflow")
            return False
        
        elif self.overflow_strategy == "compress_all":
            # Try to compress all uncompressed chunks
            compressed_any = False
            for chunk in self.buffer:
                if not chunk.compressed and len(chunk.data) > 100:
                    try:
                        compressed_data = zlib.compress(chunk.data, level=9)  # Max compression
                        if len(compressed_data) < len(chunk.data):
                            chunk.data = compressed_data
                            chunk.compressed = True
                            compressed_any = True
                    except Exception as e:
                        logger.warning(f"Emergency compression failed: {e}")
            
            if compressed_any:
                logger.debug("Emergency compression applied during overflow")
                return True
            else:
                # Fall back to dropping oldest
                if self.buffer:
                    dropped = self.buffer.popleft()
                    logger.debug(f"Dropped oldest chunk {dropped.chunk_index} after compression failed")
                    return True
        
        return False
    
    def set_overflow_callback(self, callback: Callable[[int], None]):
        """Set callback for buffer overflow events"""
        self.overflow_callback = callback
    
    def set_compression_callback(self, callback: Callable[[int, int], None]):
        """Set callback for compression events (original_size, compressed_size)"""
        self.compression_callback = callback
    
    def optimize_settings(self, target_latency_ms: float = 100.0):
        """
        Automatically optimize buffer settings based on performance metrics
        
        Args:
            target_latency_ms: Target latency in milliseconds
        """
        stats = self.get_buffer_stats()
        
        # Adjust buffer size based on throughput and overflow rate
        if stats["buffer_overflows"] > 5 and stats["uptime_seconds"] > 10:
            # Increase buffer size if we're overflowing frequently
            new_size = min(self.max_buffer_size * 1.2, 200)
            logger.info(f"Increasing buffer size from {self.max_buffer_size} to {new_size} due to overflows")
            self.max_buffer_size = int(new_size)
        
        # Adjust compression threshold based on compression ratio
        if stats["compression_ratio"] > 0.3:  # 30% compression savings
            # Compression is effective, lower threshold
            self.compression_threshold = max(self.compression_threshold * 0.8, 100)
        elif stats["compression_ratio"] < 0.1:  # Less than 10% savings
            # Compression not very effective, raise threshold
            self.compression_threshold = min(self.compression_threshold * 1.2, 1000)


class ReconnectionBuffer:
    """
    Specialized buffer for handling audio chunks during connection interruptions
    """
    
    def __init__(self, max_buffer_time_seconds: float = 10.0):
        self.max_buffer_time = max_buffer_time_seconds
        self.chunks: List[AudioChunkData] = []
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def add_chunk(self, chunk: AudioChunkData) -> bool:
        """Add chunk to reconnection buffer"""
        with self.lock:
            # Remove old chunks that exceed time limit
            cutoff_time = time.time() - self.max_buffer_time
            self.chunks = [c for c in self.chunks if c.buffered_at > cutoff_time]
            
            # Add new chunk
            self.chunks.append(chunk)
            return True
    
    def get_all_chunks(self) -> List[AudioChunkData]:
        """Get all buffered chunks for transmission after reconnection"""
        with self.lock:
            chunks = self.chunks.copy()
            self.chunks.clear()
            return chunks
    
    def clear(self):
        """Clear all buffered chunks"""
        with self.lock:
            self.chunks.clear()
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get reconnection buffer information"""
        with self.lock:
            if not self.chunks:
                return {"chunk_count": 0, "time_span": 0, "oldest_chunk": None, "newest_chunk": None}
            
            oldest = min(self.chunks, key=lambda c: c.timestamp)
            newest = max(self.chunks, key=lambda c: c.timestamp)
            
            return {
                "chunk_count": len(self.chunks),
                "time_span": newest.timestamp - oldest.timestamp,
                "oldest_chunk": oldest.chunk_index,
                "newest_chunk": newest.chunk_index
            } 