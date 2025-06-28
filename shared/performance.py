#!/usr/bin/env python3
"""
Performance Monitoring and Optimization Module (Task 1.5)

This module provides comprehensive performance tracking, measurement, and optimization
utilities for the Pi-Whispr WebSocket communication system.

Features:
- End-to-end latency measurement and tracking
- Message ordering validation
- Throughput monitoring for audio streaming and transcription
- Network quality assessment
- Performance optimization recommendations
- Real-time metrics collection and analysis
"""

import time
import asyncio
import threading
import statistics
import json
import psutil
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from .protocol import PerformanceMetricsPayload, MessageType, WebSocketMessage


class PerformanceLevel(Enum):
    """Performance quality levels"""
    EXCELLENT = "excellent"  # <50ms latency
    GOOD = "good"            # 50-80ms latency
    ACCEPTABLE = "acceptable"  # 80-100ms latency
    POOR = "poor"            # >100ms latency


@dataclass
class LatencyMeasurement:
    """Single latency measurement"""
    timestamp: float
    latency_ms: float
    message_type: MessageType
    sequence_id: Optional[int] = None
    correlation_id: Optional[str] = None


@dataclass
class ThroughputMeasurement:
    """Single throughput measurement"""
    timestamp: float
    bytes_transferred: int
    duration_seconds: float
    throughput_mbps: float
    direction: str  # 'upload' or 'download'


@dataclass
class MessageOrderTracker:
    """Tracks message ordering violations"""
    expected_sequence: int
    received_sequence: int
    timestamp: float
    message_type: MessageType


class PerformanceTracker:
    """Comprehensive performance tracking and analysis"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Measurement storage
        self.latency_history: deque = deque(maxlen=max_history)
        self.throughput_history: deque = deque(maxlen=max_history)
        self.order_violations: deque = deque(maxlen=max_history)
        
        # Real-time tracking
        self.pending_requests: Dict[str, float] = {}  # correlation_id -> start_time
        self.message_sequences: Dict[MessageType, int] = {}  # Expected next sequence per type
        
        # Performance statistics
        self.stats_lock = threading.Lock()
        self.current_stats: Dict[str, Any] = {}
        
        # Optimization settings
        self.latency_threshold_ms = 100.0
        self.throughput_threshold_mbps = 1.0
        self.optimization_callbacks: List[Callable] = []
    
    def start_request_timing(self, correlation_id: str) -> None:
        """Start timing a request-response cycle"""
        self.pending_requests[correlation_id] = time.time()
    
    def end_request_timing(self, correlation_id: str, message_type: MessageType, 
                          sequence_id: Optional[int] = None) -> Optional[float]:
        """End timing a request-response cycle and record latency"""
        if correlation_id not in self.pending_requests:
            return None
        
        start_time = self.pending_requests.pop(correlation_id)
        latency_ms = (time.time() - start_time) * 1000
        
        measurement = LatencyMeasurement(
            timestamp=time.time(),
            latency_ms=latency_ms,
            message_type=message_type,
            sequence_id=sequence_id,
            correlation_id=correlation_id
        )
        
        self.latency_history.append(measurement)
        self._update_latency_stats()
        self._check_performance_thresholds()
        
        return latency_ms
    
    def record_throughput(self, bytes_transferred: int, duration_seconds: float, 
                         direction: str = "upload") -> float:
        """Record throughput measurement"""
        throughput_bps = (bytes_transferred * 8) / duration_seconds
        throughput_mbps = throughput_bps / (1024 * 1024)
        
        measurement = ThroughputMeasurement(
            timestamp=time.time(),
            bytes_transferred=bytes_transferred,
            duration_seconds=duration_seconds,
            throughput_mbps=throughput_mbps,
            direction=direction
        )
        
        self.throughput_history.append(measurement)
        self._update_throughput_stats()
        
        return throughput_mbps
    
    def validate_message_order(self, message: WebSocketMessage) -> bool:
        """Validate message ordering and record violations"""
        msg_type = message.header.message_type
        received_seq = message.header.sequence_id
        
        expected_seq = self.message_sequences.get(msg_type, 0)
        
        if received_seq < expected_seq:
            # Out of order message
            violation = MessageOrderTracker(
                expected_sequence=expected_seq,
                received_sequence=received_seq,
                timestamp=time.time(),
                message_type=msg_type
            )
            self.order_violations.append(violation)
            return False
        
        # Update expected sequence
        self.message_sequences[msg_type] = max(expected_seq, received_seq + 1)
        return True
    
    def get_latency_stats(self, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get latency statistics for specified time window"""
        with self.stats_lock:
            measurements = self._filter_by_time_window(self.latency_history, window_seconds)
            
            if not measurements:
                return {}
            
            latencies = [m.latency_ms for m in measurements]
            
            return {
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "avg_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies),
                "p95_ms": self._percentile(latencies, 95),
                "p99_ms": self._percentile(latencies, 99),
                "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "count": len(latencies)
            }
    
    def get_throughput_stats(self, window_seconds: Optional[float] = None, 
                           direction: Optional[str] = None) -> Dict[str, float]:
        """Get throughput statistics for specified time window and direction"""
        with self.stats_lock:
            measurements = self._filter_by_time_window(self.throughput_history, window_seconds)
            
            if direction:
                measurements = [m for m in measurements if m.direction == direction]
            
            if not measurements:
                return {}
            
            throughputs = [m.throughput_mbps for m in measurements]
            
            return {
                "min_mbps": min(throughputs),
                "max_mbps": max(throughputs),
                "avg_mbps": statistics.mean(throughputs),
                "median_mbps": statistics.median(throughputs),
                "total_bytes": sum(m.bytes_transferred for m in measurements),
                "count": len(throughputs)
            }
    
    def get_ordering_violations(self, window_seconds: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get message ordering violations"""
        violations = self._filter_by_time_window(self.order_violations, window_seconds)
        
        return [
            {
                "timestamp": v.timestamp,
                "expected_sequence": v.expected_sequence,
                "received_sequence": v.received_sequence,
                "message_type": v.message_type.value,
                "violation_magnitude": abs(v.expected_sequence - v.received_sequence)
            }
            for v in violations
        ]
    
    def get_performance_level(self) -> PerformanceLevel:
        """Determine current performance level"""
        stats = self.get_latency_stats(window_seconds=30.0)  # Last 30 seconds
        
        if not stats:
            return PerformanceLevel.GOOD
        
        avg_latency = stats.get("avg_ms", 0)
        
        if avg_latency < 50:
            return PerformanceLevel.EXCELLENT
        elif avg_latency < 80:
            return PerformanceLevel.GOOD
        elif avg_latency < 100:
            return PerformanceLevel.ACCEPTABLE
        else:
            return PerformanceLevel.POOR
    
    def get_performance_metrics_payload(self) -> PerformanceMetricsPayload:
        """Generate performance metrics payload for transmission"""
        latency_stats = self.get_latency_stats(window_seconds=60.0)
        throughput_stats = self.get_throughput_stats(window_seconds=60.0)
        
        return PerformanceMetricsPayload(
            latency_ms=latency_stats.get("avg_ms", 0),
            throughput_mbps=throughput_stats.get("avg_mbps", 0),
            cpu_usage=self._get_cpu_usage(),
            memory_usage=self._get_memory_usage(),
            network_quality=self._calculate_network_quality(),
            processing_queue_size=len(self.pending_requests),
            error_count=len(self.order_violations),
            uptime_seconds=time.time() - getattr(self, '_start_time', time.time())
        )
    
    def add_optimization_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for performance optimization events"""
        self.optimization_callbacks.append(callback)
    
    def _update_latency_stats(self) -> None:
        """Update cached latency statistics"""
        with self.stats_lock:
            self.current_stats.update(self.get_latency_stats())
    
    def _update_throughput_stats(self) -> None:
        """Update cached throughput statistics"""
        with self.stats_lock:
            throughput_stats = self.get_throughput_stats()
            self.current_stats.update({
                f"throughput_{k}": v for k, v in throughput_stats.items()
            })
    
    def _check_performance_thresholds(self) -> None:
        """Check if performance optimization is needed"""
        stats = self.get_latency_stats(window_seconds=10.0)
        
        if not stats:
            return
        
        avg_latency = stats.get("avg_ms", 0)
        
        if avg_latency > self.latency_threshold_ms:
            optimization_data = {
                "type": "high_latency",
                "avg_latency_ms": avg_latency,
                "threshold_ms": self.latency_threshold_ms,
                "recommendations": self._get_latency_optimization_recommendations(stats)
            }
            
            for callback in self.optimization_callbacks:
                try:
                    callback(optimization_data)
                except Exception as e:
                    # Log error but don't fail
                    pass
    
    def _get_latency_optimization_recommendations(self, stats: Dict[str, float]) -> List[str]:
        """Generate latency optimization recommendations"""
        recommendations = []
        
        avg_latency = stats.get("avg_ms", 0)
        p95_latency = stats.get("p95_ms", 0)
        
        if avg_latency > 100:
            recommendations.append("Consider reducing message size or enabling compression")
        
        if p95_latency > avg_latency * 2:
            recommendations.append("High latency variance detected - check network stability")
        
        if len(self.pending_requests) > 10:
            recommendations.append("High number of pending requests - consider request batching")
        
        return recommendations
    
    def _filter_by_time_window(self, measurements: deque, 
                              window_seconds: Optional[float]) -> List[Any]:
        """Filter measurements by time window"""
        if window_seconds is None:
            return list(measurements)
        
        cutoff_time = time.time() - window_seconds
        return [m for m in measurements if m.timestamp >= cutoff_time]
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    def _calculate_network_quality(self) -> float:
        """Calculate network quality score (0-1)"""
        latency_stats = self.get_latency_stats(window_seconds=30.0)
        
        if not latency_stats:
            return 1.0
        
        avg_latency = latency_stats.get("avg_ms", 0)
        order_violations = len(self.get_ordering_violations(window_seconds=30.0))
        
        # Calculate quality based on latency and ordering
        latency_score = max(0, 1 - (avg_latency / 200))  # 200ms = 0 score
        ordering_score = max(0, 1 - (order_violations / 10))  # 10 violations = 0 score
        
        return (latency_score + ordering_score) / 2


class PerformanceOptimizer:
    """Performance optimization engine"""
    
    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Register optimization callback
        self.tracker.add_optimization_callback(self._handle_optimization_event)
    
    def _handle_optimization_event(self, data: Dict[str, Any]) -> None:
        """Handle performance optimization events"""
        optimization_type = data.get("type")
        
        if optimization_type == "high_latency":
            self._optimize_latency(data)
        elif optimization_type == "low_throughput":
            self._optimize_throughput(data)
        elif optimization_type == "ordering_violations":
            self._optimize_ordering(data)
    
    def _optimize_latency(self, data: Dict[str, Any]) -> None:
        """Implement latency optimizations"""
        recommendations = data.get("recommendations", [])
        
        optimization = {
            "timestamp": time.time(),
            "type": "latency",
            "trigger_latency_ms": data.get("avg_latency_ms"),
            "actions_taken": [],
            "recommendations": recommendations
        }
        
        # Implement specific optimizations here
        # For now, just log the event
        self.optimization_history.append(optimization)
    
    def _optimize_throughput(self, data: Dict[str, Any]) -> None:
        """Implement throughput optimizations"""
        # Implementation for throughput optimization
        pass
    
    def _optimize_ordering(self, data: Dict[str, Any]) -> None:
        """Implement message ordering optimizations"""
        # Implementation for ordering optimization
        pass


class NetworkQualityAssessment:
    """Network quality assessment and monitoring"""
    
    def __init__(self):
        self.ping_history: deque = deque(maxlen=100)
        self.packet_loss_history: deque = deque(maxlen=100)
    
    async def assess_network_quality(self, target_host: str = "8.8.8.8") -> Dict[str, Any]:
        """Assess current network quality"""
        # Implement ping test
        ping_results = await self._ping_test(target_host)
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(ping_results)
        
        return {
            "quality_score": quality_score,
            "ping_avg_ms": ping_results.get("avg_ms", 0),
            "ping_jitter_ms": ping_results.get("jitter_ms", 0),
            "packet_loss_percent": ping_results.get("packet_loss", 0),
            "timestamp": time.time()
        }
    
    async def _ping_test(self, host: str, count: int = 5) -> Dict[str, float]:
        """Perform ping test to assess network latency"""
        # This is a simplified implementation
        # Real implementation would use actual ping
        
        import random
        ping_times = []
        
        for _ in range(count):
            # Simulate ping time with some randomness
            ping_time = random.uniform(10, 50)  # 10-50ms
            ping_times.append(ping_time)
            await asyncio.sleep(0.1)
        
        return {
            "avg_ms": statistics.mean(ping_times),
            "min_ms": min(ping_times),
            "max_ms": max(ping_times),
            "jitter_ms": statistics.stdev(ping_times) if len(ping_times) > 1 else 0,
            "packet_loss": 0.0  # Simplified
        }
    
    def _calculate_quality_score(self, ping_results: Dict[str, float]) -> float:
        """Calculate network quality score from ping results"""
        avg_ping = ping_results.get("avg_ms", 0)
        jitter = ping_results.get("jitter_ms", 0)
        packet_loss = ping_results.get("packet_loss", 0)
        
        # Score based on ping (0-1, lower ping = higher score)
        ping_score = max(0, 1 - (avg_ping / 100))
        
        # Score based on jitter (0-1, lower jitter = higher score)
        jitter_score = max(0, 1 - (jitter / 20))
        
        # Score based on packet loss (0-1, no loss = 1.0)
        loss_score = max(0, 1 - (packet_loss / 10))
        
        # Weighted average
        return (ping_score * 0.5) + (jitter_score * 0.3) + (loss_score * 0.2)


# Global performance tracker instance
_global_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get or create global performance tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
        _global_tracker._start_time = time.time()
    return _global_tracker


def initialize_performance_monitoring(max_history: int = 1000) -> PerformanceTracker:
    """Initialize global performance monitoring"""
    global _global_tracker
    _global_tracker = PerformanceTracker(max_history=max_history)
    _global_tracker._start_time = time.time()
    return _global_tracker 