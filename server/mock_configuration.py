"""
Mock Server Configuration System (Task 2.4)

This module provides a comprehensive configuration system for the mock server
to simulate different latency profiles, error scenarios, and resource constraints.

Features:
- Latency profiles: fast, normal, slow, custom
- Error injection scenarios: none, light, moderate, heavy, custom
- Resource constraints: unlimited, Pi5 simulated, Pi5 stressed, custom
- Environment variable and config file support
- Predefined scenario configurations for easy testing
"""

import json
import os
import random
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LatencyProfile:
    """Configuration for latency simulation"""
    processing_delay_ms: int
    network_delay_ms: int
    variability_factor: float  # 0.0 to 1.0, adds random variation
    
    @classmethod
    def fast(cls):
        """Fast latency profile for rapid development"""
        return cls(
            processing_delay_ms=50,
            network_delay_ms=10,
            variability_factor=0.2
        )
    
    @classmethod
    def normal(cls):
        """Normal latency profile simulating realistic Pi performance"""
        return cls(
            processing_delay_ms=200,
            network_delay_ms=30,
            variability_factor=0.3
        )
    
    @classmethod
    def slow(cls):
        """Slow latency profile simulating stressed Pi or poor network"""
        return cls(
            processing_delay_ms=800,
            network_delay_ms=100,
            variability_factor=0.5
        )
    
    @classmethod
    def custom(cls, processing_delay_ms: int, network_delay_ms: int, variability_factor: float):
        """Create custom latency profile"""
        return cls(processing_delay_ms, network_delay_ms, variability_factor)
    
    def calculate_actual_latency(self) -> int:
        """Calculate actual latency with variability"""
        base_latency = self.processing_delay_ms + self.network_delay_ms
        if self.variability_factor > 0:
            # Add random variation from -variability_factor to +variability_factor
            variation = random.uniform(-self.variability_factor, self.variability_factor)
            actual_latency = base_latency * (1 + variation)
        else:
            actual_latency = base_latency
        
        return max(0, int(actual_latency))  # Ensure non-negative


# Predefined latency profiles
LatencyProfile.FAST = LatencyProfile.fast()
LatencyProfile.NORMAL = LatencyProfile.normal()
LatencyProfile.SLOW = LatencyProfile.slow()


@dataclass
class ErrorScenario:
    """Configuration for error injection simulation"""
    failure_rate: float  # 0.0 to 1.0, probability of connection failures
    timeout_rate: float  # 0.0 to 1.0, probability of request timeouts
    processing_error_rate: float  # 0.0 to 1.0, probability of processing errors
    error_types: List[str]  # Types of errors to inject
    
    @classmethod
    def none(cls):
        """No error injection"""
        return cls(
            failure_rate=0.0,
            timeout_rate=0.0,
            processing_error_rate=0.0,
            error_types=[]
        )
    
    @classmethod
    def light(cls):
        """Light error injection for basic resilience testing"""
        return cls(
            failure_rate=0.02,
            timeout_rate=0.015,
            processing_error_rate=0.01,
            error_types=["NETWORK_ERROR", "TIMEOUT_ERROR", "PROCESSING_TIMEOUT"]
        )
    
    @classmethod
    def moderate(cls):
        """Moderate error injection for thorough testing"""
        return cls(
            failure_rate=0.08,
            timeout_rate=0.05,
            processing_error_rate=0.04,
            error_types=["NETWORK_ERROR", "TIMEOUT_ERROR", "PROCESSING_TIMEOUT", 
                        "TRANSCRIPTION_ERROR", "MEMORY_ERROR"]
        )
    
    @classmethod
    def heavy(cls):
        """Heavy error injection for stress testing"""
        return cls(
            failure_rate=0.20,
            timeout_rate=0.15,
            processing_error_rate=0.10,
            error_types=["NETWORK_ERROR", "TIMEOUT_ERROR", "PROCESSING_TIMEOUT",
                        "TRANSCRIPTION_ERROR", "MEMORY_ERROR", "CPU_OVERLOAD", 
                        "CONNECTION_DROPPED"]
        )
    
    @classmethod
    def custom(cls, failure_rate: float, timeout_rate: float, 
               processing_error_rate: float, error_types: List[str]):
        """Create custom error scenario"""
        return cls(failure_rate, timeout_rate, processing_error_rate, error_types)
    
    def should_inject_failure(self) -> bool:
        """Check if should inject connection failure"""
        return random.random() < self.failure_rate
    
    def should_inject_timeout(self) -> bool:
        """Check if should inject timeout error"""
        return random.random() < self.timeout_rate
    
    def should_inject_processing_error(self) -> bool:
        """Check if should inject processing error"""
        return random.random() < self.processing_error_rate
    
    def get_random_error_type(self) -> Optional[str]:
        """Get random error type from available types"""
        if not self.error_types:
            return None
        return random.choice(self.error_types)


# Predefined error scenarios
ErrorScenario.NONE = ErrorScenario.none()
ErrorScenario.LIGHT = ErrorScenario.light()
ErrorScenario.MODERATE = ErrorScenario.moderate()
ErrorScenario.HEAVY = ErrorScenario.heavy()


@dataclass
class ResourceConstraints:
    """Configuration for resource constraint simulation"""
    max_memory_mb: Optional[int]  # Maximum memory usage in MB
    max_cpu_percent: Optional[int]  # Maximum CPU usage percentage
    max_concurrent_connections: Optional[int]  # Maximum concurrent connections
    memory_pressure_threshold: float  # 0.0 to 1.0, when to simulate memory pressure
    cpu_throttle_threshold: float  # 0.0 to 1.0, when to throttle CPU
    
    @classmethod
    def unlimited(cls):
        """No resource constraints"""
        return cls(
            max_memory_mb=None,
            max_cpu_percent=None,
            max_concurrent_connections=None,
            memory_pressure_threshold=0.0,
            cpu_throttle_threshold=0.0
        )
    
    @classmethod
    def pi5_simulated(cls):
        """Simulate Raspberry Pi 5 resource constraints (8GB RAM, 4-core ARM)"""
        return cls(
            max_memory_mb=7680,  # 8GB - 512MB for system
            max_cpu_percent=80,   # Leave headroom for system processes
            max_concurrent_connections=10,
            memory_pressure_threshold=0.75,
            cpu_throttle_threshold=0.75
        )
    
    @classmethod
    def pi5_stressed(cls):
        """Simulate stressed Raspberry Pi 5 with limited resources"""
        return cls(
            max_memory_mb=6144,  # 6GB available (more system overhead)
            max_cpu_percent=60,   # More conservative CPU usage
            max_concurrent_connections=5,
            memory_pressure_threshold=0.55,
            cpu_throttle_threshold=0.55
        )
    
    @classmethod
    def custom(cls, max_memory_mb: Optional[int], max_cpu_percent: Optional[int],
               max_concurrent_connections: Optional[int], 
               memory_pressure_threshold: float, cpu_throttle_threshold: float):
        """Create custom resource constraints"""
        return cls(max_memory_mb, max_cpu_percent, max_concurrent_connections,
                  memory_pressure_threshold, cpu_throttle_threshold)
    
    def is_memory_exceeded(self, current_memory_mb: int) -> bool:
        """Check if memory limit is exceeded"""
        if self.max_memory_mb is None:
            return False
        return current_memory_mb > self.max_memory_mb
    
    def is_cpu_exceeded(self, current_cpu_percent: int) -> bool:
        """Check if CPU limit is exceeded"""
        if self.max_cpu_percent is None:
            return False
        return current_cpu_percent > self.max_cpu_percent
    
    def is_connection_limit_exceeded(self, current_connections: int) -> bool:
        """Check if connection limit is exceeded"""
        if self.max_concurrent_connections is None:
            return False
        return current_connections > self.max_concurrent_connections
    
    def is_under_memory_pressure(self, current_memory_mb: int) -> bool:
        """Check if under memory pressure (approaching limit)"""
        if self.max_memory_mb is None:
            return False
        usage_ratio = current_memory_mb / self.max_memory_mb
        return usage_ratio > self.memory_pressure_threshold
    
    def is_under_cpu_pressure(self, current_cpu_percent: int) -> bool:
        """Check if under CPU pressure (approaching limit)"""
        if self.max_cpu_percent is None:
            return False
        usage_ratio = current_cpu_percent / self.max_cpu_percent
        return usage_ratio > self.cpu_throttle_threshold


# Predefined resource constraints
ResourceConstraints.UNLIMITED = ResourceConstraints.unlimited()
ResourceConstraints.PI5_SIMULATED = ResourceConstraints.pi5_simulated()
ResourceConstraints.PI5_STRESSED = ResourceConstraints.pi5_stressed()


@dataclass
class MockServerConfig:
    """Complete mock server configuration"""
    latency_profile: LatencyProfile
    error_scenario: ErrorScenario
    resource_constraints: ResourceConstraints
    host: str = "0.0.0.0"
    port: int = 8765
    model_size: str = "tiny.en"
    enable_detailed_logging: bool = False
    
    @classmethod
    def default(cls):
        """Default configuration for development"""
        return cls(
            latency_profile=LatencyProfile.NORMAL,
            error_scenario=ErrorScenario.NONE,
            resource_constraints=ResourceConstraints.UNLIMITED
        )
    
    @classmethod
    def fast_development(cls):
        """Fast configuration for rapid development"""
        return cls(
            latency_profile=LatencyProfile.FAST,
            error_scenario=ErrorScenario.NONE,
            resource_constraints=ResourceConstraints.UNLIMITED,
            enable_detailed_logging=True
        )
    
    @classmethod
    def pi_simulation(cls):
        """Realistic Pi simulation configuration"""
        return cls(
            latency_profile=LatencyProfile.NORMAL,
            error_scenario=ErrorScenario.LIGHT,
            resource_constraints=ResourceConstraints.PI5_SIMULATED,
            enable_detailed_logging=False
        )
    
    @classmethod
    def stress_testing(cls):
        """Stress testing configuration"""
        return cls(
            latency_profile=LatencyProfile.SLOW,
            error_scenario=ErrorScenario.HEAVY,
            resource_constraints=ResourceConstraints.PI5_STRESSED,
            enable_detailed_logging=True
        )


class ConfigurationManager:
    """Manager for loading configuration from various sources"""
    
    @staticmethod
    def load_from_environment() -> MockServerConfig:
        """Load configuration from environment variables"""
        # Latency profile
        latency_profile_name = os.getenv("MOCK_LATENCY_PROFILE", "normal").lower()
        latency_profile = ConfigurationManager._get_latency_profile(latency_profile_name)
        
        # Error scenario
        error_scenario_name = os.getenv("MOCK_ERROR_SCENARIO", "none").lower()
        error_scenario = ConfigurationManager._get_error_scenario(error_scenario_name)
        
        # Resource constraints
        resource_constraints_name = os.getenv("MOCK_RESOURCE_CONSTRAINTS", "unlimited").lower()
        resource_constraints = ConfigurationManager._get_resource_constraints(resource_constraints_name)
        
        # Basic server settings
        host = os.getenv("MOCK_HOST", "0.0.0.0")
        try:
            port = int(os.getenv("MOCK_PORT", "8765"))
        except ValueError:
            port = 8765
        model_size = os.getenv("MOCK_MODEL_SIZE", "tiny.en")
        enable_detailed_logging = os.getenv("MOCK_ENABLE_DETAILED_LOGGING", "false").lower() == "true"
        
        return MockServerConfig(
            latency_profile=latency_profile,
            error_scenario=error_scenario,
            resource_constraints=resource_constraints,
            host=host,
            port=port,
            model_size=model_size,
            enable_detailed_logging=enable_detailed_logging
        )
    
    @staticmethod
    def load_from_file(config_file_path: str) -> MockServerConfig:
        """Load configuration from JSON file"""
        config_path = Path(config_file_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Parse latency profile
        latency_profile_name = config_data.get("latency_profile", "normal").lower()
        if latency_profile_name == "custom" and "custom_latency" in config_data:
            custom_latency = config_data["custom_latency"]
            latency_profile = LatencyProfile.custom(
                processing_delay_ms=custom_latency["processing_delay_ms"],
                network_delay_ms=custom_latency["network_delay_ms"],
                variability_factor=custom_latency["variability_factor"]
            )
        else:
            latency_profile = ConfigurationManager._get_latency_profile(latency_profile_name)
        
        # Parse error scenario
        error_scenario_name = config_data.get("error_scenario", "none").lower()
        if error_scenario_name == "custom" and "custom_errors" in config_data:
            custom_errors = config_data["custom_errors"]
            error_scenario = ErrorScenario.custom(
                failure_rate=custom_errors["failure_rate"],
                timeout_rate=custom_errors["timeout_rate"],
                processing_error_rate=custom_errors["processing_error_rate"],
                error_types=custom_errors["error_types"]
            )
        else:
            error_scenario = ConfigurationManager._get_error_scenario(error_scenario_name)
        
        # Parse resource constraints
        resource_constraints_name = config_data.get("resource_constraints", "unlimited").lower()
        if resource_constraints_name == "custom" and "custom_resources" in config_data:
            custom_resources = config_data["custom_resources"]
            resource_constraints = ResourceConstraints.custom(
                max_memory_mb=custom_resources.get("max_memory_mb"),
                max_cpu_percent=custom_resources.get("max_cpu_percent"),
                max_concurrent_connections=custom_resources.get("max_concurrent_connections"),
                memory_pressure_threshold=custom_resources["memory_pressure_threshold"],
                cpu_throttle_threshold=custom_resources["cpu_throttle_threshold"]
            )
        else:
            resource_constraints = ConfigurationManager._get_resource_constraints(resource_constraints_name)
        
        # Basic server settings
        host = config_data.get("host", "0.0.0.0")
        port = int(config_data.get("port", 8765))
        model_size = config_data.get("model_size", "tiny.en")
        enable_detailed_logging = config_data.get("enable_detailed_logging", False)
        
        return MockServerConfig(
            latency_profile=latency_profile,
            error_scenario=error_scenario,
            resource_constraints=resource_constraints,
            host=host,
            port=port,
            model_size=model_size,
            enable_detailed_logging=enable_detailed_logging
        )
    
    @staticmethod
    def _get_latency_profile(name: str) -> LatencyProfile:
        """Get latency profile by name"""
        profiles = {
            "fast": LatencyProfile.FAST,
            "normal": LatencyProfile.NORMAL,
            "slow": LatencyProfile.SLOW
        }
        return profiles.get(name, LatencyProfile.NORMAL)
    
    @staticmethod
    def _get_error_scenario(name: str) -> ErrorScenario:
        """Get error scenario by name"""
        scenarios = {
            "none": ErrorScenario.NONE,
            "light": ErrorScenario.LIGHT,
            "moderate": ErrorScenario.MODERATE,
            "heavy": ErrorScenario.HEAVY
        }
        return scenarios.get(name, ErrorScenario.NONE)
    
    @staticmethod
    def _get_resource_constraints(name: str) -> ResourceConstraints:
        """Get resource constraints by name"""
        constraints = {
            "unlimited": ResourceConstraints.UNLIMITED,
            "pi5_simulated": ResourceConstraints.PI5_SIMULATED,
            "pi5_stressed": ResourceConstraints.PI5_STRESSED
        }
        return constraints.get(name, ResourceConstraints.UNLIMITED) 