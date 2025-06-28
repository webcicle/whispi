"""
Test suite for Mock Server Configuration System (Task 2.4)

This test suite validates the configuration system for mock scenarios including:
- Different latency profiles (fast/normal/slow)
- Error injection scenarios (network timeouts, processing failures)
- Resource constraint simulation (memory limits, CPU throttling)
- Concurrent connection limits
- Environment variable and config file support
"""

import pytest
import os
import tempfile
import json
from unittest.mock import patch

from server.mock_configuration import (
    MockServerConfig,
    LatencyProfile,
    ErrorScenario,
    ResourceConstraints,
    ConfigurationManager
)


class TestLatencyProfile:
    """Test latency profile configuration"""
    
    def test_fast_latency_profile(self):
        """Test fast latency profile configuration"""
        profile = LatencyProfile.FAST
        assert profile.processing_delay_ms == 50
        assert profile.network_delay_ms == 10
        assert profile.variability_factor == 0.2
    
    def test_normal_latency_profile(self):
        """Test normal latency profile configuration"""
        profile = LatencyProfile.NORMAL
        assert profile.processing_delay_ms == 200
        assert profile.network_delay_ms == 30
        assert profile.variability_factor == 0.3
    
    def test_slow_latency_profile(self):
        """Test slow latency profile configuration"""
        profile = LatencyProfile.SLOW
        assert profile.processing_delay_ms == 800
        assert profile.network_delay_ms == 100
        assert profile.variability_factor == 0.5
    
    def test_custom_latency_profile(self):
        """Test custom latency profile creation"""
        custom = LatencyProfile.custom(
            processing_delay_ms=300,
            network_delay_ms=30,
            variability_factor=0.4
        )
        assert custom.processing_delay_ms == 300
        assert custom.network_delay_ms == 30
        assert custom.variability_factor == 0.4
    
    def test_latency_calculation_with_variability(self):
        """Test latency calculation with variability factor"""
        profile = LatencyProfile.NORMAL
        
        # Test multiple calculations to ensure variability
        latencies = [profile.calculate_actual_latency() for _ in range(100)]
        
        # Should have some variation
        assert len(set(latencies)) > 1
        
        # All values should be positive
        assert all(latency >= 0 for latency in latencies)
        
        # Should be within reasonable bounds considering variability
        base_total = profile.processing_delay_ms + profile.network_delay_ms
        max_expected = base_total * (1 + profile.variability_factor)
        assert all(latency <= max_expected for latency in latencies)
    
    def test_latency_calculation_no_variability(self):
        """Test latency calculation with zero variability"""
        profile = LatencyProfile.custom(100, 20, 0.0)
        latency = profile.calculate_actual_latency()
        assert latency == 120  # No variation


class TestErrorScenario:
    """Test error scenario configuration"""
    
    def test_none_error_scenario(self):
        """Test no error injection scenario"""
        scenario = ErrorScenario.NONE
        assert scenario.failure_rate == 0.0
        assert scenario.timeout_rate == 0.0
        assert scenario.processing_error_rate == 0.0
        assert len(scenario.error_types) == 0
    
    def test_light_error_scenario(self):
        """Test light error injection scenario"""
        scenario = ErrorScenario.LIGHT
        assert scenario.failure_rate == 0.02
        assert scenario.timeout_rate == 0.015
        assert scenario.processing_error_rate == 0.01
        assert len(scenario.error_types) > 0
        assert "NETWORK_ERROR" in scenario.error_types
    
    def test_moderate_error_scenario(self):
        """Test moderate error injection scenario"""
        scenario = ErrorScenario.MODERATE
        assert scenario.failure_rate == 0.08
        assert scenario.timeout_rate == 0.05
        assert scenario.processing_error_rate == 0.04
        assert len(scenario.error_types) > 0
        assert "TRANSCRIPTION_ERROR" in scenario.error_types
    
    def test_heavy_error_scenario(self):
        """Test heavy error injection scenario"""
        scenario = ErrorScenario.HEAVY
        assert scenario.failure_rate == 0.20
        assert scenario.timeout_rate == 0.15
        assert scenario.processing_error_rate == 0.10
        assert len(scenario.error_types) > 0
        assert "CPU_OVERLOAD" in scenario.error_types
    
    def test_custom_error_scenario(self):
        """Test custom error scenario creation"""
        custom = ErrorScenario.custom(
            failure_rate=0.12,
            timeout_rate=0.08,
            processing_error_rate=0.05,
            error_types=["CUSTOM_ERROR", "TEST_ERROR"]
        )
        assert custom.failure_rate == 0.12
        assert custom.timeout_rate == 0.08
        assert custom.processing_error_rate == 0.05
        assert custom.error_types == ["CUSTOM_ERROR", "TEST_ERROR"]
    
    def test_should_inject_error_probability(self):
        """Test error injection probability calculation"""
        scenario = ErrorScenario.MODERATE
        
        # Test multiple calls to check probability distribution
        failure_results = [scenario.should_inject_failure() for _ in range(1000)]
        timeout_results = [scenario.should_inject_timeout() for _ in range(1000)]
        processing_results = [scenario.should_inject_processing_error() for _ in range(1000)]
        
        # Should have some True results based on probability
        failure_rate = sum(failure_results) / len(failure_results)
        timeout_rate = sum(timeout_results) / len(timeout_results)
        processing_rate = sum(processing_results) / len(processing_results)
        
        # Allow for some variance in random testing (within 50% of expected)
        assert 0.04 <= failure_rate <= 0.12   # Around MODERATE.failure_rate (0.08)
        assert 0.025 <= timeout_rate <= 0.075  # Around MODERATE.timeout_rate (0.05)
        assert 0.02 <= processing_rate <= 0.06  # Around MODERATE.processing_error_rate (0.04)
    
    def test_get_random_error_type(self):
        """Test getting random error type"""
        scenario = ErrorScenario.LIGHT
        error_type = scenario.get_random_error_type()
        assert error_type in scenario.error_types
        
        # Test empty error types
        empty_scenario = ErrorScenario.NONE
        assert empty_scenario.get_random_error_type() is None


class TestResourceConstraints:
    """Test resource constraint configuration"""
    
    def test_unlimited_resources(self):
        """Test unlimited resource configuration"""
        constraints = ResourceConstraints.UNLIMITED
        assert constraints.max_memory_mb is None
        assert constraints.max_cpu_percent is None
        assert constraints.max_concurrent_connections is None
        assert constraints.memory_pressure_threshold == 0.0
        assert constraints.cpu_throttle_threshold == 0.0
    
    def test_pi5_simulated_resources(self):
        """Test Pi 5 simulated resource constraints"""
        constraints = ResourceConstraints.PI5_SIMULATED
        assert constraints.max_memory_mb == 7680  # 8GB - 512MB for system
        assert constraints.max_cpu_percent == 80   # Leave headroom for system
        assert constraints.max_concurrent_connections == 10
        assert constraints.memory_pressure_threshold == 0.75
        assert constraints.cpu_throttle_threshold == 0.75
    
    def test_pi5_stressed_resources(self):
        """Test Pi 5 stressed resource constraints"""
        constraints = ResourceConstraints.PI5_STRESSED
        assert constraints.max_memory_mb == 6144  # 6GB available
        assert constraints.max_cpu_percent == 60   # More conservative
        assert constraints.max_concurrent_connections == 5
        assert constraints.memory_pressure_threshold == 0.55
        assert constraints.cpu_throttle_threshold == 0.55
    
    def test_custom_resource_constraints(self):
        """Test custom resource constraint creation"""
        custom = ResourceConstraints.custom(
            max_memory_mb=4096,
            max_cpu_percent=50,
            max_concurrent_connections=3,
            memory_pressure_threshold=0.75,
            cpu_throttle_threshold=0.65
        )
        assert custom.max_memory_mb == 4096
        assert custom.max_cpu_percent == 50
        assert custom.max_concurrent_connections == 3
        assert custom.memory_pressure_threshold == 0.75
        assert custom.cpu_throttle_threshold == 0.65
    
    def test_resource_limit_checks(self):
        """Test resource limit checking methods"""
        constraints = ResourceConstraints.PI5_SIMULATED
        
        # Test memory checks
        assert not constraints.is_memory_exceeded(1000)  # 1GB is fine
        assert constraints.is_memory_exceeded(8000)      # 8GB exceeds limit
        
        # Test CPU checks
        assert not constraints.is_cpu_exceeded(50)       # 50% is fine
        assert constraints.is_cpu_exceeded(90)           # 90% exceeds limit
        
        # Test connection checks
        assert not constraints.is_connection_limit_exceeded(5)   # 5 connections fine
        assert constraints.is_connection_limit_exceeded(15)     # 15 exceeds limit
    
    def test_pressure_threshold_checks(self):
        """Test memory and CPU pressure threshold detection"""
        constraints = ResourceConstraints.PI5_SIMULATED
        
        # Test memory pressure
        safe_memory = int(constraints.max_memory_mb * 0.5)      # 50% is safe
        pressure_memory = int(constraints.max_memory_mb * 0.85) # 85% triggers pressure
        
        assert not constraints.is_under_memory_pressure(safe_memory)
        assert constraints.is_under_memory_pressure(pressure_memory)
        
        # Test CPU pressure
        safe_cpu = int(constraints.max_cpu_percent * 0.5)      # 50% is safe
        pressure_cpu = int(constraints.max_cpu_percent * 0.85) # 85% triggers pressure
        
        assert not constraints.is_under_cpu_pressure(safe_cpu)
        assert constraints.is_under_cpu_pressure(pressure_cpu)
    
    def test_unlimited_constraints_never_exceeded(self):
        """Test that unlimited constraints are never exceeded"""
        constraints = ResourceConstraints.UNLIMITED
        
        # Should never be exceeded with unlimited constraints
        assert not constraints.is_memory_exceeded(999999)
        assert not constraints.is_cpu_exceeded(999)
        assert not constraints.is_connection_limit_exceeded(999)
        assert not constraints.is_under_memory_pressure(999999)
        assert not constraints.is_under_cpu_pressure(999)


class TestMockServerConfig:
    """Test complete mock server configuration"""
    
    def test_default_configuration(self):
        """Test default configuration creation"""
        config = MockServerConfig.default()
        
        assert config.latency_profile == LatencyProfile.NORMAL
        assert config.error_scenario == ErrorScenario.NONE
        assert config.resource_constraints == ResourceConstraints.UNLIMITED
        assert config.host == "0.0.0.0"
        assert config.port == 8765
        assert config.model_size == "tiny.en"
        assert config.enable_detailed_logging is False
    
    def test_custom_configuration(self):
        """Test custom configuration creation"""
        config = MockServerConfig(
            latency_profile=LatencyProfile.FAST,
            error_scenario=ErrorScenario.LIGHT,
            resource_constraints=ResourceConstraints.PI5_SIMULATED,
            host="localhost",
            port=9999,
            model_size="small.en",
            enable_detailed_logging=True
        )
        
        assert config.latency_profile == LatencyProfile.FAST
        assert config.error_scenario == ErrorScenario.LIGHT
        assert config.resource_constraints == ResourceConstraints.PI5_SIMULATED
        assert config.host == "localhost"
        assert config.port == 9999
        assert config.model_size == "small.en"
        assert config.enable_detailed_logging is True
    
    def test_fast_development_scenario(self):
        """Test fast development scenario"""
        config = MockServerConfig.fast_development()
        assert config.latency_profile == LatencyProfile.FAST
        assert config.error_scenario == ErrorScenario.NONE
        assert config.resource_constraints == ResourceConstraints.UNLIMITED
        assert config.enable_detailed_logging is True
    
    def test_pi_simulation_scenario(self):
        """Test realistic Pi simulation scenario"""
        config = MockServerConfig.pi_simulation()
        assert config.latency_profile == LatencyProfile.NORMAL
        assert config.error_scenario == ErrorScenario.LIGHT
        assert config.resource_constraints == ResourceConstraints.PI5_SIMULATED
        assert config.enable_detailed_logging is False
    
    def test_stress_testing_scenario(self):
        """Test stress testing scenario"""
        config = MockServerConfig.stress_testing()
        assert config.latency_profile == LatencyProfile.SLOW
        assert config.error_scenario == ErrorScenario.HEAVY
        assert config.resource_constraints == ResourceConstraints.PI5_STRESSED
        assert config.enable_detailed_logging is True


class TestConfigurationManager:
    """Test configuration manager for loading from environment and files"""
    
    def test_load_from_environment_variables(self):
        """Test loading configuration from environment variables"""
        env_vars = {
            "MOCK_LATENCY_PROFILE": "fast",
            "MOCK_ERROR_SCENARIO": "moderate", 
            "MOCK_RESOURCE_CONSTRAINTS": "pi5_simulated",
            "MOCK_HOST": "127.0.0.1",
            "MOCK_PORT": "8888",
            "MOCK_MODEL_SIZE": "small.en",
            "MOCK_ENABLE_DETAILED_LOGGING": "true"
        }
        
        with patch.dict(os.environ, env_vars):
            config = ConfigurationManager.load_from_environment()
            
            assert config.latency_profile == LatencyProfile.FAST
            assert config.error_scenario == ErrorScenario.MODERATE
            assert config.resource_constraints == ResourceConstraints.PI5_SIMULATED
            assert config.host == "127.0.0.1"
            assert config.port == 8888
            assert config.model_size == "small.en"
            assert config.enable_detailed_logging is True
    
    def test_load_from_config_file(self):
        """Test loading configuration from JSON config file"""
        config_data = {
            "latency_profile": "slow",
            "error_scenario": "heavy",
            "resource_constraints": "pi5_stressed",
            "host": "192.168.1.100",
            "port": 7777,
            "model_size": "medium.en",
            "enable_detailed_logging": False
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file_path = f.name
        
        try:
            config = ConfigurationManager.load_from_file(config_file_path)
            
            assert config.latency_profile == LatencyProfile.SLOW
            assert config.error_scenario == ErrorScenario.HEAVY
            assert config.resource_constraints == ResourceConstraints.PI5_STRESSED
            assert config.host == "192.168.1.100"
            assert config.port == 7777
            assert config.model_size == "medium.en"
            assert config.enable_detailed_logging is False
        finally:
            os.unlink(config_file_path)
    
    def test_load_with_custom_profiles_from_file(self):
        """Test loading configuration with custom profiles from file"""
        config_data = {
            "latency_profile": "custom",
            "custom_latency": {
                "processing_delay_ms": 150,
                "network_delay_ms": 25,
                "variability_factor": 0.2
            },
            "error_scenario": "custom",
            "custom_errors": {
                "failure_rate": 0.06,
                "timeout_rate": 0.04,
                "processing_error_rate": 0.02,
                "error_types": ["TEST_ERROR"]
            },
            "resource_constraints": "custom",
            "custom_resources": {
                "max_memory_mb": 2048,
                "max_cpu_percent": 70,
                "max_concurrent_connections": 8,
                "memory_pressure_threshold": 0.8,
                "cpu_throttle_threshold": 0.75
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file_path = f.name
        
        try:
            config = ConfigurationManager.load_from_file(config_file_path)
            
            # Verify custom latency profile
            assert config.latency_profile.processing_delay_ms == 150
            assert config.latency_profile.network_delay_ms == 25
            assert config.latency_profile.variability_factor == 0.2
            
            # Verify custom error scenario
            assert config.error_scenario.failure_rate == 0.06
            assert config.error_scenario.timeout_rate == 0.04
            assert config.error_scenario.processing_error_rate == 0.02
            assert config.error_scenario.error_types == ["TEST_ERROR"]
            
            # Verify custom resource constraints
            assert config.resource_constraints.max_memory_mb == 2048
            assert config.resource_constraints.max_cpu_percent == 70
            assert config.resource_constraints.max_concurrent_connections == 8
            assert config.resource_constraints.memory_pressure_threshold == 0.8
            assert config.resource_constraints.cpu_throttle_threshold == 0.75
        finally:
            os.unlink(config_file_path)
    
    def test_environment_variables_override_defaults(self):
        """Test that environment variables override default values"""
        env_vars = {
            "MOCK_LATENCY_PROFILE": "slow",
            "MOCK_PORT": "9999"
        }
        
        with patch.dict(os.environ, env_vars):
            config = ConfigurationManager.load_from_environment()
            
            # Environment variables should override defaults
            assert config.latency_profile == LatencyProfile.SLOW
            assert config.port == 9999
            
            # Non-specified variables should use defaults
            assert config.error_scenario == ErrorScenario.NONE
            assert config.resource_constraints == ResourceConstraints.UNLIMITED
            assert config.host == "0.0.0.0"
    
    def test_invalid_environment_values_use_defaults(self):
        """Test that invalid environment values fall back to defaults"""
        env_vars = {
            "MOCK_LATENCY_PROFILE": "invalid_profile",
            "MOCK_PORT": "not_a_number",
            "MOCK_ENABLE_DETAILED_LOGGING": "not_a_boolean"
        }
        
        with patch.dict(os.environ, env_vars):
            config = ConfigurationManager.load_from_environment()
            
            # Invalid values should fall back to defaults
            assert config.latency_profile == LatencyProfile.NORMAL
            assert config.port == 8765
            assert config.enable_detailed_logging is False
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading from nonexistent file raises appropriate error"""
        with pytest.raises(FileNotFoundError):
            ConfigurationManager.load_from_file("/nonexistent/config.json")
    
    def test_load_invalid_json_file_raises_error(self):
        """Test that loading invalid JSON file raises appropriate error"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            invalid_file_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                ConfigurationManager.load_from_file(invalid_file_path)
        finally:
            os.unlink(invalid_file_path)


if __name__ == "__main__":
    pytest.main([__file__]) 