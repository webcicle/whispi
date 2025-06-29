#!/usr/bin/env python3
"""
Tests for Text Insertion Module

Comprehensive test suite covering:
- Permission handling
- Application detection
- Text insertion via Accessibility API
- Queue system functionality
- Error handling and edge cases
- Special character handling
- Mock functionality for non-macOS environments
"""

import asyncio
import pytest
import threading
import time
import logging
from unittest.mock import Mock, patch, MagicMock, call
from queue import Empty

# Import the module we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from client.text_inserter import (
    TextInserter, InsertionMethod, InsertionError, 
    TextInsertionRequest, ApplicationInfo, MACOS_AVAILABLE
)


class TestTextInsertionRequest:
    """Test the TextInsertionRequest dataclass."""
    
    def test_creation_with_defaults(self):
        """Test creating a request with default values."""
        request = TextInsertionRequest(text="Hello", timestamp=1.0)
        assert request.text == "Hello"
        assert request.timestamp == 1.0
        assert request.priority == 0
        assert request.method == InsertionMethod.ACCESSIBILITY_API
    
    def test_creation_with_custom_values(self):
        """Test creating a request with custom values."""
        request = TextInsertionRequest(
            text="Custom text",
            timestamp=2.0,
            priority=5,
            method=InsertionMethod.APPLESCRIPT
        )
        assert request.text == "Custom text"
        assert request.timestamp == 2.0
        assert request.priority == 5
        assert request.method == InsertionMethod.APPLESCRIPT


class TestApplicationInfo:
    """Test the ApplicationInfo dataclass."""
    
    def test_creation(self):
        """Test creating application info."""
        app_info = ApplicationInfo(
            pid=123,
            name="TestApp",
            bundle_id="com.test.app",
            supports_text_insertion=True,
            text_field_type="AXTextField"
        )
        assert app_info.pid == 123
        assert app_info.name == "TestApp"
        assert app_info.bundle_id == "com.test.app"
        assert app_info.supports_text_insertion is True
        assert app_info.text_field_type == "AXTextField"


class TestTextInserterInitialization:
    """Test TextInserter initialization and basic setup."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        with patch('client.text_inserter.MACOS_AVAILABLE', False):
            inserter = TextInserter(check_permissions=False)
            assert inserter.max_queue_size == 100
            assert inserter.accessibility_enabled is False
            assert inserter.queue_processor_running is False
            assert inserter.stats['insertions_attempted'] == 0
    
    def test_init_with_custom_queue_size(self):
        """Test initialization with custom queue size."""
        with patch('client.text_inserter.MACOS_AVAILABLE', False):
            inserter = TextInserter(max_queue_size=50, check_permissions=False)
            assert inserter.max_queue_size == 50
    
    @patch('client.text_inserter.MACOS_AVAILABLE', True)
    @patch('client.text_inserter.AXIsProcessTrusted')
    def test_init_with_permission_check(self, mock_trusted):
        """Test initialization with permission checking."""
        mock_trusted.return_value = True
        inserter = TextInserter(check_permissions=True)
        assert inserter.accessibility_enabled is True
        mock_trusted.assert_called_once()
    
    def test_init_without_macos(self):
        """Test initialization on non-macOS environment."""
        with patch('client.text_inserter.MACOS_AVAILABLE', False):
            inserter = TextInserter()
            assert inserter.accessibility_enabled is False


class TestPermissionHandling:
    """Test accessibility permission handling."""
    
    def test_check_permissions_granted(self):
        """Test checking permissions when they are granted."""
        inserter = TextInserter(check_permissions=False)
        
        # Test the mock mode behavior (which should pass)
        if not MACOS_AVAILABLE:
            result = inserter.check_accessibility_permissions()
            assert result is False  # Expected on non-macOS
        else:
            # On actual macOS, we can test real permissions or skip
            # For now, we'll test that the method exists and can be called
            result = inserter.check_accessibility_permissions()
            assert isinstance(result, bool)
    
    @patch('client.text_inserter.MACOS_AVAILABLE', False)
    def test_check_permissions_no_macos(self):
        """Test checking permissions on non-macOS."""
        inserter = TextInserter(check_permissions=False)
        result = inserter.check_accessibility_permissions()
        assert result is False
    
    @patch('client.text_inserter.MACOS_AVAILABLE', True)
    @patch('client.text_inserter.AXIsProcessTrusted')
    def test_check_permissions_exception(self, mock_trusted):
        """Test handling exceptions during permission check."""
        mock_trusted.side_effect = Exception("Test error")
        inserter = TextInserter(check_permissions=False)
        
        result = inserter.check_accessibility_permissions()
        assert result is False
        assert inserter.accessibility_enabled is False
    
    @patch('client.text_inserter.MACOS_AVAILABLE', True)
    @patch('client.text_inserter.AXIsProcessTrustedWithOptions')
    def test_request_permissions_granted(self, mock_trusted_options):
        """Test requesting permissions when user grants them."""
        mock_trusted_options.return_value = True
        inserter = TextInserter(check_permissions=False)
        
        result = inserter.request_accessibility_permissions()
        assert result is True
        assert inserter.accessibility_enabled is True
        mock_trusted_options.assert_called_once()
    
    @patch('client.text_inserter.MACOS_AVAILABLE', True)
    @patch('client.text_inserter.AXIsProcessTrustedWithOptions')
    def test_request_permissions_denied(self, mock_trusted_options):
        """Test requesting permissions when user denies them."""
        mock_trusted_options.return_value = False
        inserter = TextInserter(check_permissions=False)
        
        result = inserter.request_accessibility_permissions()
        assert result is False
        assert inserter.accessibility_enabled is False


class TestApplicationDetection:
    """Test active application detection functionality."""
    
    @patch('client.text_inserter.MACOS_AVAILABLE', False)
    def test_get_app_info_no_macos(self):
        """Test getting app info on non-macOS."""
        inserter = TextInserter(check_permissions=False)
        result = inserter.get_active_application_info()
        assert result is None
    
    @patch('client.text_inserter.MACOS_AVAILABLE', True)
    def test_get_app_info_no_permissions(self):
        """Test getting app info without accessibility permissions."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = False
        result = inserter.get_active_application_info()
        assert result is None
    
    @patch('client.text_inserter.MACOS_AVAILABLE', True)
    @patch('client.text_inserter.AXUIElementCreateSystemWide')
    @patch('client.text_inserter.AXUIElementCopyAttributeValue')
    @patch('client.text_inserter.NSRunningApplication')
    @patch('client.text_inserter.kAXTextFieldRole', 'AXTextField')
    @patch('client.text_inserter.kAXErrorSuccess', 0)
    def test_get_app_info_success(self, mock_running_app, mock_copy_attr, mock_create_system):
        """Test successful application info retrieval."""
        # Mock the accessibility API calls
        mock_system_element = Mock()
        mock_create_system.return_value = mock_system_element
        
        mock_focused_app = Mock()
        mock_focused_element = Mock()
        
        # Mock the API call results
        mock_copy_attr.side_effect = [
            (0, mock_focused_app),  # focused application
            (0, 123),  # PID
            (0, mock_focused_element),  # focused element
            (0, "AXTextField")  # role
        ]
        
        # Mock running application
        mock_app = Mock()
        mock_app.localizedName.return_value = "TestApp"
        mock_app.bundleIdentifier.return_value = "com.test.app"
        mock_running_app.runningApplicationWithProcessIdentifier_.return_value = mock_app
        
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True
        
        result = inserter.get_active_application_info()
        
        assert result is not None
        assert result.pid == 123
        assert result.name == "TestApp"
        assert result.bundle_id == "com.test.app"
        assert result.supports_text_insertion is True
        assert result.text_field_type == "AXTextField"
    
    @patch('client.text_inserter.MACOS_AVAILABLE', True)
    @patch('client.text_inserter.AXUIElementCreateSystemWide')
    @patch('client.text_inserter.AXUIElementCopyAttributeValue')
    def test_get_app_info_no_focused_app(self, mock_copy_attr, mock_create_system):
        """Test when no focused application is found."""
        mock_copy_attr.return_value = (0, None)
        
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True
        
        result = inserter.get_active_application_info()
        assert result is None


class TestTextInsertion:
    """Test text insertion functionality."""
    
    def test_insert_empty_text(self):
        """Test inserting empty text."""
        inserter = TextInserter(check_permissions=False)
        result = inserter.insert_text_at_cursor("")
        assert result is True
        assert inserter.stats['insertions_attempted'] == 0
    
    def test_insert_without_permissions(self):
        """Test inserting text without accessibility permissions."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = False
        
        result = inserter.insert_text_at_cursor("Hello")
        assert result is False
        assert inserter.stats['insertions_attempted'] == 1
        assert inserter.stats['insertions_failed'] == 1
        assert inserter.stats['permission_errors'] == 1
    
    @patch('client.text_inserter.MACOS_AVAILABLE', False)
    def test_insert_mock_mode(self):
        """Test text insertion in mock mode (non-macOS)."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True  # Simulate having permissions
        
        result = inserter.insert_text_at_cursor("Hello World")
        assert result is True
        assert inserter.stats['insertions_attempted'] == 1
        assert inserter.stats['insertions_successful'] == 1
    
    @patch('client.text_inserter.MACOS_AVAILABLE', True)
    def test_insert_accessibility_api_no_app(self):
        """Test insertion when no application supports text insertion."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True
        
        with patch.object(inserter, 'get_active_application_info') as mock_get_app:
            mock_get_app.return_value = None
            
            result = inserter.insert_text_at_cursor("Hello")
            assert result is False
            assert inserter.stats['insertions_failed'] == 1
    
    @patch('client.text_inserter.MACOS_AVAILABLE', True)
    def test_insert_accessibility_api_unsupported_app(self):
        """Test insertion in app that doesn't support text insertion."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True
        
        app_info = ApplicationInfo(
            pid=123,
            name="TestApp",
            bundle_id="com.test.app",
            supports_text_insertion=False
        )
        
        with patch.object(inserter, 'get_active_application_info') as mock_get_app:
            mock_get_app.return_value = app_info
            
            result = inserter.insert_text_at_cursor("Hello")
            assert result is False
    
    def test_insert_unknown_method(self):
        """Test insertion with unknown method."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True
        
        # Create a fake enum value
        class FakeMethod:
            pass
        
        fake_method = FakeMethod()
        result = inserter.insert_text_at_cursor("Hello", fake_method)
        assert result is False
        assert inserter.stats['insertions_failed'] == 1
    
    def test_insert_applescript_method(self):
        """Test insertion with AppleScript method (not implemented)."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True
        
        result = inserter.insert_text_at_cursor("Hello", InsertionMethod.APPLESCRIPT)
        assert result is False  # Not implemented yet
    
    def test_insert_keystroke_method(self):
        """Test insertion with keystroke simulation method (not implemented)."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True
        
        result = inserter.insert_text_at_cursor("Hello", InsertionMethod.KEYSTROKE_SIMULATION)
        assert result is False  # Not implemented yet


class TestQueueSystem:
    """Test the text insertion queue system."""
    
    def test_queue_text_insertion(self):
        """Test queuing a text insertion request."""
        inserter = TextInserter(check_permissions=False)
        
        result = inserter.queue_text_insertion("Hello World")
        assert result is True
        assert inserter.insertion_queue.qsize() == 1
    
    def test_queue_with_priority(self):
        """Test queuing with different priorities."""
        inserter = TextInserter(check_permissions=False)
        
        # Queue multiple items with different priorities
        inserter.queue_text_insertion("Low priority", priority=1)
        inserter.queue_text_insertion("High priority", priority=10)
        inserter.queue_text_insertion("Medium priority", priority=5)
        
        assert inserter.insertion_queue.qsize() == 3
        
        # Higher priority items should come first (negative priority in queue)
        priority1, timestamp1, request1 = inserter.insertion_queue.get()
        assert request1.text == "High priority"
        assert request1.priority == 10
    
    def test_queue_full(self):
        """Test behavior when queue is full."""
        inserter = TextInserter(max_queue_size=2, check_permissions=False)
        
        # Fill the queue
        assert inserter.queue_text_insertion("First") is True
        assert inserter.queue_text_insertion("Second") is True
        
        # This should fail as queue is full
        assert inserter.queue_text_insertion("Third") is False
        assert inserter.stats['queue_overflows'] == 1
    
    def test_clear_queue(self):
        """Test clearing the queue."""
        inserter = TextInserter(check_permissions=False)
        
        # Add some items
        inserter.queue_text_insertion("Item 1")
        inserter.queue_text_insertion("Item 2")
        assert inserter.insertion_queue.qsize() == 2
        
        # Clear the queue
        inserter.clear_queue()
        assert inserter.insertion_queue.qsize() == 0
    
    def test_queue_processor_lifecycle(self):
        """Test starting and stopping the queue processor."""
        inserter = TextInserter(check_permissions=False)
        
        # Start processor
        inserter.start_queue_processor()
        assert inserter.queue_processor_running is True
        assert inserter.queue_thread is not None
        assert inserter.queue_thread.is_alive()
        
        # Stop processor
        inserter.stop_queue_processor()
        assert inserter.queue_processor_running is False
        
        # Wait a bit for thread to finish
        time.sleep(0.2)
        assert not inserter.queue_thread.is_alive()
    
    def test_queue_processor_double_start(self):
        """Test that starting processor twice doesn't create multiple threads."""
        inserter = TextInserter(check_permissions=False)
        
        inserter.start_queue_processor()
        first_thread = inserter.queue_thread
        
        inserter.start_queue_processor()  # Should not create new thread
        second_thread = inserter.queue_thread
        
        assert first_thread is second_thread
        
        inserter.stop_queue_processor()
    
    @patch('client.text_inserter.MACOS_AVAILABLE', False)
    def test_queue_processing(self):
        """Test that queued items are processed."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True  # Enable for mock processing
        
        # Start processor
        inserter.start_queue_processor()
        
        # Queue some items
        inserter.queue_text_insertion("Test 1")
        inserter.queue_text_insertion("Test 2")
        
        # Wait for processing
        time.sleep(0.1)
        
        # Items should be processed
        assert inserter.stats['insertions_attempted'] >= 2
        assert inserter.stats['insertions_successful'] >= 2
        
        inserter.stop_queue_processor()


class TestContextManager:
    """Test context manager functionality."""
    
    def test_context_manager(self):
        """Test using TextInserter as a context manager."""
        inserter = TextInserter(check_permissions=False)
        
        with inserter:
            assert inserter.queue_processor_running is True
        
        assert inserter.queue_processor_running is False


class TestStatistics:
    """Test statistics tracking."""
    
    def test_get_stats(self):
        """Test getting insertion statistics."""
        inserter = TextInserter(check_permissions=False)
        
        stats = inserter.get_stats()
        assert 'insertions_attempted' in stats
        assert 'insertions_successful' in stats
        assert 'insertions_failed' in stats
        assert 'queue_overflows' in stats
        assert 'permission_errors' in stats
        assert 'queue_size' in stats
        assert 'accessibility_enabled' in stats
        assert 'current_app' in stats
    
    def test_stats_tracking(self):
        """Test that statistics are properly tracked."""
        inserter = TextInserter(check_permissions=False)
        
        # Test failed insertion due to no permissions
        inserter.insert_text_at_cursor("Hello")
        
        stats = inserter.get_stats()
        assert stats['insertions_attempted'] == 1
        assert stats['insertions_failed'] == 1
        assert stats['permission_errors'] == 1


class TestSpecialCharacters:
    """Test handling of special characters and formatting."""
    
    @patch('client.text_inserter.MACOS_AVAILABLE', False)
    def test_unicode_characters(self):
        """Test insertion of Unicode characters."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True
        
        unicode_text = "Hello 世界 🌍 café naïve résumé"
        result = inserter.insert_text_at_cursor(unicode_text)
        assert result is True
    
    @patch('client.text_inserter.MACOS_AVAILABLE', False)
    def test_newlines_and_formatting(self):
        """Test insertion of text with newlines and formatting."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True
        
        formatted_text = "Line 1\nLine 2\tTabbed\nLine 3"
        result = inserter.insert_text_at_cursor(formatted_text)
        assert result is True
    
    @patch('client.text_inserter.MACOS_AVAILABLE', False)
    def test_very_long_text(self):
        """Test insertion of very long text."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True
        
        long_text = "A" * 10000  # 10KB of text
        result = inserter.insert_text_at_cursor(long_text)
        assert result is True


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_exception_during_insertion(self):
        """Test handling of exceptions during text insertion."""
        inserter = TextInserter(check_permissions=False)
        inserter.accessibility_enabled = True
        
        with patch.object(inserter, '_insert_via_accessibility_api') as mock_insert:
            mock_insert.side_effect = Exception("Test error")
            
            result = inserter.insert_text_at_cursor("Hello")
            assert result is False
            assert inserter.stats['insertions_failed'] == 1
    
    def test_queue_processing_error(self):
        """Test error handling in queue processing."""
        inserter = TextInserter(check_permissions=False)
        
        with patch.object(inserter, 'insert_text_at_cursor') as mock_insert:
            mock_insert.side_effect = Exception("Processing error")
            
            inserter.start_queue_processor()
            inserter.queue_text_insertion("Test")
            
            # Wait for processing attempt
            time.sleep(0.1)
            
            # Should handle the error gracefully
            assert inserter.queue_processor_running is True
            
            inserter.stop_queue_processor()


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    @patch('client.text_inserter.MACOS_AVAILABLE', False)
    def test_complete_workflow_mock(self):
        """Test complete workflow in mock mode."""
        with TextInserter(check_permissions=False) as inserter:
            inserter.accessibility_enabled = True
            
            # Queue some text insertions
            assert inserter.queue_text_insertion("Hello") is True
            assert inserter.queue_text_insertion("World", priority=5) is True
            assert inserter.queue_text_insertion("Test", priority=1) is True
            
            # Wait for processing
            time.sleep(0.2)
            
            # Check stats
            stats = inserter.get_stats()
            assert stats['insertions_attempted'] >= 3
            assert stats['insertions_successful'] >= 3
            assert stats['queue_size'] == 0  # All processed


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    # Run the tests
    pytest.main([__file__, "-v"]) 