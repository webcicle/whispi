#!/usr/bin/env python3
"""
Text Insertion Module for macOS using Accessibility API

This module provides functionality to insert transcribed text at the cursor position
in any macOS application using the Accessibility API.
"""

import asyncio
import logging
import queue
import threading
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# macOS-specific imports
try:
    import Cocoa
    import ApplicationServices
    from Cocoa import (
        NSWorkspace, NSRunningApplication, NSApplication, NSEvent,
        NSApplicationActivationPolicyRegular
    )
    from ApplicationServices import (
        AXUIElementCreateSystemWide, AXUIElementCreateApplication,
        AXUIElementCopyAttributeValue, AXUIElementSetAttributeValue,
        AXUIElementGetAttributeValueCount, AXUIElementCopyAttributeNames,
        AXValueCreate, AXValueGetValue, AXUIElementPerformAction,
        kAXFocusedUIElementAttribute, kAXSelectedTextAttribute,
        kAXSelectedTextRangeAttribute, kAXValueAttribute,
        kAXTitleAttribute, kAXRoleAttribute, kAXTextFieldRole,
        kAXTextAreaRole, kAXComboBoxRole, kAXFocusedApplicationAttribute,
        kAXInsertionPointLineNumberAttribute, AXError,
        kAXErrorSuccess, kAXErrorAPIDisabled, kAXErrorNotImplemented,
        kAXErrorAttributeUnsupported, kAXErrorActionUnsupported,
        AXIsProcessTrusted, AXIsProcessTrustedWithOptions,
        kAXTrustedCheckOptionPrompt
    )
    MACOS_AVAILABLE = True
except ImportError:
    MACOS_AVAILABLE = False
    # Mock objects for non-macOS environments
    class MockCocoa:
        pass
    
    # Create mock versions of all macOS-specific functions and constants
    # This ensures they're always available as module attributes for testing
    # Use globals() to explicitly set module-level attributes
    mock_attrs = {
        'Cocoa': MockCocoa(),
        'ApplicationServices': MockCocoa(),
        'NSWorkspace': None,
        'NSRunningApplication': None,
        'NSApplication': None,
        'NSEvent': None,
        'NSApplicationActivationPolicyRegular': None,
        'AXUIElementCreateSystemWide': None,
        'AXUIElementCreateApplication': None,
        'AXUIElementCopyAttributeValue': None,
        'AXUIElementSetAttributeValue': None,
        'AXUIElementGetAttributeValueCount': None,
        'AXUIElementCopyAttributeNames': None,
        'AXValueCreate': None,
        'AXValueGetValue': None,
        'AXUIElementPerformAction': None,
        'kAXFocusedUIElementAttribute': None,
        'kAXSelectedTextAttribute': None,
        'kAXSelectedTextRangeAttribute': None,
        'kAXValueAttribute': None,
        'kAXTitleAttribute': None,
        'kAXRoleAttribute': None,
        'kAXTextFieldRole': None,
        'kAXTextAreaRole': None,
        'kAXComboBoxRole': None,
        'kAXFocusedApplicationAttribute': None,
        'kAXInsertionPointLineNumberAttribute': None,
        'AXError': None,
        'kAXErrorSuccess': None,
        'kAXErrorAPIDisabled': None,
        'kAXErrorNotImplemented': None,
        'kAXErrorAttributeUnsupported': None,
        'kAXErrorActionUnsupported': None,
        'AXIsProcessTrusted': None,
        'AXIsProcessTrustedWithOptions': None,
        'kAXTrustedCheckOptionPrompt': None
    }
    
    # Set all mock attributes as module globals
    globals().update(mock_attrs)


class InsertionMethod(Enum):
    """Available text insertion methods."""
    ACCESSIBILITY_API = "accessibility_api"
    APPLESCRIPT = "applescript"
    KEYSTROKE_SIMULATION = "keystroke_simulation"


class InsertionError(Exception):
    """Custom exception for text insertion errors."""
    pass


@dataclass
class TextInsertionRequest:
    """Represents a text insertion request."""
    text: str
    timestamp: float
    priority: int = 0  # Higher number = higher priority
    method: Optional[InsertionMethod] = None
    
    def __post_init__(self):
        if self.method is None:
            self.method = InsertionMethod.ACCESSIBILITY_API


@dataclass
class ApplicationInfo:
    """Information about the currently active application."""
    pid: int
    name: str
    bundle_id: str
    focused_element: Any = None
    supports_text_insertion: bool = False
    text_field_type: Optional[str] = None


class TextInserter:
    """
    Main class for inserting text via macOS Accessibility API.
    
    Features:
    - Permission handling for accessibility features
    - Active application and text field detection
    - Queue system for rapid transcription results
    - Special character and formatting handling
    - Fallback methods when primary insertion fails
    """
    
    def __init__(self, max_queue_size: int = 100, check_permissions: bool = True):
        """
        Initialize the TextInserter.
        
        Args:
            max_queue_size: Maximum number of queued insertion requests
            check_permissions: Whether to check accessibility permissions on init
        """
        self.logger = logging.getLogger(__name__)
        self.max_queue_size = max_queue_size
        self.insertion_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.queue_processor_running = False
        self.queue_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Accessibility state
        self.accessibility_enabled = False
        self.current_app_info: Optional[ApplicationInfo] = None
        
        # Statistics
        self.stats = {
            'insertions_attempted': 0,
            'insertions_successful': 0,
            'insertions_failed': 0,
            'queue_overflows': 0,
            'permission_errors': 0
        }
        
        if not MACOS_AVAILABLE:
            self.logger.warning("macOS frameworks not available. TextInserter will use mock functionality.")
            return
            
        if check_permissions:
            self.check_accessibility_permissions()
    
    def check_accessibility_permissions(self) -> bool:
        """
        Check if accessibility permissions are granted.
        
        Returns:
            True if permissions are granted, False otherwise
        """
        if not MACOS_AVAILABLE:
            self.logger.warning("macOS not available, cannot check accessibility permissions")
            return False
            
        try:
            # Check if process is trusted (has accessibility permissions)
            trusted = AXIsProcessTrusted()
            if not trusted:
                self.logger.warning("Accessibility permissions not granted")
                self.accessibility_enabled = False
                return False
            
            self.accessibility_enabled = True
            self.logger.info("Accessibility permissions verified")
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking accessibility permissions: {e}")
            self.accessibility_enabled = False
            return False
    
    def request_accessibility_permissions(self) -> bool:
        """
        Request accessibility permissions from the user.
        
        Returns:
            True if permissions are granted after request, False otherwise
        """
        if not MACOS_AVAILABLE:
            return False
            
        try:
            # Request permissions with prompt
            options = {kAXTrustedCheckOptionPrompt: True}
            trusted = AXIsProcessTrustedWithOptions(options)
            
            if trusted:
                self.accessibility_enabled = True
                self.logger.info("Accessibility permissions granted")
                return True
            else:
                self.logger.warning("Accessibility permissions denied by user")
                self.accessibility_enabled = False
                return False
                
        except Exception as e:
            self.logger.error(f"Error requesting accessibility permissions: {e}")
            self.accessibility_enabled = False
            return False
    
    def get_active_application_info(self) -> Optional[ApplicationInfo]:
        """
        Get information about the currently active application.
        
        Returns:
            ApplicationInfo object or None if detection fails
        """
        if not MACOS_AVAILABLE or not self.accessibility_enabled:
            return None
            
        try:
            # Get the system-wide accessibility object
            system_element = AXUIElementCreateSystemWide()
            
            # Get the focused application
            focused_app_ref = AXUIElementCopyAttributeValue(
                system_element, kAXFocusedApplicationAttribute
            )[1]
            
            if focused_app_ref is None:
                return None
            
            # Get application PID
            pid = AXUIElementCopyAttributeValue(focused_app_ref, "AXProcessIdentifier")[1]
            
            # Get running application info
            running_app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            if not running_app:
                return None
            
            app_name = running_app.localizedName()
            bundle_id = running_app.bundleIdentifier()
            
            # Get focused UI element
            focused_element_ref = AXUIElementCopyAttributeValue(
                focused_app_ref, kAXFocusedUIElementAttribute
            )[1]
            
            supports_text = False
            text_field_type = None
            
            if focused_element_ref:
                # Check if focused element supports text insertion
                role_result = AXUIElementCopyAttributeValue(focused_element_ref, kAXRoleAttribute)
                if role_result[0] == kAXErrorSuccess:
                    role = role_result[1]
                    text_roles = [kAXTextFieldRole, kAXTextAreaRole, kAXComboBoxRole]
                    if role in text_roles:
                        supports_text = True
                        text_field_type = role
            
            app_info = ApplicationInfo(
                pid=pid,
                name=app_name,
                bundle_id=bundle_id,
                focused_element=focused_element_ref,
                supports_text_insertion=supports_text,
                text_field_type=text_field_type
            )
            
            self.current_app_info = app_info
            return app_info
            
        except Exception as e:
            self.logger.error(f"Error getting active application info: {e}")
            return None
    
    def insert_text_at_cursor(self, text: str, method: InsertionMethod = InsertionMethod.ACCESSIBILITY_API) -> bool:
        """
        Insert text at the current cursor position.
        
        Args:
            text: Text to insert
            method: Insertion method to use
            
        Returns:
            True if insertion was successful, False otherwise
        """
        if not text:
            return True  # Nothing to insert
            
        self.stats['insertions_attempted'] += 1
        
        if not self.accessibility_enabled and method == InsertionMethod.ACCESSIBILITY_API:
            self.logger.error("Accessibility API not available")
            self.stats['permission_errors'] += 1
            self.stats['insertions_failed'] += 1
            return False
        
        try:
            if method == InsertionMethod.ACCESSIBILITY_API:
                success = self._insert_via_accessibility_api(text)
            elif method == InsertionMethod.APPLESCRIPT:
                success = self._insert_via_applescript(text)
            elif method == InsertionMethod.KEYSTROKE_SIMULATION:
                success = self._insert_via_keystroke_simulation(text)
            else:
                self.logger.error(f"Unknown insertion method: {method}")
                success = False
            
            if success:
                self.stats['insertions_successful'] += 1
                self.logger.debug(f"Successfully inserted text: '{text[:50]}...'")
            else:
                self.stats['insertions_failed'] += 1
                self.logger.warning(f"Failed to insert text: '{text[:50]}...'")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error inserting text: {e}")
            self.stats['insertions_failed'] += 1
            return False
    
    def _insert_via_accessibility_api(self, text: str) -> bool:
        """Insert text using the Accessibility API."""
        if not MACOS_AVAILABLE:
            self.logger.debug(f"Mock insertion via Accessibility API: {text}")
            return True
            
        try:
            app_info = self.get_active_application_info()
            if not app_info or not app_info.supports_text_insertion:
                self.logger.warning("Current application does not support text insertion")
                return False
            
            focused_element = app_info.focused_element
            if not focused_element:
                self.logger.warning("No focused text element found")
                return False
            
            # Get current selection range
            selection_range_result = AXUIElementCopyAttributeValue(
                focused_element, kAXSelectedTextRangeAttribute
            )
            
            if selection_range_result[0] != kAXErrorSuccess:
                # If we can't get selection range, try to set the value directly
                return self._set_text_value_directly(focused_element, text)
            
            # Insert text at the current cursor position
            current_value_result = AXUIElementCopyAttributeValue(focused_element, kAXValueAttribute)
            if current_value_result[0] != kAXErrorSuccess:
                current_value = ""
            else:
                current_value = current_value_result[1] or ""
            
            # Get selection range
            selection_range = selection_range_result[1]
            if selection_range:
                range_value = AXValueGetValue(selection_range, None, None)
                if range_value:
                    location, length = range_value
                    # Insert text at cursor position
                    new_value = current_value[:location] + text + current_value[location + length:]
                    
                    # Set the new value
                    set_result = AXUIElementSetAttributeValue(focused_element, kAXValueAttribute, new_value)
                    if set_result == kAXErrorSuccess:
                        # Update cursor position
                        new_location = location + len(text)
                        new_range = AXValueCreate(ApplicationServices.kAXValueCFRangeType, (new_location, 0))
                        AXUIElementSetAttributeValue(focused_element, kAXSelectedTextRangeAttribute, new_range)
                        return True
            
            # Fallback: append text
            return self._set_text_value_directly(focused_element, current_value + text)
            
        except Exception as e:
            self.logger.error(f"Error in accessibility API insertion: {e}")
            return False
    
    def _set_text_value_directly(self, element, text: str) -> bool:
        """Set text value directly on an element."""
        try:
            result = AXUIElementSetAttributeValue(element, kAXValueAttribute, text)
            return result == kAXErrorSuccess
        except Exception as e:
            self.logger.error(f"Error setting text value directly: {e}")
            return False
    
    def _insert_via_applescript(self, text: str) -> bool:
        """Insert text using AppleScript (fallback method)."""
        # This would be implemented in task 10 as a fallback
        self.logger.info("AppleScript insertion not yet implemented (task 10)")
        return False
    
    def _insert_via_keystroke_simulation(self, text: str) -> bool:
        """Insert text using keystroke simulation (emergency fallback)."""
        # Emergency fallback - would simulate typing
        self.logger.info("Keystroke simulation not yet implemented")
        return False
    
    def queue_text_insertion(self, text: str, priority: int = 0, 
                           method: Optional[InsertionMethod] = None) -> bool:
        """
        Queue a text insertion request for processing.
        
        Args:
            text: Text to insert
            priority: Priority level (higher = more urgent)
            method: Insertion method to use
            
        Returns:
            True if request was queued, False if queue is full
        """
        try:
            request = TextInsertionRequest(
                text=text,
                timestamp=time.time(),
                priority=priority,
                method=method or InsertionMethod.ACCESSIBILITY_API
            )
            
            # Use negative priority for priority queue (higher priority = lower number)
            self.insertion_queue.put((-priority, request.timestamp, request), block=False)
            self.logger.debug(f"Queued text insertion: '{text[:50]}...'")
            return True
            
        except queue.Full:
            self.logger.warning("Text insertion queue is full, dropping request")
            self.stats['queue_overflows'] += 1
            return False
    
    def start_queue_processor(self):
        """Start the background queue processor thread."""
        if self.queue_processor_running:
            return
            
        self._stop_event.clear()
        self.queue_processor_running = True
        self.queue_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.queue_thread.start()
        self.logger.info("Text insertion queue processor started")
    
    def stop_queue_processor(self):
        """Stop the background queue processor thread."""
        if not self.queue_processor_running:
            return
            
        self._stop_event.set()
        self.queue_processor_running = False
        
        if self.queue_thread and self.queue_thread.is_alive():
            self.queue_thread.join(timeout=2.0)
            
        self.logger.info("Text insertion queue processor stopped")
    
    def _process_queue(self):
        """Process queued text insertion requests."""
        while self.queue_processor_running and not self._stop_event.is_set():
            try:
                # Get next request with timeout
                priority, timestamp, request = self.insertion_queue.get(timeout=0.1)
                
                # Process the request
                self.insert_text_at_cursor(request.text, request.method)
                
                # Mark task as done
                self.insertion_queue.task_done()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing queue: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get insertion statistics."""
        return {
            **self.stats,
            'queue_size': self.insertion_queue.qsize(),
            'accessibility_enabled': self.accessibility_enabled,
            'current_app': self.current_app_info.name if self.current_app_info else None
        }
    
    def clear_queue(self):
        """Clear all pending insertion requests."""
        while not self.insertion_queue.empty():
            try:
                self.insertion_queue.get_nowait()
                self.insertion_queue.task_done()
            except queue.Empty:
                break
        self.logger.info("Text insertion queue cleared")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_queue_processor()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_queue_processor()
