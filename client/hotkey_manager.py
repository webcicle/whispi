"""
Global Hotkey Manager for Pi-Whispr

This module implements comprehensive global hotkey detection and recording state management,
providing the core functionality for push-to-talk and continuous recording modes.

Features:
- Global Fn key detection for push-to-talk recording
- Fn+SPACE combination for continuous recording toggle
- State machine for recording modes (idle, push-to-talk, continuous)
- Visual feedback for recording state
- Integration with AudioRecorder
- Robust edge case handling for focus loss and key events
- Cross-application hotkey support using pynput
"""

import asyncio
import time
import logging
import threading
from typing import Optional, Callable, List, Dict, Any
from enum import Enum
from dataclasses import dataclass

try:
    from pynput import keyboard
    from pynput.keyboard import Key, KeyCode, Listener
except ImportError:
    # Fallback for testing or environments without pynput
    keyboard = None
    Key = None
    KeyCode = None
    Listener = None

from client.audio_recorder import AudioRecorder
from shared.constants import HOTKEY_SPACE, HOTKEY_LOCK_COMBO

logger = logging.getLogger(__name__)


class RecordingMode(Enum):
    """Recording mode enumeration"""
    IDLE = "idle"
    PUSH_TO_TALK = "push_to_talk"
    CONTINUOUS = "continuous"


class HotkeyState(Enum):
    """Hotkey state enumeration"""
    RELEASED = "released"
    PRESSED = "pressed"
    COMBO_DETECTED = "combo_detected"


@dataclass
class HotkeyEvent:
    """Container for hotkey event data"""
    key: str
    action: str  # 'press' or 'release'
    timestamp: float
    combo_keys: Optional[List[str]] = None


class HotkeyManager:
    """
    Comprehensive hotkey manager for global recording control
    
    Handles:
    - Fn key for push-to-talk recording
    - Fn+SPACE for continuous recording toggle
    - State management and visual feedback
    - Integration with AudioRecorder
    - Edge case handling and error recovery
    """
    
    def __init__(self, 
                 audio_recorder: AudioRecorder,
                 primary_key: str = "fn",
                 lock_combo: List[str] = None):
        """
        Initialize HotkeyManager
        
        Args:
            audio_recorder: AudioRecorder instance for recording control
            primary_key: Primary hotkey for push-to-talk (default: "fn")
            lock_combo: Key combination for continuous mode (default: ["fn", "space"])
        """
        self.audio_recorder = audio_recorder
        self.primary_key = primary_key
        self.lock_combo = lock_combo or ["fn", "space"]
        
        # State management
        self.recording_mode = RecordingMode.IDLE
        self.hotkey_state = HotkeyState.RELEASED
        self.is_recording = False
        self.continuous_mode_active = False
        
        # Timing and tracking
        self.start_time = time.time()
        self.last_key_event_time = 0
        self.current_combo_keys: List[str] = []
        self.combo_timeout = 0.5  # Max time between combo keys
        
        # Keyboard listener
        self.keyboard_listener: Optional[Listener] = None
        self.is_running = False
        
        # Callbacks
        self.state_change_callback: Optional[Callable[[RecordingMode, RecordingMode], None]] = None
        self.recording_change_callback: Optional[Callable[[bool], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None
        
        # Threading
        self._lock = threading.Lock()
        self._combo_timer: Optional[threading.Timer] = None
        
        logger.info(f"HotkeyManager initialized with primary_key='{primary_key}', lock_combo={lock_combo}")
    
    async def start(self) -> bool:
        """
        Start the hotkey manager and global keyboard listener
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("HotkeyManager already running")
            return True
        
        if not keyboard:
            logger.error("pynput not available - hotkey functionality disabled")
            return False
        
        try:
            # Start keyboard listener
            self.keyboard_listener = Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            
            self.keyboard_listener.start()
            self.is_running = True
            
            logger.info("HotkeyManager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start HotkeyManager: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the hotkey manager and cleanup resources"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop keyboard listener
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        
        # Cancel any pending combo timer
        if self._combo_timer:
            self._combo_timer.cancel()
            self._combo_timer = None
        
        # Stop any active recording
        if self.is_recording:
            await self._stop_recording()
        
        logger.info("HotkeyManager stopped")
    
    async def cleanup(self) -> None:
        """Cleanup resources and stop any active recording"""
        await self.stop()
        
        # Ensure recording is stopped
        if self.is_recording:
            await self._stop_recording()
        
        # Reset state
        self.recording_mode = RecordingMode.IDLE
        self.hotkey_state = HotkeyState.RELEASED
        self.continuous_mode_active = False
        
        logger.info("HotkeyManager cleanup completed")
    
    def set_callbacks(self,
                     state_change_callback: Optional[Callable[[RecordingMode, RecordingMode], None]] = None,
                     recording_change_callback: Optional[Callable[[bool], None]] = None,
                     error_callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Set callbacks for state changes and events
        
        Args:
            state_change_callback: Called when recording mode changes (new_mode, old_mode)
            recording_change_callback: Called when recording state changes (is_recording)
            error_callback: Called when errors occur (error_message)
        """
        self.state_change_callback = state_change_callback
        self.recording_change_callback = recording_change_callback
        self.error_callback = error_callback
    
    def _on_key_press(self, key) -> None:
        """Handle key press events from pynput"""
        try:
            key_str = self._normalize_key(key)
            if key_str:
                asyncio.create_task(self._handle_key_press(key_str))
        except Exception as e:
            logger.error(f"Error handling key press: {e}")
    
    def _on_key_release(self, key) -> None:
        """Handle key release events from pynput"""
        try:
            key_str = self._normalize_key(key)
            if key_str:
                asyncio.create_task(self._handle_key_release(key_str))
        except Exception as e:
            logger.error(f"Error handling key release: {e}")
    
    def _normalize_key(self, key) -> Optional[str]:
        """
        Normalize key objects to string representation
        
        Args:
            key: Key object from pynput
            
        Returns:
            str: Normalized key string or None if not recognized
        """
        try:
            if hasattr(key, 'name'):
                # Special keys (Key.space, Key.fn, etc.)
                return key.name.lower()
            elif hasattr(key, 'char') and key.char:
                # Character keys
                return key.char.lower()
            else:
                # Unknown key type
                return str(key).lower().replace("'", "")
        except Exception as e:
            logger.debug(f"Could not normalize key {key}: {e}")
            return None
    
    async def _handle_key_press(self, key: str) -> None:
        """
        Handle key press events with state management
        
        Args:
            key: Normalized key string
        """
        with self._lock:
            current_time = time.time()
            self.last_key_event_time = current_time
            
            logger.debug(f"Key press: {key}, current_mode: {self.recording_mode}, state: {self.hotkey_state}")
            
            # Track combo keys
            if key not in self.current_combo_keys:
                self.current_combo_keys.append(key)
            
            # Check for combo completion
            if self._is_combo_complete():
                await self._handle_combo_detected(self.current_combo_keys.copy())
                return
            
            # Handle primary key for push-to-talk (only if not in continuous mode)
            if key == self.primary_key and not self.continuous_mode_active:
                if self.hotkey_state == HotkeyState.RELEASED:
                    self.hotkey_state = HotkeyState.PRESSED
                    await self._start_push_to_talk()
            
            # Start combo timer if this could be part of a combo
            if len(self.current_combo_keys) == 1 and key in self.lock_combo:
                self._start_combo_timer()
    
    async def _handle_key_release(self, key: str) -> None:
        """
        Handle key release events with state management
        
        Args:
            key: Normalized key string
        """
        with self._lock:
            logger.debug(f"Key release: {key}, current_mode: {self.recording_mode}, state: {self.hotkey_state}")
            
            # Remove from combo tracking
            if key in self.current_combo_keys:
                self.current_combo_keys.remove(key)
            
            # Handle primary key release for push-to-talk
            if (key == self.primary_key and 
                self.recording_mode == RecordingMode.PUSH_TO_TALK and
                self.hotkey_state == HotkeyState.PRESSED):
                
                self.hotkey_state = HotkeyState.RELEASED
                await self._stop_push_to_talk()
            
            # Clear combo tracking if no keys pressed
            if not self.current_combo_keys:
                self._cancel_combo_timer()
    
    def _is_combo_complete(self) -> bool:
        """Check if current keys complete the lock combo"""
        if len(self.current_combo_keys) != len(self.lock_combo):
            return False
        
        return set(self.current_combo_keys) == set(self.lock_combo)
    
    async def _handle_combo_detected(self, combo_keys: List[str]) -> None:
        """
        Handle combo key detection
        
        Args:
            combo_keys: List of keys that formed the combo
        """
        logger.debug(f"Combo detected: {combo_keys}")
        
        # Clear combo tracking
        self.current_combo_keys.clear()
        self._cancel_combo_timer()
        
        # Handle lock combo (Fn+SPACE)
        if set(combo_keys) == set(self.lock_combo):
            self.hotkey_state = HotkeyState.COMBO_DETECTED
            await self._toggle_continuous_mode()
    
    def _start_combo_timer(self) -> None:
        """Start timer for combo detection timeout"""
        self._cancel_combo_timer()
        
        def timeout_handler():
            with self._lock:
                if self.current_combo_keys:
                    logger.debug("Combo timeout - clearing combo tracking")
                    self.current_combo_keys.clear()
        
        self._combo_timer = threading.Timer(self.combo_timeout, timeout_handler)
        self._combo_timer.start()
    
    def _cancel_combo_timer(self) -> None:
        """Cancel any active combo timer"""
        if self._combo_timer:
            self._combo_timer.cancel()
            self._combo_timer = None
    
    async def _start_push_to_talk(self) -> None:
        """Start push-to-talk recording"""
        old_mode = self.recording_mode
        self.recording_mode = RecordingMode.PUSH_TO_TALK
        
        success = await self._start_recording()
        if not success:
            # Recording failed, revert state but keep key press state
            self.recording_mode = old_mode
            return
        
        # Notify callbacks
        if self.state_change_callback:
            self.state_change_callback(self.recording_mode, old_mode)
        
        logger.info("Started push-to-talk recording")
    
    async def _stop_push_to_talk(self) -> None:
        """Stop push-to-talk recording"""
        old_mode = self.recording_mode
        self.recording_mode = RecordingMode.IDLE
        
        await self._stop_recording()
        
        # Notify callbacks
        if self.state_change_callback:
            self.state_change_callback(self.recording_mode, old_mode)
        
        logger.info("Stopped push-to-talk recording")
    
    async def _toggle_continuous_mode(self) -> None:
        """Toggle continuous recording mode"""
        old_mode = self.recording_mode
        
        if self.continuous_mode_active:
            # Turn off continuous mode
            self.continuous_mode_active = False
            self.recording_mode = RecordingMode.IDLE
            await self._stop_recording()
            logger.info("Disabled continuous recording mode")
        else:
            # Turn on continuous mode
            self.continuous_mode_active = True
            self.recording_mode = RecordingMode.CONTINUOUS
            
            success = await self._start_recording()
            if not success:
                # Recording failed, revert state
                self.continuous_mode_active = False
                self.recording_mode = old_mode
                return
            
            logger.info("Enabled continuous recording mode")
        
        # Notify callbacks
        if self.state_change_callback:
            self.state_change_callback(self.recording_mode, old_mode)
    
    async def _start_recording(self) -> bool:
        """
        Start audio recording
        
        Returns:
            bool: True if recording started successfully
        """
        if self.is_recording:
            return True
        
        try:
            success = self.audio_recorder.start_recording()
            if success:
                self.is_recording = True
                
                # Notify callback
                if self.recording_change_callback:
                    self.recording_change_callback(True)
                
                logger.debug("Audio recording started")
                return True
            else:
                logger.error("Failed to start audio recording")
                return False
                
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            if self.error_callback:
                self.error_callback(f"Recording start failed: {e}")
            return False
    
    async def _stop_recording(self) -> None:
        """Stop audio recording"""
        if not self.is_recording:
            return
        
        try:
            self.audio_recorder.stop_recording()
            self.is_recording = False
            
            # Notify callback
            if self.recording_change_callback:
                self.recording_change_callback(False)
            
            logger.debug("Audio recording stopped")
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            if self.error_callback:
                self.error_callback(f"Recording stop failed: {e}")
    
    def get_visual_feedback(self) -> Dict[str, Any]:
        """
        Get visual feedback information for UI display
        
        Returns:
            dict: Visual feedback data with mode, indicator, color, and status text
        """
        if self.recording_mode == RecordingMode.IDLE:
            return {
                "mode": "idle",
                "indicator": "●",
                "color": "gray",
                "status_text": "Ready - Press Fn to record, Fn+Space for continuous mode"
            }
        elif self.recording_mode == RecordingMode.PUSH_TO_TALK:
            return {
                "mode": "push_to_talk",
                "indicator": "🔴",
                "color": "red",
                "status_text": "Recording - Release Fn to stop"
            }
        elif self.recording_mode == RecordingMode.CONTINUOUS:
            return {
                "mode": "continuous",
                "indicator": "🟢",
                "color": "green",
                "status_text": "Continuous recording active - Press Fn+Space to stop"
            }
        else:
            return {
                "mode": "unknown",
                "indicator": "?",
                "color": "yellow",
                "status_text": "Unknown state"
            }
    
    def get_status_info(self) -> Dict[str, Any]:
        """
        Get comprehensive status information
        
        Returns:
            dict: Complete status information for monitoring and debugging
        """
        uptime = time.time() - self.start_time
        
        return {
            "recording_mode": self.recording_mode.value,
            "hotkey_state": self.hotkey_state.value,
            "is_recording": self.is_recording,
            "continuous_mode_active": self.continuous_mode_active,
            "primary_key": self.primary_key,
            "lock_combo": self.lock_combo,
            "is_running": self.is_running,
            "uptime": uptime,
            "last_key_event_time": self.last_key_event_time,
            "current_combo_keys": self.current_combo_keys.copy(),
            "visual_feedback": self.get_visual_feedback(),
            "audio_recorder_status": self.audio_recorder.get_status_info() if self.audio_recorder else None
        } 