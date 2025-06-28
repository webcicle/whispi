"""
Tests for HotkeyManager - Global hotkey detection and recording state management

This module tests the comprehensive hotkey system that handles:
- Global Fn key detection for push-to-talk
- Fn+SPACE combination for continuous recording toggle
- State machine for recording modes (idle, push-to-talk, continuous)
- Visual feedback for recording state
- Integration with AudioRecorder
- Edge case handling for focus loss and key events
"""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch, call
from enum import Enum

# Mock pynput before importing our modules
import sys
sys.modules['pynput'] = MagicMock()
sys.modules['pynput.keyboard'] = MagicMock()

from client.hotkey_manager import HotkeyManager, RecordingMode, HotkeyState
from client.audio_recorder import AudioRecorder


class TestHotkeyManager:
    """Test cases for HotkeyManager functionality"""
    
    @pytest.fixture
    def mock_audio_recorder(self):
        """Create a mock audio recorder"""
        recorder = MagicMock(spec=AudioRecorder)
        recorder.start_recording.return_value = True
        recorder.stop_recording.return_value = True
        recorder.get_status_info.return_value = {
            "state": "stopped",
            "audio_level_db": -40.0,
            "buffer_size": 0,
            "chunk_count": 0
        }
        return recorder
    
    @pytest.fixture
    def hotkey_manager(self, mock_audio_recorder):
        """Create a HotkeyManager instance for testing"""
        manager = HotkeyManager(audio_recorder=mock_audio_recorder)
        return manager
    
    def test_hotkey_manager_initialization(self, hotkey_manager, mock_audio_recorder):
        """Test HotkeyManager initializes with correct default state"""
        assert hotkey_manager.audio_recorder == mock_audio_recorder
        assert hotkey_manager.recording_mode == RecordingMode.IDLE
        assert hotkey_manager.hotkey_state == HotkeyState.RELEASED
        assert not hotkey_manager.is_recording
        assert not hotkey_manager.continuous_mode_active
        assert hotkey_manager.primary_key == "fn"
        assert hotkey_manager.lock_combo == ["fn", "space"]
    
    def test_recording_mode_enum(self):
        """Test RecordingMode enum has correct values"""
        assert RecordingMode.IDLE.value == "idle"
        assert RecordingMode.PUSH_TO_TALK.value == "push_to_talk"
        assert RecordingMode.CONTINUOUS.value == "continuous"
    
    def test_hotkey_state_enum(self):
        """Test HotkeyState enum has correct values"""
        assert HotkeyState.RELEASED.value == "released"
        assert HotkeyState.PRESSED.value == "pressed"
        assert HotkeyState.COMBO_DETECTED.value == "combo_detected"
    
    @pytest.mark.asyncio
    async def test_start_push_to_talk_recording(self, hotkey_manager, mock_audio_recorder):
        """Test starting recording in push-to-talk mode"""
        # Simulate Fn key press
        await hotkey_manager._handle_key_press("fn")
        
        # Verify state changes
        assert hotkey_manager.recording_mode == RecordingMode.PUSH_TO_TALK
        assert hotkey_manager.hotkey_state == HotkeyState.PRESSED
        assert hotkey_manager.is_recording
        
        # Verify audio recorder was called
        mock_audio_recorder.start_recording.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_push_to_talk_recording(self, hotkey_manager, mock_audio_recorder):
        """Test stopping recording when Fn key is released"""
        # First start recording
        await hotkey_manager._handle_key_press("fn")
        
        # Then release key
        await hotkey_manager._handle_key_release("fn")
        
        # Verify state changes
        assert hotkey_manager.recording_mode == RecordingMode.IDLE
        assert hotkey_manager.hotkey_state == HotkeyState.RELEASED
        assert not hotkey_manager.is_recording
        
        # Verify audio recorder was called
        mock_audio_recorder.stop_recording.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_continuous_mode_toggle_on(self, hotkey_manager, mock_audio_recorder):
        """Test toggling continuous recording mode on with Fn+SPACE"""
        # Simulate Fn+SPACE combo
        await hotkey_manager._handle_combo_detected(["fn", "space"])
        
        # Verify state changes
        assert hotkey_manager.recording_mode == RecordingMode.CONTINUOUS
        assert hotkey_manager.continuous_mode_active
        assert hotkey_manager.is_recording
        
        # Verify audio recorder was called
        mock_audio_recorder.start_recording.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_continuous_mode_toggle_off(self, hotkey_manager, mock_audio_recorder):
        """Test toggling continuous recording mode off with Fn+SPACE"""
        # First enable continuous mode
        await hotkey_manager._handle_combo_detected(["fn", "space"])
        
        # Reset mock to track second call
        mock_audio_recorder.reset_mock()
        
        # Toggle off with Fn+SPACE again
        await hotkey_manager._handle_combo_detected(["fn", "space"])
        
        # Verify state changes
        assert hotkey_manager.recording_mode == RecordingMode.IDLE
        assert not hotkey_manager.continuous_mode_active
        assert not hotkey_manager.is_recording
        
        # Verify audio recorder was called
        mock_audio_recorder.stop_recording.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ignore_fn_press_in_continuous_mode(self, hotkey_manager, mock_audio_recorder):
        """Test that Fn key presses are ignored when in continuous mode"""
        # First enable continuous mode
        await hotkey_manager._handle_combo_detected(["fn", "space"])
        
        # Reset mock to track subsequent calls
        mock_audio_recorder.reset_mock()
        
        # Try to use Fn key while in continuous mode
        await hotkey_manager._handle_key_press("fn")
        await hotkey_manager._handle_key_release("fn")
        
        # Verify no additional recording calls
        mock_audio_recorder.start_recording.assert_not_called()
        mock_audio_recorder.stop_recording.assert_not_called()
        
        # Verify still in continuous mode
        assert hotkey_manager.recording_mode == RecordingMode.CONTINUOUS
        assert hotkey_manager.continuous_mode_active
    
    def test_get_visual_feedback_idle(self, hotkey_manager):
        """Test visual feedback display for idle state"""
        feedback = hotkey_manager.get_visual_feedback()
        
        assert feedback["mode"] == "idle"
        assert feedback["indicator"] == "●"
        assert feedback["color"] == "gray"
        assert "ready" in feedback["status_text"].lower()
    
    def test_get_visual_feedback_push_to_talk(self, hotkey_manager):
        """Test visual feedback display for push-to-talk state"""
        hotkey_manager.recording_mode = RecordingMode.PUSH_TO_TALK
        hotkey_manager.is_recording = True
        
        feedback = hotkey_manager.get_visual_feedback()
        
        assert feedback["mode"] == "push_to_talk"
        assert feedback["indicator"] == "🔴"
        assert feedback["color"] == "red"
        assert "recording" in feedback["status_text"].lower()
    
    def test_get_visual_feedback_continuous(self, hotkey_manager):
        """Test visual feedback display for continuous state"""
        hotkey_manager.recording_mode = RecordingMode.CONTINUOUS
        hotkey_manager.continuous_mode_active = True
        hotkey_manager.is_recording = True
        
        feedback = hotkey_manager.get_visual_feedback()
        
        assert feedback["mode"] == "continuous"
        assert feedback["indicator"] == "🟢"
        assert feedback["color"] == "green"
        assert "continuous" in feedback["status_text"].lower()
    
    @pytest.mark.asyncio
    async def test_edge_case_key_release_without_press(self, hotkey_manager, mock_audio_recorder):
        """Test handling key release without corresponding press"""
        # Simulate key release without press (can happen with focus loss)
        await hotkey_manager._handle_key_release("fn")
        
        # Should remain in idle state and not crash
        assert hotkey_manager.recording_mode == RecordingMode.IDLE
        assert not hotkey_manager.is_recording
        
        # Should not call audio recorder
        mock_audio_recorder.start_recording.assert_not_called()
        mock_audio_recorder.stop_recording.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_edge_case_multiple_key_presses(self, hotkey_manager, mock_audio_recorder):
        """Test handling multiple Fn key presses without release"""
        # First press
        await hotkey_manager._handle_key_press("fn")
        
        # Reset mock to track subsequent calls
        mock_audio_recorder.reset_mock()
        
        # Second press (should be ignored)
        await hotkey_manager._handle_key_press("fn")
        
        # Verify no additional calls
        mock_audio_recorder.start_recording.assert_not_called()
        
        # Still in recording state
        assert hotkey_manager.is_recording
        assert hotkey_manager.recording_mode == RecordingMode.PUSH_TO_TALK
    
    @pytest.mark.asyncio
    async def test_cleanup_stops_recording(self, hotkey_manager, mock_audio_recorder):
        """Test cleanup stops any active recording"""
        # Start recording
        await hotkey_manager._handle_key_press("fn")
        
        # Reset mock to track cleanup call
        mock_audio_recorder.reset_mock()
        
        # Cleanup
        await hotkey_manager.cleanup()
        
        # Verify recording was stopped
        mock_audio_recorder.stop_recording.assert_called_once()
        assert not hotkey_manager.is_recording
        assert hotkey_manager.recording_mode == RecordingMode.IDLE
    
    def test_get_status_info(self, hotkey_manager):
        """Test status info provides comprehensive state information"""
        status = hotkey_manager.get_status_info()
        
        assert "recording_mode" in status
        assert "hotkey_state" in status
        assert "is_recording" in status
        assert "continuous_mode_active" in status
        assert "primary_key" in status
        assert "lock_combo" in status
        assert "visual_feedback" in status
        assert "uptime" in status
    
    @pytest.mark.asyncio
    async def test_audio_recorder_failure_handling(self, hotkey_manager, mock_audio_recorder):
        """Test handling when audio recorder fails to start"""
        # Mock audio recorder failure
        mock_audio_recorder.start_recording.return_value = False
        
        # Try to start recording
        await hotkey_manager._handle_key_press("fn")
        
        # Should not be in recording state if audio recorder failed
        assert not hotkey_manager.is_recording
        # But should track the key press state
        assert hotkey_manager.hotkey_state == HotkeyState.PRESSED
    
    @pytest.mark.asyncio
    async def test_invalid_key_handling(self, hotkey_manager, mock_audio_recorder):
        """Test handling of invalid/unrecognized keys"""
        # Press an unrecognized key
        await hotkey_manager._handle_key_press("invalid_key")
        
        # Should remain in idle state
        assert hotkey_manager.recording_mode == RecordingMode.IDLE
        assert not hotkey_manager.is_recording
        
        # Should not call audio recorder
        mock_audio_recorder.start_recording.assert_not_called()
    
    def test_callback_registration(self, hotkey_manager):
        """Test callback registration for state changes"""
        state_callback = MagicMock()
        recording_callback = MagicMock()
        
        hotkey_manager.set_callbacks(
            state_change_callback=state_callback,
            recording_change_callback=recording_callback
        )
        
        assert hotkey_manager.state_change_callback == state_callback
        assert hotkey_manager.recording_change_callback == recording_callback
    
    @pytest.mark.asyncio
    async def test_callbacks_called_on_state_change(self, hotkey_manager, mock_audio_recorder):
        """Test that callbacks are called when state changes"""
        state_callback = MagicMock()
        recording_callback = MagicMock()
        
        hotkey_manager.set_callbacks(
            state_change_callback=state_callback,
            recording_change_callback=recording_callback
        )
        
        # Start recording
        await hotkey_manager._handle_key_press("fn")
        
        # Verify callbacks were called
        state_callback.assert_called_with(RecordingMode.PUSH_TO_TALK, RecordingMode.IDLE)
        recording_callback.assert_called_with(True)


class TestHotkeyManagerIntegration:
    """Integration tests for HotkeyManager with real components"""
    
    @pytest.fixture
    def real_audio_recorder(self):
        """Create a real AudioRecorder for integration testing"""
        # We'll use a mock here too since we can't test real audio in CI
        recorder = MagicMock(spec=AudioRecorder)
        recorder.start_recording.return_value = True
        recorder.stop_recording.return_value = True
        return recorder
    
    @pytest.mark.asyncio
    async def test_complete_push_to_talk_workflow(self, real_audio_recorder):
        """Test complete push-to-talk workflow"""
        manager = HotkeyManager(audio_recorder=real_audio_recorder)
        
        # Start workflow
        await manager.start()
        
        # Simulate user pressing and releasing Fn key
        await manager._handle_key_press("fn")
        assert manager.is_recording
        
        # Simulate some recording time
        await asyncio.sleep(0.1)
        
        await manager._handle_key_release("fn")
        assert not manager.is_recording
        
        # Cleanup
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_complete_continuous_mode_workflow(self, real_audio_recorder):
        """Test complete continuous recording workflow"""
        manager = HotkeyManager(audio_recorder=real_audio_recorder)
        
        # Start workflow
        await manager.start()
        
        # Toggle continuous mode on
        await manager._handle_combo_detected(["fn", "space"])
        assert manager.continuous_mode_active
        assert manager.is_recording
        
        # Simulate some recording time
        await asyncio.sleep(0.1)
        
        # Toggle continuous mode off
        await manager._handle_combo_detected(["fn", "space"])
        assert not manager.continuous_mode_active
        assert not manager.is_recording
        
        # Cleanup
        await manager.cleanup() 