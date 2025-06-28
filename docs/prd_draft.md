# Pi-Whispr Product Requirements Document

## Overview

This Product Requirements Document (PRD) defines the specifications for Pi-Whispr, a local speech-to-text system that provides fast, private transcription through Raspberry Pi 5 server integration with macOS clients. The system achieves sub-2 second latency for real-time dictation without relying on cloud services, ensuring complete privacy and zero ongoing costs.

Pi-Whispr implements the "fastest" approach from three evaluated architectures, prioritizing rapid deployment and reliable performance using WebSocket communication and Python-based components. This MVP establishes the foundation for future enhancements including WebRTC optimization and native Swift applications.

## Product vision

Create a seamless, local speech-to-text solution that enables users to dictate text into any macOS application with minimal latency, maximum privacy, and zero ongoing costs through Raspberry Pi-powered local processing.

## Product goals

### Primary objectives

- Achieve consistent sub-2 second transcription latency from voice input to text insertion
- Provide completely local processing with no cloud dependencies or internet requirements
- Enable universal text insertion across all macOS applications and text fields
- Deliver reliable push-to-talk and continuous recording modes through intuitive hotkeys
- Maintain zero ongoing operational costs after initial hardware setup

### Secondary objectives

- Establish development foundation for future WebRTC and Swift native optimizations
- Demonstrate viability of edge AI processing for real-time applications
- Provide Docker-based development environment for rapid iteration and testing
- Create extensible architecture supporting multiple transcription models and configurations

## Target users

### Primary users

- **Software developers and writers** who spend significant time typing and would benefit from efficient dictation workflows
- **Remote workers** requiring private, offline transcription capabilities for sensitive communications
- **Privacy-conscious users** seeking alternatives to cloud-based speech recognition services

### Secondary users

- **Accessibility users** who benefit from voice input alternatives to traditional typing
- **Content creators** needing efficient transcription for video, podcast, or written content development
- **Technical enthusiasts** interested in local AI processing and edge computing implementations

## Product scope

### In scope (MVP - Phase 1)

- **Core transcription workflow**: Voice recording → Pi processing → text insertion
- **macOS client application**: Python-based client with global hotkey detection and audio recording
- **Raspberry Pi server**: WebSocket server with faster-whisper integration
- **Communication protocol**: Structured WebSocket messages for audio streaming and results
  - **Text insertion system**: Accessibility API integration with AppleScript fallback
  - **Development environment**: Docker containers for local testing and development
  - **Recording modes**: Push-to-talk (Fn) and continuous recording lock (Fn+SPACE)

### Out of scope (Future phases)

- **WebRTC integration**: Low-latency real-time communication (Phase 2)
- **Swift native application**: Native macOS performance optimization (Phase 3)
- **Multi-language support**: Languages beyond English transcription (Phase 4)
- **Custom vocabulary training**: Domain-specific model fine-tuning (Phase 4)
- **Cross-platform clients**: Windows or Linux client applications (Phase 5)

## User stories and acceptance criteria

### Core recording and transcription workflow

**US-001: Basic voice recording and transcription**

- **As a** user **I want to** record my voice and receive accurate text transcription **so that** I can efficiently convert speech to text for any application
- **Acceptance criteria:**
  - User can initiate recording using designated hotkey (Fn)
  - Audio is captured at 16kHz mono quality for optimal Whisper processing
  - Recording stops when user releases hotkey or after maximum duration (30 seconds)
- Transcribed text appears within 2 seconds of recording completion
- Transcription accuracy exceeds 70% for clear English speech
- System handles background noise and audio interruptions gracefully

  **US-002: Push-to-talk recording mode**

  - **As a** user **I want to** hold the Fn key to record voice input **so that** I can control exactly when audio is captured
  - **Acceptance criteria:**
    - Recording begins immediately when Fn key is pressed down
    - Visual feedback indicates active recording state
    - Recording continues while Fn key remains pressed
    - Recording stops immediately when Fn key is released
    - No audio is captured when Fn key is not pressed
    - System prevents conflicts with other applications using Fn key

**US-003: Continuous recording lock mode**

- **As a** user **I want to** toggle continuous recording with Fn+SPACE **so that** I can dictate longer content without holding keys
- **Acceptance criteria:**
  - Fn+SPACE combination toggles recording lock on/off
  - Clear visual feedback shows when lock mode is active
  - Recording continues automatically until lock is disabled
  - Voice Activity Detection (VAD) manages recording segments
  - User can disable lock mode with another Fn+SPACE press
  - Lock mode times out after reasonable inactivity period (30 seconds)

### Text insertion and application integration

**US-004: Universal text insertion**

- **As a** user **I want** transcribed text automatically inserted at my cursor position **so that** I can seamlessly dictate into any application
- **Acceptance criteria:**
  - Text appears at current cursor/focus position in any macOS application
  - Insertion works in text editors, browsers, messaging apps, and form fields
  - Text formatting is preserved appropriately for the target application
  - Insertion process doesn't disrupt user's current workflow or window focus
  - Fallback mechanisms ensure compatibility across different application types

**US-005: Accessibility API text insertion**

- **As a** user **I want** reliable text insertion using macOS Accessibility APIs **so that** I have precise control over text placement
- **Acceptance criteria:**
  - Primary text insertion uses macOS Accessibility API for accurate positioning
  - System requests and validates accessibility permissions on first use
  - Text insertion respects existing text selection and cursor position
  - Multi-line text is handled correctly with proper line breaks
  - Unicode characters and special symbols are inserted accurately

**US-006: AppleScript fallback text insertion**

- **As a** user **I want** AppleScript fallback for text insertion **so that** I have reliable text insertion even when Accessibility API fails
- **Acceptance criteria:**
  - AppleScript fallback activates automatically when Accessibility API is unavailable
  - Fallback method simulates typing for broad application compatibility
  - User is notified which insertion method is active
  - Both methods handle special characters and formatting appropriately
  - Fallback performance remains within acceptable latency limits

### System communication and reliability

**US-007: WebSocket client-server communication**

- **As a** developer **I want** reliable WebSocket communication between client and server **so that** audio data and results are transmitted efficiently
- **Acceptance criteria:**
  - Client establishes WebSocket connection to Pi server on startup
  - Audio data is streamed in real-time 20ms chunks for processing
  - Structured JSON messages handle all communication types (audio, results, status)
  - Connection automatically reconnects after network interruptions
  - Communication protocol supports multiple concurrent client connections
  - Message acknowledgment ensures data integrity

**US-008: Mock server development environment**

- **As a** developer **I want** a mock server for development testing **so that** I can develop and test the client without requiring a Pi setup
- **Acceptance criteria:**
  - Mock server simulates realistic transcription processing times (0.2-1.5 seconds)
  - Mock responses include transcribed text, confidence scores, and processing metadata
  - Docker container provides consistent development environment
  - Mock server supports all WebSocket protocol message types
  - Development workflow tests complete client functionality before Pi deployment

**US-009: Pi server deployment and operation**

- **As a** user **I want** the Pi server to run reliably with optimal performance **so that** I can depend on consistent transcription services
- **Acceptance criteria:**
  - Server runs continuously as systemd service on Raspberry Pi 5
  - faster-whisper model loads efficiently with appropriate memory usage
  - Temperature monitoring prevents thermal throttling during sustained operation
  - Server handles multiple client connections without performance degradation
  - Graceful error handling for audio processing failures and model loading issues

### Configuration and performance management

**US-010: Audio configuration optimization**

- **As a** user **I want** optimized audio settings for best transcription quality **so that** I achieve maximum accuracy and performance
- **Acceptance criteria:**
  - Audio recorded at 16kHz sample rate (Whisper's native format)
  - Mono channel recording reduces processing overhead
  - 20ms chunk size optimizes real-time processing
  - Voice Activity Detection reduces unnecessary processing by 60-80%
  - Audio format specifications prevent resampling overhead

**US-011: Model configuration management**

- **As a** user **I want** appropriate model selection for development vs production **so that** I can optimize for speed or accuracy based on context
- **Acceptance criteria:**
  - Development environment uses tiny.en model for fast iteration (3-9 second processing)
  - Production deployment uses small.en model for improved accuracy (10-30 second processing)
  - Model switching requires only configuration changes, not code modifications
  - Models are cached locally to prevent repeated downloads
  - System provides feedback about model loading status and memory usage

**US-012: Performance monitoring and optimization**

- **As a** user **I want** system performance monitoring **so that** I can ensure optimal operation and troubleshoot issues
- **Acceptance criteria:**
  - Processing time metrics are logged for each transcription request
  - Network latency monitoring tracks WebSocket communication performance
  - Pi temperature and CPU usage monitoring prevents thermal throttling
  - Client provides visual feedback for connection status and processing state
  - Performance logs help identify bottlenecks and optimization opportunities

### Error handling and reliability

**US-013: Network interruption handling**

- **As a** user **I want** graceful handling of network interruptions **so that** I can continue using the system despite connectivity issues
- **Acceptance criteria:**
  - Client automatically attempts reconnection with exponential backoff
  - Audio recording continues locally during connection interruptions
  - Queued audio is transmitted once connection is restored
  - User receives clear notification of connection status
  - Maximum retry attempts prevent infinite connection loops

**US-014: Audio hardware error handling**

- **As a** user **I want** robust audio system error handling **so that** microphone issues don't crash the application
- **Acceptance criteria:**
  - System detects and reports microphone access permission issues
  - Graceful handling of audio device disconnection/reconnection
  - Alternative audio device selection when primary device fails
  - Clear error messages guide user through permission and hardware troubleshooting
  - Audio system recovery without requiring application restart

**US-015: Transcription processing error handling**

- **As a** user **I want** reliable error handling for transcription failures **so that** processing errors don't interrupt my workflow
- **Acceptance criteria:**
  - Server handles malformed audio data without crashing
  - Client receives meaningful error messages for transcription failures
  - Timeout handling prevents indefinite waiting for processing results
  - Failed transcription attempts are logged for debugging and improvement
  - System continues operating normally after individual transcription failures

## Functional requirements

### Audio recording system

- **Microphone access**: Request and manage macOS microphone permissions
- **Audio capture**: Record 16kHz mono audio using PyAudio with 20ms chunk size
  - **Voice Activity Detection**: Built-in faster-whisper VAD with 500ms silence detection
  - **Recording controls**: Global hotkey detection for Fn (push-to-talk) and Fn+SPACE (toggle lock)
- **Audio processing**: Real-time audio streaming with silence detection and timeout handling
- **Format optimization**: Direct 16-bit integer audio format to minimize preprocessing overhead

### Network communication system

- **WebSocket protocol**: Implement structured JSON messaging protocol for client-server communication
- **Connection management**: Automatic connection establishment, heartbeat monitoring, and reconnection logic
- **Audio streaming**: Real-time transmission of 20ms audio chunks with minimal buffering
- **Message handling**: Support for connection, audio processing, transcription result, and error message types
- **Multiple clients**: Server capability to handle concurrent client connections

### Speech recognition system

- **Model integration**: faster-whisper implementation with configurable model selection (tiny.en/small.en)
- **Processing pipeline**: Audio preprocessing, transcription processing, and result formatting
- **Performance optimization**: Model caching, memory management, and processing timeout handling
- **Result formatting**: Structured transcription results with confidence scores and processing metrics
- **Error handling**: Graceful handling of audio processing failures and model loading issues

### Text insertion system

- **Accessibility API**: Primary text insertion using macOS Accessibility framework for precise cursor positioning
- **AppleScript fallback**: Secondary insertion method using System Events keystroke simulation
- **Permission management**: Request and validate accessibility permissions with user guidance
- **Format preservation**: Maintain text formatting and handle special characters appropriately
- **Focus management**: Respect current application focus and cursor position for accurate insertion

### Development and deployment system

- **Docker environment**: Containerized mock server for development testing without Pi hardware
- **Mock transcription**: Simulated processing with realistic response times and data structures
- **Configuration management**: Environment-based configuration for development vs production settings
- **Logging system**: Structured logging with configurable verbosity for debugging and monitoring
- **Testing framework**: Automated testing capabilities for core functionality validation

## Technical requirements

### Performance specifications

- **Total latency target**: Maximum 2 seconds from recording end to text insertion
- **Network latency**: WebSocket communication under 100ms for local network
- **Audio processing latency**: Voice Activity Detection and streaming under 500ms
- **Transcription processing**: tiny.en model 3-9 seconds, small.en model 10-30 seconds (Pi 5 hardware)
- **Text insertion latency**: Accessibility API insertion under 50ms

### Hardware requirements

- **Raspberry Pi 5**: 8GB RAM model with active cooling for sustained operation
- **Storage**: 32GB+ high-speed storage for model files and system operation
- **Network**: Ethernet or WiFi connection between Pi server and macOS client
- **macOS client**: macOS 12+ with microphone access and accessibility permissions
- **Audio hardware**: Built-in or external microphone with clear speech capture capability

### Software requirements

- **Pi server**: Python 3.9+ with faster-whisper, websockets, and supporting libraries
- **macOS client**: Python 3.9+ with PyAudio, pynput, pyobjc frameworks
- **Development environment**: Docker for containerized testing and development
- **Operating systems**: Raspberry Pi OS Bookworm, macOS 12+ (client)
- **Model dependencies**: Whisper tiny.en (39MB) and small.en (244MB) models

### Security and privacy requirements

- **Local processing**: All audio processing occurs locally without cloud transmission
- **Data encryption**: WebSocket communication encrypted for network security
- **Audio storage**: No persistent audio storage; processing occurs in memory only
- **Permission management**: Explicit user consent for microphone and accessibility access
- **Network isolation**: Option to operate on isolated local network without internet access

### Scalability and reliability requirements

- **Concurrent connections**: Support multiple macOS clients connecting to single Pi server
- **Thermal management**: Active cooling and temperature monitoring to prevent throttling
- **Error recovery**: Automatic recovery from network, audio, and processing failures
- **Resource management**: Efficient memory usage and model caching for sustained operation
- **Update capability**: Configuration updates without system restart requirements

## Non-functional requirements

### Usability requirements

- **Intuitive hotkeys**: Natural key combinations that don't conflict with common application shortcuts
- **Visual feedback**: Clear indicators for recording state, connection status, and processing activity
- **Minimal setup**: Streamlined installation and configuration process under 30 minutes
- **Error messaging**: User-friendly error messages with actionable troubleshooting guidance
- **Documentation**: Comprehensive setup guides and troubleshooting resources

### Reliability requirements

- **System uptime**: Pi server maintains 99%+ uptime during normal operation
- **Error handling**: Graceful degradation and recovery from individual component failures
- **Data integrity**: Ensure audio data transmission accuracy without corruption
- **Connection stability**: Robust WebSocket connection management with automatic recovery
- **Processing consistency**: Reliable transcription quality across varying audio conditions

### Performance requirements

- **Response time**: 95% of transcription requests complete within latency targets
- **Throughput**: Support sustained dictation workloads without performance degradation
- **Resource efficiency**: Optimal CPU and memory usage on both client and server
- **Audio quality**: Maintain transcription accuracy across reasonable background noise levels
- **Network efficiency**: Minimize bandwidth usage while maintaining real-time performance

### Maintainability requirements

- **Code organization**: Modular architecture with clear separation of concerns
- **Configuration management**: Environment-based settings for easy deployment variation
- **Logging and monitoring**: Comprehensive logging for troubleshooting and optimization
- **Documentation**: Technical documentation for development and deployment procedures
- **Testing framework**: Automated testing capabilities for regression prevention

### Compatibility requirements

- **macOS versions**: Support macOS 12+ with graceful handling of older versions
- **Application compatibility**: Text insertion works across diverse macOS applications
- **Hardware compatibility**: Support various microphone hardware and audio devices
- **Network environments**: Function on standard home and office network configurations
- **Python versions**: Compatible with Python 3.9+ installations

## Success metrics

### Performance metrics

- **Latency measurement**: 95% of transcriptions complete within 2-second target
- **Accuracy measurement**: Transcription accuracy exceeds 70% for clear English speech
- **Uptime measurement**: System availability exceeds 99% during active usage periods
- **Error rate**: Less than 5% of transcription attempts result in processing failures
- **Connection stability**: WebSocket connections maintain stability with <1% disconnection rate

### User experience metrics

- **Setup completion**: 90% of users complete initial setup within 30 minutes
- **Feature adoption**: 80% of users successfully use both push-to-talk and lock modes
- **Error resolution**: Users resolve permission and configuration issues within 10 minutes
- **Usage satisfaction**: User feedback indicates satisfaction with transcription speed and accuracy
- **Workflow integration**: Users report successful integration into daily productivity workflows

### Technical metrics

- **Resource utilization**: Pi CPU usage remains under 80% during sustained operation
- **Memory efficiency**: Application memory usage stays within acceptable limits on both client and server
- **Network performance**: WebSocket communication latency averages under 50ms on local networks
- **Temperature management**: Pi temperature remains under thermal throttling threshold (80°C) with active cooling
- **Model performance**: Transcription processing times meet specifications for selected models

### Development and deployment metrics

- **Development velocity**: Mock server enables rapid client development without Pi dependency
- **Testing coverage**: Automated tests cover critical functionality and error scenarios
- **Documentation quality**: Setup documentation enables successful deployment without developer assistance
- **Issue resolution**: Technical issues are diagnosed and resolved using available logging and monitoring
- **Deployment success**: Pi deployment process completes successfully with minimal manual intervention

## Implementation roadmap

### Phase 1: MVP foundation (Current scope)

- **Week 1-2**: Mock server implementation and WebSocket protocol establishment
- **Week 3-4**: Python client development with audio recording and hotkey management
- **Week 5-6**: Text insertion system with Accessibility API and AppleScript fallback
- **Week 7-8**: Pi server deployment with faster-whisper integration and testing
- **Week 9-10**: Integration testing, performance optimization, and documentation completion

### Phase 2: Performance optimization (Future)

- **WebRTC migration**: Replace WebSocket with WebRTC data channels for sub-250ms latency
- **Opus codec integration**: Implement real-time audio streaming with optimal compression
- **Advanced VAD**: Enhanced voice activity detection for improved processing efficiency
- **Network optimization**: STUN/TURN server support and adaptive quality management

### Phase 3: Native experience (Future)

- **Swift macOS application**: Native performance and background operation capabilities
- **Menu bar integration**: System tray controls and status indicators
- **App Store preparation**: Sandboxing compatibility and distribution optimization
- **Performance profiling**: Native audio engine integration for minimal latency

### Phase 4: Advanced features (Future)

- **Multi-language support**: Expand beyond English with dynamic language switching
- **Custom vocabulary**: Domain-specific transcription accuracy improvements
- **Multiple model support**: Specialized models for different use cases and contexts
- **Advanced configuration**: User-customizable settings for power users and specific workflows
