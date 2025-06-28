# Pi-Whispr WebSocket Protocol Schema

This document describes the WebSocket protocol schema implemented for Pi-Whispr communication between the macOS client and Raspberry Pi server.

## Overview

The protocol uses structured JSON messaging with reliable ordering through sequence identifiers. All messages follow a consistent format with a header containing metadata and a payload containing the actual data.

## Message Structure

### Base Message Format

```json
{
  "header": {
    "type": "message_type",
    "sequence_id": 123,
    "timestamp": 1672531200.123,
    "client_id": "client-uuid",
    "session_id": "session-uuid",
    "priority": "normal",
    "ttl": 10.0,
    "correlation_id": "correlation-uuid"
  },
  "payload": {
    // Message-specific data
  }
}
```

### Header Fields

- **type** (required): Message type identifier
- **sequence_id** (required): Auto-incrementing sequence number for ordering
- **timestamp** (required): Unix timestamp when message was created
- **client_id** (required): Unique identifier for the client
- **session_id** (optional): Session identifier for grouping related messages
- **priority** (optional): Message priority (low, normal, high, critical)
- **ttl** (optional): Time-to-live in seconds
- **correlation_id** (optional): For request-response correlation

## Message Types

### Connection Management

- `connect` - Initial connection establishment
- `connect_ack` - Connection acknowledgment
- `disconnect` - Connection termination
- `ping` - Connection health check
- `pong` - Ping response

### Client Management

- `client_register` - Register client with server
- `client_unregister` - Unregister client
- `client_list_request` - Request list of connected clients
- `client_list_response` - Response with client list
- `client_status_update` - Update client status

### Audio Processing

- `audio_start` - Begin audio session
- `audio_data` - Audio chunk data
- `audio_end` - End audio session
- `audio_config` - Audio configuration
- `audio_resume` - Resume audio processing
- `audio_pause` - Pause audio processing

### Transcription Results

- `transcription_result` - Complete transcription result
- `transcription_partial` - Partial transcription result
- `transcription_error` - Transcription processing error
- `transcription_progress` - Processing progress update

### Status and Monitoring

- `status_request` - Request system status
- `status_response` - System status response
- `health_check` - Health check request
- `health_response` - Health check response

### Performance Tracking

- `performance_metrics` - System performance data
- `latency_test` - Latency measurement test
- `latency_response` - Latency test response
- `bandwidth_test` - Bandwidth measurement test

### Error Handling

- `error` - Error notification
- `warning` - Warning notification
- `ack` - Message acknowledgment
- `nack` - Negative acknowledgment

## Payload Schemas

### Audio Data Payload

```json
{
  "audio_data": "base64_encoded_audio",
  "chunk_index": 1,
  "is_final": false,
  "timestamp_offset": 0.02,
  "energy_level": 0.75
}
```

### Transcription Result Payload

```json
{
  "text": "Hello world",
  "confidence": 0.95,
  "processing_time": 1.2,
  "model_used": "tiny.en",
  "language": "en",
  "word_timestamps": [
    { "word": "Hello", "start": 0.0, "end": 0.5 },
    { "word": "world", "start": 0.6, "end": 1.0 }
  ],
  "audio_duration": 2.0,
  "is_partial": false
}
```

### Performance Metrics Payload

```json
{
  "latency_ms": 150.5,
  "throughput_mbps": 10.2,
  "cpu_usage": 45.0,
  "memory_usage": 60.5,
  "temperature": 42.0,
  "network_quality": 0.95,
  "processing_queue_size": 3,
  "error_count": 0,
  "uptime_seconds": 3600.0
}
```

### Client Info Payload

```json
{
  "client_name": "Pi-Whispr macOS Client",
  "client_version": "1.0.0",
  "platform": "macOS",
  "capabilities": ["audio_recording", "text_insertion", "vad"],
  "status": "connected",
  "last_seen": 1672531200.123
}
```

### Error Payload

```json
{
  "error_code": "AUDIO_001",
  "error_message": "Failed to initialize audio device",
  "error_details": {
    "device_id": "default",
    "sample_rate": 16000
  },
  "recoverable": true,
  "suggested_action": "Check microphone permissions"
}
```

## Usage Examples

### Connection Flow

1. Client sends `connect` message with client info
2. Server responds with `connect_ack`
3. Regular `ping`/`pong` for health monitoring

### Audio Transcription Flow

1. Client sends `audio_start` with configuration
2. Client streams `audio_data` chunks
3. Client sends `audio_end` when complete
4. Server responds with `transcription_result`

### Error Handling

1. Any message can generate an `error` response
2. Errors include recovery suggestions
3. ACK/NACK messages confirm receipt

## Implementation Classes

The protocol is implemented in `shared/protocol.py` with the following key classes:

- `MessageHeader` - Standard message header
- `WebSocketMessage` - Base message class with JSON serialization
- `MessageBuilder` - Helper for creating typed messages
- `MessageValidator` - Validates message structure and content
- Various payload classes for type safety

## Validation

All messages are validated for:

- Required header fields
- Valid message types
- Proper sequence IDs
- TTL expiration
- Message-specific payload requirements

## Legacy Compatibility

The protocol maintains backward compatibility with existing `TranscriptionResult` and `AudioConfig` classes for smooth migration.
