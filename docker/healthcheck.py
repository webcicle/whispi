#!/usr/bin/env python3
"""
Health check script for Docker containers
"""

import sys
import socket


def check_health():
    """Check if the WebSocket server is responding"""
    try:
        # Try to connect to the WebSocket port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 8765))
        sock.close()
        
        if result == 0:
            print("Health check passed: WebSocket server is running")
            return 0
        else:
            print("Health check failed: Cannot connect to WebSocket server")
            return 1
            
    except Exception as e:
        print(f"Health check error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(check_health()) 