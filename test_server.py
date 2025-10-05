#!/usr/bin/env python3
"""
Test script for the TTS server
"""

import requests
import json
import sys
import time


def test_server():
    base_url = "http://localhost:8000"

    # Test health endpoint
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the server is running on http://localhost:8000")
        return False

    # Test TTS endpoint
    print("🎤 Testing TTS endpoint...")
    test_request = {
        "model": "tts-1",
        "input": "The weather today is sunny with a chance of rain in the evening. Remember to carry an umbrella!",
        "voice": "alloy",
        "response_format": "mp3",
        "speed": 1.0,
    }

    try:
        print("📤 Sending TTS request...")
        response = requests.post(
            f"{base_url}/v1/audio/speech",
            headers={"Content-Type": "application/json"},
            json=test_request,
            timeout=60,  # TTS can take a while
        )

        if response.status_code == 200:
            # Save audio file
            with open("test_output.mp3", "wb") as f:
                f.write(response.content)
            print("✅ TTS request successful!")
            print("🔊 Audio saved to test_output.mp3")
            print(f"📊 Audio file size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ TTS request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ TTS request failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing OpenAI-compatible TTS Server...")
    print("Make sure the server is running first (python tts_server.py)")
    print()

    if test_server():
        print()
        print("🎉 All tests passed!")
        print("Your TTS server is working correctly.")
    else:
        print()
        print("❌ Tests failed. Please check the server logs for errors.")
        sys.exit(1)
