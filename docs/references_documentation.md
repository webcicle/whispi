# Complete References and Documentation Links

## Core Technologies and Libraries

### Speech Recognition Models

1. **OpenAI Whisper** (Original)
   - GitHub: https://github.com/openai/whisper
   - Paper: "Robust Speech Recognition via Large-Scale Weak Supervision"
   - Documentation: https://openai.com/research/whisper

2. **faster-whisper** (Primary choice)
   - GitHub: https://github.com/SYSTRAN/faster-whisper
   - PyPI: https://pypi.org/project/faster-whisper/
   - Based on CTranslate2: https://github.com/OpenNMT/CTranslate2

3. **whisper.cpp** (Alternative)
   - GitHub: https://github.com/ggerganov/whisper.cpp
   - Performance discussions: https://github.com/ggerganov/whisper.cpp/discussions/166
   - Pi 4 benchmarks: https://github.com/ggerganov/whisper.cpp/issues/599

### Python Libraries

#### Audio Processing
4. **PyAudio** (Audio I/O)
   - Documentation: https://people.csail.mit.edu/hubert/pyaudio/docs/
   - PyPI: https://pypi.org/project/PyAudio/
   - PortAudio (dependency): http://www.portaudio.com/

5. **WebRTCVAD** (Voice Activity Detection)
   - GitHub: https://github.com/wiseman/py-webrtcvad
   - PyPI: https://pypi.org/project/webrtcvad/
   - Google WebRTC project: https://webrtc.org/

6. **NumPy** (Audio processing)
   - Documentation: https://numpy.org/doc/stable/
   - PyPI: https://pypi.org/project/numpy/

7. **SciPy** (Signal processing)
   - Documentation: https://docs.scipy.org/doc/scipy/
   - PyPI: https://pypi.org/project/scipy/

#### Network Communication
8. **websockets** (WebSocket client/server)
   - Documentation: https://websockets.readthedocs.io/
   - PyPI: https://pypi.org/project/websockets/
   - GitHub: https://github.com/python-websockets/websockets

9. **asyncio** (Async programming)
   - Documentation: https://docs.python.org/3/library/asyncio.html
   - Tutorial: https://realpython.com/async-io-python/

#### Input/Output
10. **pynput** (Global hotkeys)
    - Documentation: https://pynput.readthedocs.io/
    - PyPI: https://pypi.org/project/pynput/
    - GitHub: https://github.com/moses-palmer/pynput

11. **pyobjc** (macOS integration)
    - Documentation: https://pyobjc.readthedocs.io/
    - PyPI Framework packages:
      - https://pypi.org/project/pyobjc-framework-Cocoa/
      - https://pypi.org/project/pyobjc-framework-ApplicationServices/

### Raspberry Pi Resources

12. **Raspberry Pi 5 Official Documentation**
    - Getting started: https://www.raspberrypi.com/documentation/computers/getting-started.html
    - Bookworm OS: https://www.raspberrypi.com/news/bookworm-the-new-version-of-raspberry-pi-os/

13. **Pi 5 Performance Benchmarks**
    - Official benchmarks: https://www.raspberrypi.com/news/benchmarking-raspberry-pi-5/
    - Thermal management: https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/

14. **Python Virtual Environments on Bookworm**
    - PEP 668: https://peps.python.org/pep-0668/
    - Debian documentation: https://www.debian.org/releases/bookworm/

### macOS Development

#### Swift and Xcode
15. **Swift Documentation**
    - Swift.org: https://swift.org/documentation/
    - Apple Swift Guide: https://docs.swift.org/swift-book/

16. **AVAudioEngine** (Audio recording)
    - Documentation: https://developer.apple.com/documentation/avfaudio/avaudioengine
    - WWDC 2019 session: https://developer.apple.com/videos/play/wwdc2019/510/

17. **Accessibility API** (Text insertion)
    - Documentation: https://developer.apple.com/documentation/applicationservices/accessibility_constants
    - Carbon framework: https://developer.apple.com/documentation/carbon

18. **Global Event Monitoring**
    - NSEvent documentation: https://developer.apple.com/documentation/appkit/nsevent
    - Key codes reference: https://stackoverflow.com/questions/3202629/where-can-i-find-a-list-of-mac-virtual-key-codes

#### macOS Automation
19. **AppleScript**
    - Documentation: https://developer.apple.com/library/archive/documentation/AppleScript/
    - System Events: https://developer.apple.com/library/archive/documentation/LanguagesUtilities/Conceptual/MacAutomationScriptingGuide/

20. **Hammerspoon** (Alternative automation)
    - Documentation: https://www.hammerspoon.org/docs/
    - Getting started: https://www.hammerspoon.org/go/

### Network and Audio Protocols

21. **WebRTC** (Real-time communication)
    - Official documentation: https://webrtc.org/getting-started/
    - MDN WebRTC API: https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API

22. **Opus Codec** (Audio compression)
    - Official site: https://opus-codec.org/
    - RFC 6716: https://tools.ietf.org/html/rfc6716
    - Wikipedia: https://en.wikipedia.org/wiki/Opus_(audio_format)

23. **WebSocket Protocol**
    - RFC 6455: https://tools.ietf.org/html/rfc6455
    - MDN documentation: https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API

### Audio Processing and Optimization

24. **Audio Sample Rates and Whisper**
    - Whisper optimal settings: https://github.com/openai/whisper/discussions/870
    - Audio preprocessing: https://huggingface.co/blog/fine-tune-whisper

25. **Voice Activity Detection**
    - WebRTC VAD paper: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/vad/
    - Implementation details: https://github.com/dpirch/libfvad

26. **Digital Signal Processing**
    - Butterworth filters: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    - Audio noise reduction: https://noisereduce.readthedocs.io/

## Performance and Optimization References

27. **ARM CPU Optimization**
    - ARM Neon SIMD: https://developer.arm.com/architectures/instruction-sets/simd-isas/neon
    - Pi 5 BCM2712 specs: https://www.raspberrypi.com/documentation/computers/processors.html

28. **CTranslate2 Optimization**
    - Performance guide: https://opennmt.net/CTranslate2/performance.html
    - Quantization: https://opennmt.net/CTranslate2/quantization.html

29. **Python Performance**
    - asyncio best practices: https://docs.python.org/3/library/asyncio-dev.html
    - Threading vs async: https://realpython.com/python-concurrency/

## System Administration and Deployment

30. **systemd Services**
    - systemd documentation: https://www.freedesktop.org/software/systemd/man/systemd.service.html
    - Pi service setup: https://www.raspberrypi.com/documentation/computers/using_linux.html

31. **Linux Audio System**
    - ALSA documentation: https://www.alsa-project.org/wiki/Main_Page
    - PulseAudio: https://www.freedesktop.org/wiki/Software/PulseAudio/

32. **Network Configuration**
    - Pi networking: https://www.raspberrypi.com/documentation/computers/configuration.html
    - UFW firewall: https://help.ubuntu.com/community/UFW

## API and Integration References

33. **OpenAI Whisper API** (Comparison/fallback)
    - API documentation: https://platform.openai.com/docs/guides/speech-to-text
    - Pricing: https://openai.com/pricing
    - API reference: https://platform.openai.com/docs/api-reference/audio

34. **Alternative Speech APIs** (For comparison)
    - Google Speech-to-Text: https://cloud.google.com/speech-to-text/docs
    - Azure Speech: https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/
    - AWS Transcribe: https://docs.aws.amazon.com/transcribe/

## Hardware and Electronics

35. **Raspberry Pi 5 Specifications**
    - Technical specifications: https://www.raspberrypi.com/products/raspberry-pi-5/
    - GPIO pinout: https://pinout.xyz/

36. **Active Cooling Solutions**
    - Official Active Cooler: https://www.raspberrypi.com/products/active-cooler/
    - Thermal testing: https://www.jeffgeerling.com/blog/2023/raspberry-pi-5-cooling-solutions

## Community Resources and Discussions

37. **Reddit Communities**
    - r/raspberry_pi: https://www.reddit.com/r/raspberry_pi/
    - r/MachineLearning: https://www.reddit.com/r/MachineLearning/
    - r/LocalLLaMA: https://www.reddit.com/r/LocalLLaMA/

38. **Stack Overflow Tags**
    - [raspberry-pi]: https://stackoverflow.com/questions/tagged/raspberry-pi
    - [speech-recognition]: https://stackoverflow.com/questions/tagged/speech-recognition
    - [macos]: https://stackoverflow.com/questions/tagged/macos
    - [websocket]: https://stackoverflow.com/questions/tagged/websocket

39. **GitHub Discussions and Issues**
    - Whisper.cpp Pi discussions: https://github.com/ggerganov/whisper.cpp/discussions
    - faster-whisper issues: https://github.com/SYSTRAN/faster-whisper/issues

## Academic and Research Papers

40. **Speech Recognition Research**
    - "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper paper)
    - ArXiv: https://arxiv.org/abs/2212.04356

41. **Audio Processing Research**
    - "Deep Speech" by Baidu: https://arxiv.org/abs/1412.5567
    - "Wav2Vec 2.0" by Facebook: https://arxiv.org/abs/2006.11477

## Tools and Utilities

42. **Audio Testing Tools**
    - Audacity: https://www.audacityteam.org/
    - FFmpeg: https://ffmpeg.org/documentation.html
    - SoX: http://sox.sourceforge.net/

43. **Network Testing Tools**
    - iperf3: https://iperf.fr/
    - Wireshark: https://www.wireshark.org/docs/
    - netcat: https://nc110.sourceforge.io/

44. **Performance Monitoring**
    - htop: https://htop.dev/
    - iotop: https://github.com/Tomas-M/iotop
    - nethogs: https://github.com/raboof/nethogs

## Package Managers and Installation

45. **Python Package Management**
    - pip documentation: https://pip.pypa.io/en/stable/
    - PyPI: https://pypi.org/
    - virtualenv: https://virtualenv.pypa.io/en/latest/

46. **macOS Package Management**
    - Homebrew: https://brew.sh/
    - MacPorts: https://www.macports.org/

47. **Linux Package Management**
    - APT documentation: https://debian-handbook.info/browse/stable/apt.html
    - Debian packages: https://packages.debian.org/

## Security and Privacy

48. **macOS Security Framework**
    - App Sandbox: https://developer.apple.com/documentation/security/app_sandbox
    - Entitlements: https://developer.apple.com/documentation/bundleresources/entitlements

49. **Network Security**
    - TLS/SSL documentation: https://tools.ietf.org/html/rfc8446
    - WebSocket security: https://tools.ietf.org/html/rfc6455#section-10

## Version Control and Collaboration

50. **Git and GitHub**
    - Git documentation: https://git-scm.com/doc
    - GitHub guides: https://guides.github.com/

## Troubleshooting Resources

51. **Pi Troubleshooting**
    - Official troubleshooting: https://www.raspberrypi.com/documentation/computers/troubleshooting.html
    - Pi forums: https://forums.raspberrypi.com/

52. **macOS Troubleshooting**
    - Apple Developer Forums: https://developer.apple.com/forums/
    - Console.app logs: https://developer.apple.com/documentation/os/logging

## Alternative Implementations and Inspiration

53. **Similar Projects**
    - Talon Voice: https://talonvoice.com/
    - Dictation-toolbox: https://github.com/dictation-toolbox
    - Whisper real-time examples: https://github.com/topics/whisper-realtime

54. **Voice Assistant Projects**
    - Mycroft: https://mycroft.ai/
    - Rhasspy: https://rhasspy.readthedocs.io/
    - Snips (archived): https://github.com/snipsco

## Additional Learning Resources

55. **Python Audio Programming**
    - "Python for Audio" tutorials: https://python-sounddevice.readthedocs.io/
    - Audio programming with Python: https://realpython.com/playing-and-recording-sound-python/

56. **Swift Audio Programming**
    - AVAudioEngine tutorial: https://www.raywenderlich.com/5154-avaudioengine-tutorial-for-ios-getting-started
    - Core Audio: https://developer.apple.com/documentation/coreaudio

57. **WebSocket Programming**
    - WebSocket tutorial: https://javascript.info/websocket
    - Real-time web apps: https://socket.io/docs/v4/

## License and Legal Information

58. **Open Source Licenses**
    - MIT License: https://opensource.org/licenses/MIT
    - Apache 2.0: https://opensource.org/licenses/Apache-2.0
    - GPL v3: https://www.gnu.org/licenses/gpl-3.0.html

59. **Model Licenses**
    - Whisper model license: https://github.com/openai/whisper/blob/main/LICENSE
    - Hugging Face model hub: https://huggingface.co/models

## Performance Benchmarking References

60. **Speech Recognition Benchmarks**
    - Common Voice dataset: https://commonvoice.mozilla.org/
    - LibriSpeech: http://www.openslr.org/12/
    - Performance metrics: https://paperswithcode.com/task/speech-recognition

---

## Citation Format

If you need to cite this work or the underlying technologies:

```
@software{whisper2023,
  title={Whisper: Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  year={2023},
  url={https://github.com/openai/whisper}
}

@software{faster_whisper2023,
  title={faster-whisper: Fast and efficient speech recognition with CTranslate2},
  author={SYSTRAN},
  year={2023},
  url={https://github.com/SYSTRAN/faster-whisper}
}
```

## Quick Reference Links for Implementation

- **Primary model**: https://github.com/SYSTRAN/faster-whisper
- **WebSocket library**: https://websockets.readthedocs.io/
- **Audio library**: https://people.csail.mit.edu/hubert/pyaudio/docs/
- **Pi 5 documentation**: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html
- **macOS accessibility**: https://developer.apple.com/documentation/applicationservices
- **Systemd services**: https://www.freedesktop.org/software/systemd/man/systemd.service.html

This comprehensive reference list covers all the technologies, libraries, documentation, and resources used in creating your speech-to-text system. Each link provides the official documentation or authoritative source for the respective technology.