# M2 Voice Cloning Tool  

## Overview  
This is a real-time voice cloning tool optimized for Mac M2 chips. It supports voice cloning from a sample file and allows text-to-speech (TTS) generation via the command line. The system is accelerated using **Core ML** and **Metal**, ensuring fast and high-quality speech synthesis.  

## Features  
- **Supports Core ML for high-speed inference**  
- **Uses PyTorch Metal acceleration as a fallback**  
- **Real-time TTS generation with low latency**  
- **Streams generated audio to WeChat (via BlackHole virtual audio device)**  
- **Multi-threaded processing for improved performance**  

## Installation  

### Requirements  
- macOS with Apple Silicon (M1/M2)  
- Python 3.8+  
- PyTorch with Metal support  
- Core ML Tools  
- `sounddevice` for audio playback  
- `BlackHole 2ch` for virtual audio output  

### Install Dependencies  
```bash
pip install torch sounddevice coremltools TTS numpy
```

## Usage  

### Run the Program  
```bash
python3 main.py --voice path/to/your/voice_sample.wav
```
Then enter text in the command line to generate speech.  

### Example  
```
> 你好，这是一个语音克隆测试
🕒 生成耗时: 350ms
🎤 音频已播放
```

## Notes  
- If Core ML models are available, they will be used for maximum performance. Otherwise, PyTorch Metal acceleration is used.  
- Ensure `BlackHole 2ch` is installed and set as an audio output device.  

## Author  
[PStarH](https://github.com/PStarH)
