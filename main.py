#!/usr/bin/env python3
"""
Mac M2 终极语音克隆工具 - 支持微信实时语音
版本：2.2 (性能优化版)
"""

import os
import sys
import time
import argparse
import numpy as np
import sounddevice as sd
from pathlib import Path
import coremltools as ct
import torch
import torch.mps
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty

# 配置区
MODEL_NAME = "zh-CN/baker/tacotron2-DDC-GST"   # 苹果优化模型
CORE_ML_MODEL = "tts_model.mlpackage"          # 转换后的 Core ML 模型
SAMPLE_RATE = 24000                            # 苹果音频硬件采样率
MAX_TEXT_LENGTH = 100                          # 最大输入字符数
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 用于队列缓冲
QUEUE_TIMEOUT = 0.1  # 队列取数据超时时间

class M2TTSEngine:
    def __init__(self, voice_sample: str):
        self.voice_sample = voice_sample
        self._setup_hardware()
        self._load_model()
        self._setup_audio()
        self._warmup()

    def _setup_hardware(self):
        """设置 M2 芯片参数"""
        torch.mps.set_per_process_memory_fraction(0.7)
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    def _load_model(self):
        """加载 Core ML 模型，若不存在则使用 PyTorch 作为后备"""
        try:
            if Path(CORE_ML_MODEL).exists():
                self.model = ct.models.MLModel(
                    CORE_ML_MODEL,
                    compute_units=ct.ComputeUnit.ALL
                )
                print("✅ Core ML 模型加载成功")
            else:
                print("⚠️ 未找到 Core ML 模型，使用 Metal 加速的 PyTorch")
                from TTS.api import TTS
                self.tts = TTS(
                    model_name=MODEL_NAME,
                    vocoder_name="zh-CN/baker/hifigan_simple",
                    progress_bar=False
                ).to(DEVICE)
        except Exception as e:
            sys.exit(f"❌ 模型加载失败: {e}")

    def _setup_audio(self):
        """检查音频设备设置"""
        try:
            sd.check_output_settings(device="BlackHole 2ch")
        except sd.PortAudioError:
            sys.exit("❌ 请安装 BlackHole 并配置聚合设备")

    def _warmup(self):
        """预热模型，减少首次生成延时"""
        start = time.monotonic()
        self.generate("预热模型")
        print(f"🔥 预热完成 (耗时 {time.monotonic()-start:.1f}s)")

    def _coreml_inference(self, text: str) -> np.ndarray:
        """使用 Core ML 推理"""
        inputs = {"text": self._preprocess_text(text)}
        mel = self.model.predict(inputs)["mel"]
        return self._vocoder(mel)

    def _pytorch_inference(self, text: str) -> np.ndarray:
        """使用 PyTorch 推理"""
        with torch.autocast(DEVICE), torch.no_grad():
            return self.tts.tts(
                text=text,
                speaker_wav=self.voice_sample,
                speed=1.2
            )

    def _preprocess_text(self, text: str) -> np.ndarray:
        """将文本转换为数值序列"""
        return np.array([ord(c) for c in text], dtype=np.float32)

    def _vocoder(self, mel: np.ndarray) -> np.ndarray:
        """调用声码器生成音频"""
        tensor = torch.from_numpy(mel).to(DEVICE)
        with torch.inference_mode():
            return self.tts.vocoder.decode(tensor).cpu().numpy()

    def generate(self, text: str) -> np.ndarray:
        """生成语音，自动选择推理后端"""
        if len(text) > MAX_TEXT_LENGTH:
            print(f"⚠️ 文本过长 (最大 {MAX_TEXT_LENGTH} 字符)")
            return None

        try:
            start_time = time.monotonic()
            if hasattr(self, "model"):
                audio = self._coreml_inference(text)
            else:
                audio = self._pytorch_inference(text)
            print(f"🕒 生成耗时: {(time.monotonic()-start_time)*1000:.0f}ms")
            return audio.astype(np.float32)
        except torch.mps.OutOfMemoryError:
            print("⚠️ GPU 内存不足，释放缓存后重试")
            torch.mps.empty_cache()
            return self.generate(text)

    def stream_to_wechat(self, audio: np.ndarray):
        """播放生成的音频到微信设备"""
        try:
            sd.play(audio, SAMPLE_RATE, device="BlackHole 2ch")
            sd.wait()
        except sd.PortAudioError as e:
            print(f"⚠️ 音频播放失败: {e}")

class CLIInterface:
    def __init__(self, engine: M2TTSEngine):
        self.engine = engine
        self.task_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)

    def run(self):
        """启动输入监听与任务处理线程"""
        print("\n🎤 输入文本 (Enter 发送 | Ctrl+C 退出):")
        # 启动任务处理线程
        self.executor.submit(self._process_tasks)

        try:
            while True:
                text = input("> ").strip()
                if text:
                    self.task_queue.put(text)
        except KeyboardInterrupt:
            print("\n🛑 正在退出...")
            self.executor.shutdown(wait=True)
            sys.exit()

    def _process_tasks(self):
        """从队列中取文本生成语音并播放"""
        while True:
            try:
                text = self.task_queue.get(timeout=QUEUE_TIMEOUT)
                audio = self.engine.generate(text)
                if audio is not None:
                    # 播放音频也放到线程池执行，避免阻塞生成流程
                    self.executor.submit(self.engine.stream_to_wechat, audio)
            except Empty:
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2 语音克隆工具")
    parser.add_argument("--voice", required=True, help="语音样本文件路径")
    args = parser.parse_args()

    if not Path(args.voice).exists():
        sys.exit(f"❌ 语音文件 {args.voice} 不存在")

    engine = M2TTSEngine(args.voice)
    cli = CLIInterface(engine)
    cli.run()
