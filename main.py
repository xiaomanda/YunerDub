import os
import json
import sys
import tempfile
import subprocess
import re
import hashlib
from dataclasses import dataclass, asdict, field
from shutil import which
from typing import List, Optional

import sounddevice as sd
import soundfile as sf
import numpy as np

from PySide6.QtCore import Qt, QUrl, QTimer, QSize, QSignalBlocker, QThread, Signal
from PySide6.QtGui import QAction, QColor, QBrush, QTextCursor, QTextBlockFormat
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QListWidget,
    QListWidgetItem, QFileDialog, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSlider, QTextEdit, QSplitter, QAbstractItemView, QComboBox, QSizePolicy,
    QGroupBox, QSizePolicy, QStyledItemDelegate, QStyleOptionViewItem
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget


LOOP_BACK_MARGIN_MS = 300 #AA

# ── TTS 配置 ──────────────────────────────────────────────────────────────────
# TTS 子进程使用专用 venv 的 Python（与 2.12 完全一致）。
# edge_tts 安装在这个独立 venv 里，该环境网络正常，与本文件运行的 venv 无关。
VENV_PYTHON = "/Volumes/Doc/16_coding(隐藏环境文件)/Edge TTS/.venv/bin/python"

EDGE_TTS_VOICES = [
    # ── 英文女声 ──────────────────────────────
    ("Jenny · 美式 · 温暖自然",        "en-US-JennyNeural"),
    ("Aria · 美式 · 轻松对话",         "en-US-AriaNeural"),
    ("Michelle · 美式 · 清晰友好",     "en-US-MichelleNeural"),
    ("Amber · 美式 · 柔和亲切",        "en-US-AmberNeural"),
    ("Sara · 美式 · 年轻活泼",         "en-US-SaraNeural"),
    ("Nancy · 美式 · 沉稳干练",        "en-US-NancyNeural"),
    ("Sonia · 英式 · 优雅正式",        "en-GB-SoniaNeural"),
    ("Libby · 英式 · 轻快现代",        "en-GB-LibbyNeural"),
    ("Natasha · 澳式 · 清晰自然",      "en-AU-NatashaNeural"),
    # ── 英文男声 ──────────────────────────────
    ("Guy · 美式 · 男声",              "en-US-GuyNeural"),
    ("Ryan · 英式 · 男声",             "en-GB-RyanNeural"),
    ("William · 澳式 · 男声",          "en-AU-WilliamNeural"),
    # ── 日文 ──────────────────────────────────
    ("Nanami · 日语女声 · 温柔自然",   "ja-JP-NanamiNeural"),
    ("Shiori · 日语女声 · 轻快明亮",   "ja-JP-ShioriNeural"),
    ("Aoi · 日语女声 · 清澈干净",      "ja-JP-AoiNeural"),
    ("Keita · 日语男声",               "ja-JP-KeitaNeural"),
]

# ── Whisper 配置 ──────────────────────────────────────────────────────────────
WHISPER_PYTHON = "/Volumes/Doc/16_coding(隐藏环境文件)/Faster-Whisper/.venv/bin/python"
WHISPER_MODEL  = "medium"


class TtsWorker(QThread):
    """后台线程：每行单独请求 TTS，拼接音频，时间戳精确对应每行。"""
    finished = Signal(str, str)
    error    = Signal(str)

    SAMPLE_RATE = 44100

    def __init__(self, lines: list, voice: str, output_path: str, timing_path: str):
        super().__init__()
        self.lines       = lines
        self.voice       = voice
        self.output_path = output_path
        self.timing_path = timing_path

    def run(self):
        try:
            ffmpeg = which("ffmpeg")

            # ── 第一阶段：子进程调用 edge_tts，逐行生成 mp3 文件 ──────────────
            # 与 2.12 方案一致：通过临时 JSON 传参 + 临时脚本，规避命令行特殊字符问题。
            # VENV_PYTHON = sys.executable，即本文件当前所在 venv 的解释器，
            # 该 venv（2.12 的运行环境）已安装可正常工作的 edge_tts，无网络问题。
            params_file = self.output_path + "_params.json"
            mp3_dir     = self.output_path + "_mp3s"
            os.makedirs(mp3_dir, exist_ok=True)

            with open(params_file, "w", encoding="utf-8") as pf:
                json.dump({
                    "lines":   self.lines,
                    "voice":   self.voice,
                    "mp3_dir": mp3_dir,
                }, pf, ensure_ascii=False)

            # 子进程脚本：逐行请求 TTS，每行输出一个编号 mp3 文件
            script = r"""
import asyncio, edge_tts, json, sys, os

async def _gen_line(text, voice):
    chunks = []
    communicate = edge_tts.Communicate(text, voice)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)

async def _gen_all(params_file):
    with open(params_file, "r", encoding="utf-8") as f:
        p = json.load(f)
    for i, line in enumerate(p["lines"]):
        mp3_data = await _gen_line(line, p["voice"])
        mp3_path = os.path.join(p["mp3_dir"], f"{i:04d}.mp3")
        with open(mp3_path, "wb") as f:
            f.write(mp3_data)

asyncio.run(_gen_all(sys.argv[1]))
"""
            script_file = self.output_path + "_gen.py"
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(script)

            ret = subprocess.run(
                [VENV_PYTHON, script_file, params_file],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            for fp in [script_file, params_file]:
                try:
                    os.remove(fp)
                except Exception:
                    pass

            if ret.returncode != 0:
                raise RuntimeError(ret.stderr.decode(errors="replace").strip())

            # ── 第二阶段：逐行解码 mp3 → numpy，计算时间戳，拼接音频 ───────────
            # 逻辑与原 4.3 完全一致，仅数据来源从内存改为读取上一阶段写出的文件。
            def mp3_bytes_to_wav_array(mp3_data: bytes) -> np.ndarray:
                """用 ffmpeg 把 mp3 字节转为 float32 numpy 数组（44100 Hz, mono）。"""
                if not ffmpeg:
                    raise FileNotFoundError("未找到 ffmpeg，无法解码 mp3")
                proc = subprocess.run(
                    [ffmpeg, "-y", "-f", "mp3", "-i", "pipe:0",
                     "-acodec", "pcm_f32le", "-ar", str(self.SAMPLE_RATE),
                     "-ac", "1", "-f", "f32le", "pipe:1"],
                    input=mp3_data,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                raw = proc.stdout
                if not raw:
                    return np.zeros(0, dtype=np.float32)
                return np.frombuffer(raw, dtype=np.float32).copy()

            timing         = []   # [{text, offset, duration}, ...]  100ns 单位
            segments       = []   # List[np.ndarray]
            cursor_samples = 0

            for i, line in enumerate(self.lines):
                mp3_path = os.path.join(mp3_dir, f"{i:04d}.mp3")
                with open(mp3_path, "rb") as f:
                    mp3_data = f.read()
                try:
                    os.remove(mp3_path)
                except Exception:
                    pass

                arr = mp3_bytes_to_wav_array(mp3_data)
                n   = len(arr)
                # offset / duration 用 100ns 单位（与原 edge-tts SentenceBoundary 一致）
                offset_ns = int(cursor_samples / self.SAMPLE_RATE * 10_000_000)
                dur_ns    = int(n / self.SAMPLE_RATE * 10_000_000)
                timing.append({
                    "text":     line,
                    "offset":   offset_ns,
                    "duration": dur_ns,
                })
                segments.append(arr)
                cursor_samples += n

            # 清理临时 mp3 目录
            try:
                os.rmdir(mp3_dir)
            except Exception:
                pass

            # 写 timing JSON
            with open(self.timing_path, "w", encoding="utf-8") as f:
                json.dump(timing, f, ensure_ascii=False)

            # 写拼接后的 WAV
            full_audio = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
            sf.write(self.output_path, full_audio, self.SAMPLE_RATE, subtype="PCM_16")

            self.finished.emit(self.output_path, self.timing_path)
        except Exception as e:
            self.error.emit(str(e))


class AsrWorker(QThread):
    """后台线程：直接 import faster_whisper 识别音频（打包版，无需子进程）。"""
    finished = Signal(str)
    progress = Signal(str)
    error    = Signal(str)

    def __init__(self, media_path: str, output_json: str, language: str = "en"):
        super().__init__()
        self.media_path  = media_path
        self.output_json = output_json
        self.language    = language

    def run(self):
        try:
            params_file = self.output_json + "_params.json"
            with open(params_file, "w", encoding="utf-8") as f:
                json.dump({
                    "media_path":  self.media_path,
                    "output_json": self.output_json,
                    "model":       WHISPER_MODEL,
                    "language":    self.language,
                }, f, ensure_ascii=False)

            script = r"""
import json, sys
from faster_whisper import WhisperModel

with open(sys.argv[1], "r", encoding="utf-8") as f:
    p = json.load(f)

model = WhisperModel(p["model"], device="cpu", compute_type="int8")
segments, info = model.transcribe(
    p["media_path"],
    language=p["language"],
    beam_size=5,
    vad_filter=True,
)

results = []
for seg in segments:
    text = seg.text.strip()
    if text:
        results.append({
            "start": round(seg.start, 3),
            "end":   round(seg.end,   3),
            "text":  text,
        })

with open(p["output_json"], "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
"""
            script_file = self.output_json + "_asr.py"
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(script)

            self.progress.emit("Whisper 模型加载中…")
            ret = subprocess.run(
                [WHISPER_PYTHON, script_file, params_file],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            for fp in [script_file, params_file]:
                try: os.remove(fp)
                except Exception: pass

            if ret.returncode != 0:
                raise RuntimeError(ret.stderr.decode(errors="replace").strip())
            if not os.path.exists(self.output_json):
                raise RuntimeError("识别结果文件未生成")

            self.finished.emit(self.output_json)
        except Exception as e:
            self.error.emit(str(e))


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str = ""
    audio_path: str = ""


@dataclass
class SubtitleEntry:
    start: float
    end: float
    text: str


@dataclass
class Project:
    media_path: str = ""
    subtitle_path: str = ""
    cut_points: List[float] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    subtitle_entries: List[SubtitleEntry] = field(default_factory=list)

    def save(self, path: str):
        data = {
            "media_path":      self.media_path,
            "subtitle_path":   self.subtitle_path,
            "cut_points":      self.cut_points,
            "segments":        [asdict(s) for s in self.segments],
            "subtitle_entries":[asdict(s) for s in self.subtitle_entries],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        proj = cls(
            media_path=data.get("media_path", ""),
            subtitle_path=data.get("subtitle_path", ""),
            cut_points=data.get("cut_points", [])
        )
        proj.segments        = [Segment(**s)       for s in data.get("segments", [])]
        proj.subtitle_entries= [SubtitleEntry(**s) for s in data.get("subtitle_entries", [])]
        return proj


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def parse_srt_time(t: str) -> float:
    hh, mm, rest = t.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_srt(path: str) -> List[SubtitleEntry]:
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()
    blocks = re.split(r"\n\s*\n", content.strip())
    items = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line  = lines[1] if "-->" in lines[1] else lines[0]
        text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
        if "-->" not in time_line:
            continue
        start_str, end_str = [x.strip() for x in time_line.split("-->")]
        text = " ".join(text_lines).strip()
        if text:
            items.append(SubtitleEntry(parse_srt_time(start_str), parse_srt_time(end_str), text))
    return items


def convert_to_wav_with_ffmpeg(input_path: str, output_path: str) -> bool:
    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("未找到 ffmpeg，请先安装 ffmpeg")
    cmd = [ffmpeg, "-y", "-i", input_path, "-vn",
           "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", output_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0


class Recorder:
    def __init__(self, samplerate=44100, channels=1):
        self.samplerate = samplerate
        self.channels   = channels
        self._stream    = None
        self._frames    = []
        self._recording = False

    def start(self):
        self._frames    = []
        self._recording = True

        def callback(indata, frames, time, status):
            if self._recording:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=self.channels, callback=callback)
        self._stream.start()

    def stop(self, output_path: str):
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._frames:
            return False
        audio = np.concatenate(self._frames, axis=0).astype(np.float32)
        audio = np.clip(audio * 1.25, -0.98, 0.98)
        sf.write(output_path, audio, self.samplerate)
        return True

    @property
    def is_recording(self):
        return self._recording


class VideoAspectWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:black;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget = QVideoWidget(self)
        self.video_widget.setStyleSheet("background:black;")
        self.video_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._video_size = QSize(0, 0)

    def set_video_size(self, width: int, height: int):
        if width > 0 and height > 0:
            self._video_size = QSize(width, height)
            self.relayout()

    def relayout(self):
        aw, ah = self.width(), self.height()
        if aw <= 0 or ah <= 0:
            return
        vw, vh = self._video_size.width(), self._video_size.height()
        if not self._video_size.isEmpty() and vw > 0 and vh > 0:
            scale    = min(aw / vw, ah / vh)
            target_w = max(1, int(vw * scale))
            target_h = max(1, int(vh * scale))
            x = max(0, (aw - target_w) // 2)
            y = max(0, (ah - target_h) // 2)
            self.video_widget.setGeometry(x, y, target_w, target_h)
        else:
            self.video_widget.setGeometry(0, 0, aw, ah)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.relayout()


HIGHLIGHT_ROLE = Qt.UserRole + 100


class HighlightDelegate(QStyledItemDelegate):
    """只负责绘制黄色背景，文字走默认渲染，不干预 state 标志。"""

    def paint(self, painter, option, index):
        if index.data(HIGHLIGHT_ROLE):
            painter.save()
            painter.fillRect(option.rect, QColor("#ffe58f"))
            painter.restore()
            # 让 super 画文字，但把背景设成透明避免覆盖我们画的底色
            opt = QStyleOptionViewItem(option)
            opt.backgroundBrush = QBrush(Qt.transparent)
            # 去掉选中/hover 标志，避免系统主题再盖一层蓝/绿
            from PySide6.QtWidgets import QStyle
            opt.state &= ~(QStyle.State_Selected | QStyle.State_MouseOver)
            super().paint(painter, opt, index)
        else:
            super().paint(painter, option, index)


class PlainPasteTextEdit(QTextEdit):
    """粘贴时自动去除富文本格式，只保留纯文字。"""

    def insertFromMimeData(self, source):
        if source.hasText():
            self.insertPlainText(source.text())
        else:
            super().insertFromMimeData(source)


class MainWindow(QMainWindow):
    def _apply_line_spacing(self, text_edit, factor=130):
        """统一设置文本框内文字行距为指定倍数（默认1.3倍）"""
        block_fmt = QTextBlockFormat()
        block_fmt.setLineHeight(float(factor), QTextBlockFormat.ProportionalHeight.value)
        cursor = text_edit.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.mergeBlockFormat(block_fmt)
        # 应用到文档默认格式，确保新输入的文字同样应用该行距
        default_cursor = QTextCursor(text_edit.document())
        default_cursor.select(QTextCursor.Document)
        default_cursor.mergeBlockFormat(block_fmt)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YunerDub")
        self.resize(1600, 980)
        self.setMinimumSize(800, 480)

        self.project = Project()
        self.current_project_path: Optional[str] = None
        self.cut_points: List[float] = []
        self.video_duration_ms = 0
        self.subtitle_entries: List[SubtitleEntry] = []

        self.selected_segment_id: Optional[int] = None
        self.selected_subtitle_index: Optional[int] = None
        self._current_subtitle_highlight_idx: Optional[int] = None

        self.recorder = Recorder()
        self.recording_segment_id: Optional[int] = None
        self.recording_output_path: Optional[str] = None

        self.player_mode = "idle"
        self.segment_loop_end_ms: Optional[int] = None
        self.compare_original_seg: Optional[Segment] = None

        self.audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)

        self.record_audio_output = QAudioOutput()
        self.record_audio_player = QMediaPlayer()
        self.record_audio_player.setAudioOutput(self.record_audio_output)

        self._segment_list_refreshing = False
        self._ui_lock = False
        self.segment_generate_mode = "cuts"

        self._tts_worker: Optional[TtsWorker] = None
        self._tts_audio_path: Optional[str] = None
        self._tts_lines: List[str] = []

        self._asr_worker: Optional[AsrWorker] = None

        # 左侧自由播放时的自动高亮跟踪（节流：上次高亮时的播放位置，单位 ms）
        self._playback_auto_highlight_ms: int = -9999

        self._build_ui()
        self._build_menu()
        self._build_player()

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(100)
        self.ui_timer.timeout.connect(self.update_time_label)

        self.record_audio_player.mediaStatusChanged.connect(self.on_record_audio_status_changed)

        self.update_generate_mode_ui()
        QTimer.singleShot(0, self.reset_splitter_equal)

    # ─────────────────────────────────────────────────────────────── UI build ──

    APP_STYLE = """
        QMainWindow { background: #f0f4f2; }
        QWidget#root { background: #f0f4f2; }
        QWidget#left_panel { background: #e8efec; border-right: 1px solid #c8d8d0; }
        QWidget#right_panel { background: #f0f4f2; }

        QLabel { color: #2d3d35; }
        QLabel#section_label {
            color: #2a7a5a; font-size: 10px; font-weight: 700;
            letter-spacing: 2px; padding: 2px 0;
        }
        QLabel#time_label { color: #4db890; font-size: 12px; font-family: monospace; }
        QLabel#status_label {
            color: #2d3d35; font-size: 12px;
            padding: 5px 12px; background: #ddeee6;
            border-radius: 6px; border: 1px solid #c8d8d0;
        }
        QLabel#tts_status { color: #7aaa90; font-size: 11px; padding: 2px 0; }

        QSlider::groove:horizontal { height: 3px; background: #c8d8d0; border-radius: 2px; }
        QSlider::handle:horizontal {
            width: 12px; height: 12px; margin: -5px 0;
            background: #2a9d6e; border-radius: 6px;
        }
        QSlider::sub-page:horizontal { background: #4db890; border-radius: 2px; }

        QListWidget {
            background: #ffffff; border: 1px solid #c8d8d0;
            border-radius: 8px; color: #2d3d35;
            font-size: 12px; padding: 4px; outline: none;
        }
        QListWidget::item { padding: 2px 8px; border-radius: 5px; margin: 1px 0; }
        QListWidget::item:hover { background: #e8f5ee; }
        QListWidget::item:selected { background: #d0ede0; color: #1a5c3a; }
        QListWidget#subtitle_list::item:selected { background: transparent; }

        QTextEdit {
            background: #ffffff; border: 1px solid #c8d8d0;
            border-radius: 8px; color: #2d3d35;
            font-size: 13px; padding: 6px;
            selection-background-color: #d0ede0;
        }
        QTextEdit:focus { border: 1px solid #4db890; }

        QComboBox {
            background: #f5faf7; border: 1px solid #c8d8d0;
            border-radius: 6px; color: #2d3d35;
            padding: 5px 10px; font-size: 12px;
        }
        QComboBox:hover { border-color: #4db890; }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background: #f5faf7; border: 1px solid #c8d8d0;
            color: #2d3d35; selection-background-color: #d0ede0;
        }

        QGroupBox {
            color: #2a7a5a; font-size: 10px; font-weight: 700;
            letter-spacing: 2px; border: 1px solid #c8d8d0;
            border-radius: 10px; margin-top: 10px; padding-top: 10px;
            background: #f5faf7;
        }
        QGroupBox::title {
            subcontrol-origin: margin; subcontrol-position: top left;
            padding: 0 6px; left: 12px;
        }

        QSplitter::handle { background: #c8d8d0; width: 1px; }

        QMenuBar {
            background: #e8efec; color: #2d3d35;
            border-bottom: 1px solid #c8d8d0;
            font-size: 13px; padding: 2px;
        }
        QMenuBar::item:selected { background: #d0ede0; border-radius: 4px; }
        QMenu {
            background: #f5faf7; border: 1px solid #c8d8d0;
            color: #2d3d35; font-size: 13px;
        }
        QMenu::item:selected { background: #d0ede0; }

        QScrollBar:vertical { background: transparent; width: 5px; }
        QScrollBar::handle:vertical {
            background: #b0ccbe; border-radius: 3px; min-height: 30px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

        /* ── 所有按钮统一基色，用透明度和边框区分层级 ── */
        QPushButton {
            background: #f0f7f3; color: #2d3d35;
            border: 1px solid #c8d8d0; border-radius: 7px;
            padding: 6px 12px; font-size: 12px; font-weight: 500;
        }
        QPushButton:hover { background: #e0f0e8; border-color: #4db890; color: #1a5c3a; }
        QPushButton:pressed { background: #d0ede0; }
        QPushButton:disabled { background: #f0f4f2; color: #a0b8ac; border-color: #dce8e2; }

        /* 主操作：稍亮的背景 + 深绿文字 */
        QPushButton#btn_primary {
            background: #d0ede0; color: #1a5c3a;
            border-color: #4db890; font-weight: 600;
        }
        QPushButton#btn_primary:hover { background: #bde5d0; border-color: #2a9d6e; color: #0e3d26; }
        QPushButton#btn_primary:disabled { background: #f0f4f2; color: #a0b8ac; border-color: #dce8e2; }

        /* 强调操作：绿色边框 + 绿色文字 */
        QPushButton#btn_accent {
            background: #f0f7f3; color: #2a9d6e;
            border-color: #4db890; font-weight: 600;
        }
        QPushButton#btn_accent:hover { background: #e0f5eb; border-color: #2a9d6e; color: #1a7a50; }

        /* 危险操作：红色边框 + 红色文字 */
        QPushButton#btn_danger {
            background: #f0f7f3; color: #c0392b;
            border-color: #e8a090; font-weight: 600;
        }
        QPushButton#btn_danger:hover { background: #fdf0ee; border-color: #c0392b; color: #a02020; }

        QPushButton#btn_collapse {
            background: transparent; border: none;
            color: #2a7a5a; font-size: 12px; padding: 0;
        }
        QPushButton#btn_collapse:hover { color: #1a5c3a; background: #e0f0e8; border-radius: 4px; }
    """

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        self.setStyleSheet(self.APP_STYLE)

        self.splitter = QSplitter(Qt.Horizontal)

        # ── 左侧媒体区 ────────────────────────────────────────────────────────
        left = QWidget()
        left.setObjectName("left_panel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        self.video_widget = VideoAspectWidget()
        self.video_widget.setMinimumHeight(200)

        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.time_label.setObjectName("time_label")
        self.time_label.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.seek_video)

        self.status_label = QLabel("○ 未加载媒体")
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(Qt.AlignCenter)

        play_row = QHBoxLayout()
        play_row.setSpacing(6)
        self.btn_toggle_play = QPushButton("▶  播放")
        self.btn_toggle_play.setObjectName("btn_primary")
        self.btn_toggle_play.setFixedHeight(34)
        self.btn_add_cut = QPushButton("◆  切分")
        self.btn_add_cut.setFixedHeight(34)
        play_row.addWidget(self.btn_toggle_play, 2)
        play_row.addWidget(self.btn_add_cut, 1)

        gen_row = QHBoxLayout()
        gen_row.setSpacing(6)
        self.combo_generate_mode = QComboBox()
        self.combo_generate_mode.addItems(["切点模式", "字幕模式"])
        self.btn_rebuild = QPushButton("生成片段")
        self.btn_rebuild.setObjectName("btn_accent")
        self.btn_rebuild.setFixedHeight(34)
        gen_row.addWidget(self.combo_generate_mode, 1)
        gen_row.addWidget(self.btn_rebuild, 1)

        self.btn_toggle_play.clicked.connect(self.toggle_play_pause)
        self.btn_add_cut.clicked.connect(self.add_cut_point)
        self.btn_rebuild.clicked.connect(self.rebuild_segments_by_mode)
        self.combo_generate_mode.currentIndexChanged.connect(self.on_generate_mode_changed)

        left_layout.addWidget(self.video_widget, 1)
        left_layout.addWidget(self.slider)
        left_layout.addWidget(self.time_label)
        left_layout.addWidget(self.status_label)
        left_layout.addLayout(play_row)
        left_layout.addLayout(gen_row)

        # ── 右侧面板 ──────────────────────────────────────────────────────────
        right = QWidget()
        right.setObjectName("right_panel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        # ── Segments section (collapsible) ────────────────────────────────────
        seg_section = QWidget()
        seg_section.setObjectName("right_panel")
        seg_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        seg_section_layout = QVBoxLayout(seg_section)
        seg_section_layout.setContentsMargins(0, 0, 0, 0)
        seg_section_layout.setSpacing(4)

        seg_header = QHBoxLayout()
        seg_label = QLabel("SEGMENTS")
        seg_label.setObjectName("section_label")
        self.btn_collapse_seg = QPushButton("▾")
        self.btn_collapse_seg.setFixedSize(20, 20)
        self.btn_collapse_seg.setObjectName("btn_collapse")
        self.btn_collapse_seg.setToolTip("折叠/展开")
        seg_header.addWidget(seg_label)
        seg_header.addStretch()
        seg_header.addWidget(self.btn_collapse_seg)

        self._seg_body = QWidget()
        self._seg_body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        seg_body_layout = QVBoxLayout(self._seg_body)
        seg_body_layout.setContentsMargins(0, 0, 0, 0)
        seg_body_layout.setSpacing(4)

        self.segment_list = QListWidget()
        self.segment_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.segment_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.segment_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.segment_list.itemClicked.connect(self.on_segment_selected)

        self.text_editor = QTextEdit()
        self.text_editor.textChanged.connect(self.auto_save_segment_text)
        self.text_editor.setLineWrapMode(QTextEdit.WidgetWidth)
        self.text_editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_editor.setMinimumHeight(46)
        self.text_editor.setMaximumHeight(70)
        self._apply_line_spacing(self.text_editor, 130)

        seg_body_layout.addWidget(self.segment_list, 1)
        seg_body_layout.addWidget(self.text_editor)

        seg_section_layout.addLayout(seg_header)
        seg_section_layout.addWidget(self._seg_body)
        self.btn_collapse_seg.clicked.connect(lambda: self._toggle_section(seg_section, self._seg_body, self.btn_collapse_seg))

        # ── Recording section (collapsible) ───────────────────────────────────
        rec_section = QWidget()
        rec_section.setObjectName("right_panel")
        rec_section_layout = QVBoxLayout(rec_section)
        rec_section_layout.setContentsMargins(0, 0, 0, 0)
        rec_section_layout.setSpacing(4)

        rec_header = QHBoxLayout()
        rec_header_label = QLabel("RECORDING")
        rec_header_label.setObjectName("section_label")
        self.btn_collapse_rec = QPushButton("▾")
        self.btn_collapse_rec.setFixedSize(20, 20)
        self.btn_collapse_rec.setObjectName("btn_collapse")
        rec_header.addWidget(rec_header_label)
        rec_header.addStretch()
        rec_header.addWidget(self.btn_collapse_rec)

        self._rec_body = QWidget()
        rec_body_layout = QVBoxLayout(self._rec_body)
        rec_body_layout.setSpacing(6)
        rec_body_layout.setContentsMargins(10, 8, 10, 8)

        rec_row1 = QHBoxLayout()
        rec_row1.setSpacing(6)
        self.btn_stop_loop     = QPushButton("▪  停止循环")
        self.btn_record_toggle = QPushButton("⏺  开始录音")
        self.btn_record_toggle.setObjectName("btn_danger")
        self.btn_play_audio    = QPushButton("▷  播放录音")
        for b in [self.btn_stop_loop, self.btn_record_toggle, self.btn_play_audio]:
            b.setFixedHeight(32)
            rec_row1.addWidget(b)

        rec_row2 = QHBoxLayout()
        rec_row2.setSpacing(6)
        self.btn_play_compare = QPushButton("⇌  播放对比")
        self.btn_play_compare.setObjectName("btn_accent")
        self.btn_open_folder  = QPushButton("↗  录音文件夹")
        self.btn_merge_recordings = QPushButton("⊕  合并录音")
        self.btn_merge_recordings.setObjectName("btn_primary")
        for b in [self.btn_play_compare, self.btn_open_folder, self.btn_merge_recordings]:
            b.setFixedHeight(32)
            rec_row2.addWidget(b)

        rec_body_layout.addLayout(rec_row1)
        rec_body_layout.addLayout(rec_row2)

        rec_section_layout.addLayout(rec_header)
        rec_section_layout.addWidget(self._rec_body)
        self.btn_collapse_rec.clicked.connect(lambda: self._toggle_section(rec_section, self._rec_body, self.btn_collapse_rec))

        self.btn_stop_loop.clicked.connect(self.stop_segment_loop)
        self.btn_record_toggle.clicked.connect(self.toggle_recording)
        self.btn_play_audio.clicked.connect(self.play_selected_audio)
        self.btn_play_compare.clicked.connect(self.toggle_play_compare)
        self.btn_open_folder.clicked.connect(self.open_recording_folder)
        self.btn_merge_recordings.clicked.connect(self.merge_all_recordings)

        # ── Subtitles section (collapsible) ───────────────────────────────────
        sub_section = QWidget()
        sub_section.setObjectName("right_panel")
        sub_section_layout = QVBoxLayout(sub_section)
        sub_section_layout.setContentsMargins(0, 0, 0, 0)
        sub_section_layout.setSpacing(4)

        sub_header = QHBoxLayout()
        sub_header_label = QLabel("SUBTITLES")
        sub_header_label.setObjectName("section_label")
        self.btn_collapse_sub = QPushButton("▾")
        self.btn_collapse_sub.setFixedSize(20, 20)
        self.btn_collapse_sub.setObjectName("btn_collapse")
        sub_header.addWidget(sub_header_label)
        sub_header.addStretch()
        sub_header.addWidget(self.btn_collapse_sub)

        self._sub_body = QWidget()
        sub_body_layout = QVBoxLayout(self._sub_body)
        sub_body_layout.setContentsMargins(8, 8, 8, 6)
        sub_body_layout.setSpacing(4)

        self.subtitle_list = QListWidget()
        self.subtitle_list.setObjectName("subtitle_list")
        self.subtitle_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.subtitle_list.setFocusPolicy(Qt.NoFocus)
        self.subtitle_list.itemClicked.connect(self.on_subtitle_selected)
        self.subtitle_list.setWordWrap(True)
        self.subtitle_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.subtitle_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.subtitle_list.setMinimumHeight(60)
        self.subtitle_list.setMaximumHeight(110)
        self.subtitle_list.setStyleSheet(
            "QListWidget { border:none; border-radius:0px; }"
            "QListWidget::item { padding: 2px 8px; border-radius: 0px; margin: 1px 0; }"
            "QListWidget::item:hover { background: transparent; }"
            "QListWidget::item:selected { background: transparent; }"
        )
        self.subtitle_list.setItemDelegate(HighlightDelegate(self.subtitle_list))

        sub_btn_row = QHBoxLayout()
        self.btn_tts_export_srt = QPushButton("↓  导出 SRT")
        self.btn_tts_export_srt.setFixedHeight(30)
        self.btn_tts_export_srt.setEnabled(False)
        sub_btn_row.addStretch()
        sub_btn_row.addWidget(self.btn_tts_export_srt)
        self.btn_tts_export_srt.clicked.connect(self.export_subtitle_srt)

        sub_body_layout.addWidget(self.subtitle_list)
        sub_body_layout.addLayout(sub_btn_row)

        sub_section_layout.addLayout(sub_header)
        sub_section_layout.addWidget(self._sub_body)
        self.btn_collapse_sub.clicked.connect(lambda: self._toggle_section(sub_section, self._sub_body, self.btn_collapse_sub))

        # ── TTS section (collapsible) ─────────────────────────────────────────
        tts_section = QWidget()
        tts_section.setObjectName("right_panel")
        tts_section_layout = QVBoxLayout(tts_section)
        tts_section_layout.setContentsMargins(0, 0, 0, 0)
        tts_section_layout.setSpacing(4)

        tts_header = QHBoxLayout()
        tts_header_label = QLabel("TTS GENERATION")
        tts_header_label.setObjectName("section_label")
        self.btn_collapse_tts = QPushButton("▾")
        self.btn_collapse_tts.setFixedSize(20, 20)
        self.btn_collapse_tts.setObjectName("btn_collapse")
        tts_header.addWidget(tts_header_label)
        tts_header.addStretch()
        tts_header.addWidget(self.btn_collapse_tts)

        self._tts_body = QWidget()
        tts_body_layout = QVBoxLayout(self._tts_body)
        tts_body_layout.setSpacing(6)
        tts_body_layout.setContentsMargins(10, 8, 10, 8)

        voice_row = QHBoxLayout()
        voice_lbl = QLabel("音色")
        voice_lbl.setObjectName("section_label")
        voice_lbl.setFixedWidth(30)
        self.combo_tts_voice = QComboBox()
        for label, _ in EDGE_TTS_VOICES:
            self.combo_tts_voice.addItem(label)
        voice_row.addWidget(voice_lbl)
        voice_row.addWidget(self.combo_tts_voice, 1)
        tts_body_layout.addLayout(voice_row)

        self.tts_text_edit = PlainPasteTextEdit()
        self.tts_text_edit.setPlaceholderText(
            '粘贴一段文字后点\u300c自动断句\u300d，程序按标点切分每句一行\n也可直接在此手动编辑，每行一句，生成后自动载入练习'
        )
        self.tts_text_edit.setMinimumHeight(62)
        self.tts_text_edit.setMaximumHeight(90)
        self.tts_text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        self.tts_text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._apply_line_spacing(self.tts_text_edit, 130)
        tts_body_layout.addWidget(self.tts_text_edit)

        tts_split_row = QHBoxLayout()
        tts_split_row.setSpacing(6)
        self.btn_tts_split = QPushButton("⌥  自动断句")
        self.btn_tts_split.setFixedHeight(30)
        self.btn_tts_split.setToolTip('按标点自动将文本切分为每行一句，可再手动调整后点\u300c生成\u300d')
        tts_split_row.addWidget(self.btn_tts_split)
        tts_split_row.addStretch()
        tts_body_layout.addLayout(tts_split_row)

        tts_btn_row = QHBoxLayout()
        tts_btn_row.setSpacing(6)
        self.btn_tts_generate = QPushButton("◈  生成并载入练习")
        self.btn_tts_generate.setObjectName("btn_accent")
        self.btn_tts_generate.setFixedHeight(34)
        self.btn_tts_play = QPushButton("▷  重新播放")
        self.btn_tts_play.setFixedHeight(34)
        self.btn_tts_play.setEnabled(False)
        tts_btn_row.addWidget(self.btn_tts_generate, 2)
        tts_btn_row.addWidget(self.btn_tts_play, 1)
        tts_body_layout.addLayout(tts_btn_row)

        self.tts_status_label = QLabel("就绪")
        self.tts_status_label.setObjectName("tts_status")
        tts_body_layout.addWidget(self.tts_status_label)

        tts_section_layout.addLayout(tts_header)
        tts_section_layout.addWidget(self._tts_body)
        self.btn_collapse_tts.clicked.connect(lambda: self._toggle_section(tts_section, self._tts_body, self.btn_collapse_tts))

        self.btn_tts_split.clicked.connect(self.tts_split_sentences)
        self.btn_tts_generate.clicked.connect(self.tts_generate)
        self.btn_tts_play.clicked.connect(self.tts_play)

        # ── Assemble right panel ──────────────────────────────────────────────
        right_layout.addWidget(seg_section, 1)
        right_layout.addWidget(rec_section, 0)
        right_layout.addWidget(sub_section, 0)
        right_layout.addWidget(tts_section, 0)

        self.splitter.addWidget(left)
        self.splitter.addWidget(right)

        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.splitter)
    def _build_menu(self):
        def add_actions(menu, actions):
            for item in actions:
                if item is None:
                    menu.addSeparator()
                else:
                    label, slot = item
                    act = QAction(label, self)
                    act.triggered.connect(slot)
                    menu.addAction(act)

        menu = self.menuBar()
        add_actions(menu.addMenu("文件"), [
            ("新建项目", self.new_project),
            ("打开项目", self.open_project),
            ("保存项目", self.save_project),
        ])
        add_actions(menu.addMenu("媒体"), [
            ("导入视频", self.load_video),
            ("导入音频", self.load_audio),
        ])
        add_actions(menu.addMenu("字幕"), [
            ("载入字幕文件",                  self.load_subtitle),
            ("自动识别字幕 · 英语 (Whisper)", lambda: self.asr_transcribe("en")),
            ("自动识别字幕 · 日语 (Whisper)", lambda: self.asr_transcribe("ja")),
            None,
            ("从片段重建字幕",                self.rebuild_subtitles_from_segments),
            None,
            ("删除字幕",                      self.delete_subtitle),
        ])
        add_actions(menu.addMenu("导出"), [
            ("导出字幕 SRT", self.export_subtitle_srt),
        ])

    def _build_player(self):
        self.player.setVideoOutput(self.video_widget.video_widget)
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        self.player.playbackStateChanged.connect(self.on_playback_state_changed)
        self.player.mediaStatusChanged.connect(self.on_player_media_status_changed)

    # ──────────────────────────────────────────────────────────── helpers ──

    def _warn_unsaved_project(self):
        """若项目尚未保存，在状态栏显示一次持续性提示，引导用户保存。"""
        if self.current_project_path:
            return  # 已保存，无需提示
        QTimer.singleShot(800, lambda: self.set_status(
            "⚠️  文件暂存在系统临时目录，关机后将丢失！请「文件 → 保存项目」"))

    def set_status(self, text: str):
        self.status_label.setText(text)

    def _update_export_btn(self):
        """只要 subtitle_entries 非空，导出按钮就可用。"""
        self.btn_tts_export_srt.setEnabled(bool(self.subtitle_entries))

    def _toggle_section(self, section: QWidget, body: QWidget, btn: QPushButton):
        """折叠或展开整个 section（含 body），真正释放空间。"""
        if body.maximumHeight() == 0:
            # 展开
            body.setMinimumHeight(0)
            body.setMaximumHeight(16777215)
            section.setMinimumHeight(0)
            section.setMaximumHeight(16777215)
            btn.setText("▾")
        else:
            # 折叠：body 归零，section 也设为 header 高度（约 24px）
            body.setMinimumHeight(0)
            body.setMaximumHeight(0)
            section.setMinimumHeight(0)
            section.setMaximumHeight(24)
            btn.setText("▸")

    def reset_splitter_equal(self):
        total = self.splitter.width()
        if total > 0:
            self.splitter.setSizes([total // 2, total - total // 2])

    def get_segment_by_id(self, seg_id: int) -> Optional[Segment]:
        return next((s for s in self.project.segments if s.id == seg_id), None)

    def get_selected_segment(self) -> Optional[Segment]:
        return self.get_segment_by_id(self.selected_segment_id) \
            if self.selected_segment_id is not None else None

    def ensure_item_visible_if_needed(self, lw: QListWidget, item: QListWidgetItem):
        if not item:
            return
        rect          = lw.visualItemRect(item)
        viewport_rect = lw.viewport().rect()
        if viewport_rect.contains(rect):
            return
        if rect.top() < viewport_rect.top():
            lw.scrollToItem(item, QAbstractItemView.PositionAtTop)
        elif rect.bottom() > viewport_rect.bottom():
            lw.scrollToItem(item, QAbstractItemView.PositionAtBottom)
        else:
            lw.scrollToItem(item, QAbstractItemView.EnsureVisible)

    def fix_current_selection_view(self):
        if self.selected_segment_id is not None:
            for i in range(self.segment_list.count()):
                item = self.segment_list.item(i)
                if item and item.data(Qt.UserRole) == self.selected_segment_id:
                    self.ensure_item_visible_if_needed(self.segment_list, item)
                    break
        if self._current_subtitle_highlight_idx is not None:
            idx = self._current_subtitle_highlight_idx
            if 0 <= idx < self.subtitle_list.count():
                self.ensure_item_visible_if_needed(
                    self.subtitle_list, self.subtitle_list.item(idx))

    def show_segment_text(self, text: str):
        self._ui_lock = True
        with QSignalBlocker(self.text_editor):
            self.text_editor.setPlainText(text or "")
        self.text_editor.moveCursor(QTextCursor.Start)
        self.text_editor.ensureCursorVisible()
        self._ui_lock = False

    # ──────────────────────────────────────────────────── segment / playback ──

    def set_current_segment(self, seg_id: int, source: str = "segment"):
        seg = self.get_segment_by_id(seg_id)
        if not seg:
            return
        self.stop_segment_loop(silent=True)
        self.selected_segment_id = seg_id
        self.selected_subtitle_index = self.find_best_subtitle_index_for_segment(seg)
        self._current_subtitle_highlight_idx = self.selected_subtitle_index
        self.show_segment_text(seg.text or "")
        self.refresh_segment_list(keep_scroll=True)
        self.refresh_highlights()
        self.fix_current_selection_view()
        QTimer.singleShot(0, self.fix_current_selection_view)
        self.start_segment_loop(seg)

    def start_segment_loop(self, seg: Segment):
        if not self.project.media_path or not os.path.exists(self.project.media_path):
            return
        self.record_audio_player.stop()
        self.player.stop()
        self.player_mode = "segment_loop"
        self.segment_loop_end_ms = int(seg.end * 1000)
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(self.project.media_path)))
        self.player.setPosition(int(seg.start * 1000))
        self.player.play()
        self.set_status("播放原音中")

    def stop_segment_loop(self, silent: bool = False):
        if self.player_mode != "segment_loop":
            if not silent:
                self.set_status("当前没有循环播放")
            return
        self.player.stop()
        self.player_mode = "idle"
        self.segment_loop_end_ms = None
        self.compare_original_seg = None
        self.refresh_segment_list(keep_scroll=True)
        self.refresh_highlights()
        if not silent:
            self.set_status("已停止循环播放")

    # ────────────────────────────────────────────────────────── project ops ──

    def new_project(self):
        self.project = Project()
        self.current_project_path = None
        self.cut_points = []
        self.subtitle_entries = []
        self.selected_segment_id = None
        self.selected_subtitle_index = None
        self.recording_segment_id = None
        self.recording_output_path = None
        self.segment_list.clear()
        self.subtitle_list.clear()
        self.text_editor.clear()
        self.player.stop()
        self.record_audio_player.stop()
        self.player.setSource(QUrl())
        self.slider.setRange(0, 0)
        self.time_label.setText("00:00.00 / 00:00.00")
        self.player_mode = "idle"
        self.segment_loop_end_ms = None
        self.compare_original_seg = None
        self.btn_play_compare.setText("⇌  播放对比")
        self.btn_tts_export_srt.setEnabled(False)
        # ── 重置 TTS 模块 ──────────────────────────────────────────────────────
        self._tts_audio_path = None
        self._tts_lines = []
        self.tts_text_edit.clear()
        self.tts_status_label.setText("就绪")
        self.btn_tts_play.setEnabled(False)
        self.btn_tts_generate.setEnabled(True)
        self.combo_tts_voice.setCurrentIndex(0)
        self.update_generate_mode_ui()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", "Project Files (*.json)")
        if not path:
            return
        try:
            self.project = Project.load(path)
            self.current_project_path = path
            self.cut_points = self.project.cut_points[:]
            self.subtitle_entries = self.project.subtitle_entries[:]
            self.render_subtitle_list()
            if self.project.media_path and os.path.exists(self.project.media_path):
                self.load_media_path(self.project.media_path)
            self.refresh_segment_list(keep_scroll=False)
            self.refresh_highlights()
            self.update_generate_mode_ui()
            self._update_export_btn()
        except Exception as e:
            self.set_status(f"打开项目失败：{e}")

    def _sync_project_state(self):
        self.project.cut_points      = self.cut_points[:]
        self.project.subtitle_entries = self.subtitle_entries[:]

    def save_project(self):
        if not self.current_project_path:
            path, _ = QFileDialog.getSaveFileName(self, "保存项目", "", "Project Files (*.json)")
            if not path:
                return
            if not path.endswith(".json"):
                path += ".json"
            self.current_project_path = path
        try:
            self._sync_project_state()
            self.project.save(self.current_project_path)
            self.set_status("项目已保存")
        except Exception as e:
            self.set_status(f"项目保存失败：{e}")

    def auto_save_project(self):
        if self.current_project_path:
            try:
                self._sync_project_state()
                self.project.save(self.current_project_path)
            except Exception:
                pass

    # ────────────────────────────────────────────────────────── media / ASR ──

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.wmv);;All Files (*)")
        if path:
            self.load_media_path(path)

    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频", "",
            "Audio Files (*.mp3 *.wav *.aac *.m4a *.flac *.ogg);;All Files (*)")
        if not path:
            return
        try:
            tmp_wav = os.path.join(tempfile.gettempdir(), "EnglishDubApp_loaded_audio.wav")
            if not convert_to_wav_with_ffmpeg(path, tmp_wav) or not os.path.exists(tmp_wav):
                self.set_status("音频转换失败")
                return
            self.load_media_path(tmp_wav)
            self.project.media_path = path
            self.auto_save_project()
        except Exception as e:
            self.set_status(f"音频加载失败: {e}")

    def load_media_path(self, path: str):
        if not os.path.exists(path):
            self.set_status("媒体文件不存在")
            return
        self.project.media_path = path
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(path)))
        self.player.pause()
        self.player_mode = "idle"
        self.segment_loop_end_ms = None
        self.set_status(f"已加载媒体：{os.path.basename(path)}")
        self.auto_save_project()
        QTimer.singleShot(200, self.update_video_size_from_metadata)

    def update_video_size_from_metadata(self):
        try:
            size = self.player.metaData().value(QMediaPlayer.MetaDataKey.Resolution)
            if size and hasattr(size, "width") and hasattr(size, "height"):
                self.video_widget.set_video_size(size.width(), size.height())
        except Exception:
            pass

    def asr_transcribe(self, language: str = "en"):
        if not self.project.media_path or not os.path.exists(self.project.media_path):
            self.set_status("请先导入媒体文件")
            return
        if self._asr_worker and self._asr_worker.isRunning():
            self.set_status("识别进行中，请稍候…")
            return

        asr_dir = os.path.join(self.get_recording_folder(), "asr")
        os.makedirs(asr_dir, exist_ok=True)
        name     = hashlib.md5(f"{language}:{self.project.media_path}".encode()).hexdigest()[:12]
        out_json = os.path.join(asr_dir, f"asr_{name}.json")

        lang_label = "英语" if language == "en" else "日语"
        self.set_status(f"Whisper 识别中（{WHISPER_MODEL} · {lang_label}），请稍候…")
        self._asr_worker = AsrWorker(self.project.media_path, out_json, language)
        self._asr_worker.finished.connect(self._on_asr_finished)
        self._asr_worker.progress.connect(self.set_status)
        self._asr_worker.error.connect(self._on_asr_error)
        self._asr_worker.start()

    def _on_asr_finished(self, json_path: str):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            if not results:
                self.set_status("识别完成，但未检测到语音内容")
                return
            self.subtitle_entries = [
                SubtitleEntry(start=r["start"], end=r["end"], text=r["text"])
                for r in results
            ]
            self.project.subtitle_entries = self.subtitle_entries[:]
            self.project.subtitle_path    = json_path
            self.render_subtitle_list()
            self.segment_generate_mode = "subtitles"
            self.update_generate_mode_ui()
            self.rebuild_segments_from_subtitles()
            self.auto_save_project()
            self.btn_tts_export_srt.setEnabled(True)
            self.set_status(f"识别完成，共 {len(self.subtitle_entries)} 句，已载入片段")
            self._warn_unsaved_project()
        except Exception as e:
            self.set_status(f"识别结果解析失败：{e}")

    def _on_asr_error(self, msg: str):
        print(f"[ASR ERROR] {msg}")
        short = msg[:120] + "…" if len(msg) > 120 else msg
        self.set_status(f"识别失败：{short}")

    def load_subtitle(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择字幕文件", "", "Subtitle Files (*.srt)")
        if not path:
            return
        try:
            self.subtitle_entries          = parse_srt(path)
            self.project.subtitle_path     = path
            self.project.subtitle_entries  = self.subtitle_entries[:]
            self.render_subtitle_list()
            self.match_subtitles_to_segments(force=False)
            self.update_generate_mode_ui()
            self.auto_save_project()
            self.btn_tts_export_srt.setEnabled(True)
        except Exception as e:
            self.set_status(f"字幕载入失败：{e}")

    def rebuild_subtitles_from_segments(self):
        """将当前所有片段的时间戳和文字反向重建为字幕列表。"""
        if not self.project.segments:
            self.set_status("没有可用的片段，请先生成片段")
            return
        entries = [
            SubtitleEntry(start=seg.start, end=seg.end, text=seg.text or "")
            for seg in sorted(self.project.segments, key=lambda s: s.start)
        ]
        self.subtitle_entries         = entries
        self.project.subtitle_entries = entries[:]
        self.render_subtitle_list()
        self.refresh_highlights()
        self._update_export_btn()
        self.auto_save_project()
        self.set_status(f"已从片段重建 {len(entries)} 条字幕")

    def delete_subtitle(self):
        self.subtitle_entries             = []
        self.project.subtitle_entries     = []
        self.project.subtitle_path        = ""
        self.selected_subtitle_index      = None
        self._current_subtitle_highlight_idx = None
        self.subtitle_list.clear()
        self.btn_tts_export_srt.setEnabled(False)
        self.refresh_segment_list()
        self.refresh_highlights()
        self.auto_save_project()

    def render_subtitle_list(self):
        with QSignalBlocker(self.subtitle_list):
            self.subtitle_list.clear()
            fm = self.subtitle_list.fontMetrics()
            available_width = max(self.subtitle_list.viewport().width() - 24, 200)
            for idx, sub in enumerate(self.subtitle_entries):
                text = f"{idx+1}. {format_time(sub.start)} ~ {format_time(sub.end)} | {sub.text}"
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, idx)
                rect = fm.boundingRect(0, 0, available_width, 0, Qt.TextWordWrap, text)
                item.setSizeHint(QSize(rect.width(), fm.height()))
                self.subtitle_list.addItem(item)

    def match_subtitles_to_segments(self, force: bool = False):
        if not self.subtitle_entries or not self.project.segments:
            return
        for seg in self.project.segments:
            best_text = max(
                self.subtitle_entries,
                key=lambda sub: min(seg.end, sub.end) - max(seg.start, sub.start)
            ).text
            if best_text and (force or not seg.text):
                seg.text = best_text
        self.refresh_segment_list()
        self.auto_save_project()

    def on_subtitle_selected(self, item: QListWidgetItem):
        idx = item.data(Qt.UserRole)
        if idx is None or idx < 0 or idx >= len(self.subtitle_entries):
            return
        sub    = self.subtitle_entries[idx]
        seg_id = self.find_best_segment_id_for_subtitle(sub)
        if seg_id is None:
            return
        self._current_subtitle_highlight_idx = idx
        self.selected_subtitle_index = idx
        self.stop_segment_loop(silent=True)
        self.set_current_segment(seg_id, source="subtitle")

    def on_segment_selected(self, item: QListWidgetItem):
        if self._segment_list_refreshing:
            return
        seg_id = item.data(Qt.UserRole)
        if seg_id is None:
            return
        if self.selected_segment_id == seg_id and self.player_mode == "segment_loop":
            self.stop_segment_loop()
            return
        self.selected_subtitle_index = self.find_best_subtitle_index_for_segment(
            self.get_segment_by_id(seg_id))
        self.set_current_segment(seg_id, source="segment")

    # ──────────────────────────────────────────────────────── list highlights ──

    def _clear_list_highlight(self, lw: QListWidget):
        for i in range(lw.count()):
            item = lw.item(i)
            if item:
                item.setBackground(QBrush())
                item.setForeground(QBrush())
                if lw is self.subtitle_list:
                    item.setData(HIGHLIGHT_ROLE, False)

    def refresh_segment_highlight(self):
        self._clear_list_highlight(self.segment_list)
        highlight_id = (
            self.recording_segment_id
            if self.recorder.is_recording and self.recording_segment_id is not None
            else self.selected_segment_id
        )
        if highlight_id is None:
            return
        for i in range(self.segment_list.count()):
            item = self.segment_list.item(i)
            if item and item.data(Qt.UserRole) == highlight_id:
                item.setBackground(QBrush(QColor("#ffe58f")))
                item.setForeground(QBrush(QColor("#000000")))
                break

    def refresh_subtitle_highlight(self, highlight_idx: Optional[int]):
        self._clear_list_highlight(self.subtitle_list)
        if highlight_idx is not None and 0 <= highlight_idx < self.subtitle_list.count():
            item = self.subtitle_list.item(highlight_idx)
            if item:
                item.setData(HIGHLIGHT_ROLE, True)
                item.setForeground(QBrush(QColor("#0a5c2e")))

    def refresh_highlights(self):
        self.refresh_segment_highlight()
        self.refresh_subtitle_highlight(self._current_subtitle_highlight_idx)

    # ─────────────────────────────────────────────────────── playback control ──

    def toggle_play_pause(self):
        if self.player.source().isEmpty():
            self.set_status("请先导入媒体")
            return
        if self.player_mode in ("segment_loop", "compare_original",
                                "compare_recording", "playing_recording"):
            self.set_status("右侧功能播放中，请先停止后再使用左侧播放")
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self._playback_auto_highlight_ms = -9999  # 立即触发首次高亮
            self.player.play()

    def seek_video(self, value):
        if self.player_mode not in ("segment_loop", "compare_original", "compare_recording"):
            self.player.setPosition(value)

    def on_position_changed(self, pos):
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)

        if (self.player_mode == "segment_loop"
                and self.segment_loop_end_ms is not None
                and self.selected_segment_id is not None):
            if pos >= self.segment_loop_end_ms - LOOP_BACK_MARGIN_MS:
                seg = self.get_segment_by_id(self.selected_segment_id)
                if seg:
                    self.player.blockSignals(True)
                    self.player.setPosition(int(seg.start * 1000))
                    self.player.blockSignals(False)
                return

        if self.player_mode == "compare_original" and self.segment_loop_end_ms is not None:
            if pos >= self.segment_loop_end_ms - LOOP_BACK_MARGIN_MS:
                seg = self.compare_original_seg
                if seg and seg.audio_path and os.path.exists(seg.audio_path):
                    self.player.stop()
                    self.player_mode = "compare_recording"
                    self.record_audio_player.setSource(QUrl.fromLocalFile(seg.audio_path))
                    self.record_audio_player.play()
                    self.set_status("播放对比：录音中")
                return

        # ── 左侧自由播放时自动高亮 segment / subtitle ───────────────────────────
        # 仅在 idle 模式（左侧 ▶ 播放 / 拖动）且有内容时执行，每 200ms 最多刷新一次
        if (self.player_mode == "idle"
                and abs(pos - self._playback_auto_highlight_ms) >= 200):
            self._playback_auto_highlight_ms = pos
            pos_sec = pos / 1000.0

            # —— segment 列表跟踪 ——
            new_seg_id: Optional[int] = None
            for seg in self.project.segments:
                if seg.start <= pos_sec < seg.end:
                    new_seg_id = seg.id
                    break
            if new_seg_id != self.selected_segment_id:
                self.selected_segment_id = new_seg_id
                # 同步高亮并滚动，不触发任何播放逻辑
                self.refresh_segment_highlight()
                if new_seg_id is not None:
                    for i in range(self.segment_list.count()):
                        item = self.segment_list.item(i)
                        if item and item.data(Qt.UserRole) == new_seg_id:
                            self.segment_list.blockSignals(True)
                            self.segment_list.setCurrentItem(item)
                            self.segment_list.blockSignals(False)
                            self.ensure_item_visible_if_needed(self.segment_list, item)
                            break
                    seg_obj = self.get_segment_by_id(new_seg_id)
                    if seg_obj:
                        self.show_segment_text(seg_obj.text or "")

            # —— subtitle 列表跟踪 ——
            new_sub_idx: Optional[int] = None
            for i, sub in enumerate(self.subtitle_entries):
                if sub.start <= pos_sec < sub.end:
                    new_sub_idx = i
                    break
            if new_sub_idx != self._current_subtitle_highlight_idx:
                self._current_subtitle_highlight_idx = new_sub_idx
                self.refresh_subtitle_highlight(new_sub_idx)
                if new_sub_idx is not None:
                    item = self.subtitle_list.item(new_sub_idx)
                    if item:
                        self.ensure_item_visible_if_needed(self.subtitle_list, item)

    def on_duration_changed(self, duration):
        self.video_duration_ms = duration
        self.slider.setRange(0, duration)
        self.update_time_label()

    def on_playback_state_changed(self, state):
        self.btn_toggle_play.setText(
            "⏸  暂停" if state == QMediaPlayer.PlaybackState.PlayingState else "▶  播放")
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.ui_timer.start()
        else:
            self.ui_timer.stop()

    def on_player_media_status_changed(self, status):
        """最后一句到达文件末尾时的兜底处理。"""
        if status != QMediaPlayer.MediaStatus.EndOfMedia:
            return
        if self.player_mode == "segment_loop" and self.selected_segment_id is not None:
            seg = self.get_segment_by_id(self.selected_segment_id)
            if seg:
                self.player.setPosition(int(seg.start * 1000))
                self.player.play()
        elif self.player_mode == "compare_original":
            seg = self.compare_original_seg
            if seg and seg.audio_path and os.path.exists(seg.audio_path):
                self.player_mode = "compare_recording"
                self.record_audio_player.setSource(QUrl.fromLocalFile(seg.audio_path))
                self.record_audio_player.play()
                self.set_status("播放对比：录音中")

    def on_record_audio_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia \
                and self.player_mode in ("playing_recording", "compare_recording"):
            if self.player_mode == "compare_recording":
                self.btn_play_compare.setText("⇌  播放对比")
            self.player_mode = "idle"
            self.refresh_segment_list(keep_scroll=True)
            self.refresh_highlights()
            self.set_status("播放结束")

    def update_time_label(self):
        self.time_label.setText(
            f"{format_time(self.player.position()/1000.0)} / "
            f"{format_time(self.player.duration()/1000.0)}")

    # ──────────────────────────────────────────────────── segment generation ──

    def update_generate_mode_ui(self):
        if self.segment_generate_mode == "subtitles" and not self.subtitle_entries:
            self.combo_generate_mode.blockSignals(True)
            self.combo_generate_mode.setCurrentIndex(0)
            self.combo_generate_mode.blockSignals(False)
            self.segment_generate_mode = "cuts"
        self.btn_add_cut.setEnabled(self.segment_generate_mode == "cuts")
        self.btn_rebuild.setText(
            "根据切点生成句子" if self.segment_generate_mode == "cuts"
            else "根据字幕生成句子")

    def on_generate_mode_changed(self, idx: int):
        self.segment_generate_mode = "cuts" if idx == 0 else "subtitles"
        self.update_generate_mode_ui()

    def add_cut_point(self):
        current_sec = self.player.position() / 1000.0
        if current_sec <= 0:
            self.set_status("切点无效")
            return
        self.cut_points = sorted(set(self.cut_points) | {current_sec})
        self.project.cut_points = self.cut_points[:]
        self.auto_save_project()

    def rebuild_segments_by_mode(self):
        if self.segment_generate_mode == "cuts":
            self.rebuild_segments_from_cuts()
        else:
            self.rebuild_segments_from_subtitles()

    def _collect_old_segment_data(self):
        return {(round(seg.start, 3), round(seg.end, 3)): (seg.text, seg.audio_path)
                for seg in self.project.segments}

    def rebuild_segments_from_cuts(self):
        if self.video_duration_ms <= 0:
            self.set_status("请先导入媒体")
            return
        points   = sorted([0.0] + self.cut_points + [self.video_duration_ms / 1000.0])
        old_data = self._collect_old_segment_data()
        self.project.segments.clear()
        for i in range(len(points) - 1):
            start, end = round(points[i], 3), round(points[i + 1], 3)
            if end - start < 0.2:
                continue
            text, audio_path = old_data.get((start, end), ("", ""))
            self.project.segments.append(Segment(
                id=len(self.project.segments) + 1,
                start=start, end=end, text=text, audio_path=audio_path))
        self.project.cut_points = self.cut_points[:]
        self.match_subtitles_to_segments(force=False)
        self.refresh_segment_list(keep_scroll=False)
        self.refresh_highlights()
        self.auto_save_project()

    def rebuild_segments_from_subtitles(self):
        if not self.subtitle_entries:
            self.set_status("请先载入字幕")
            return
        old_data = self._collect_old_segment_data()
        self.project.segments.clear()
        for i, sub in enumerate(self.subtitle_entries):
            key = (round(sub.start, 3), round(sub.end, 3))
            text, audio_path = old_data.get(key, ("", ""))
            if not text:
                text = sub.text
            self.project.segments.append(Segment(
                id=i + 1, start=key[0], end=key[1], text=text, audio_path=audio_path))
        self.cut_points = []
        self.project.cut_points = []
        self.refresh_segment_list(keep_scroll=False)
        self.refresh_highlights()
        self.auto_save_project()

    def refresh_segment_list(self, keep_scroll: bool = True):
        old_scroll = self.segment_list.verticalScrollBar().value() if keep_scroll else 0
        self._segment_list_refreshing = True
        with QSignalBlocker(self.segment_list):
            self.segment_list.clear()
            for seg in self.project.segments:
                play_status  = self.get_segment_play_status(seg)
                audio_status = self.get_segment_audio_status(seg)
                play_icon, audio_icon = self.status_to_icons(play_status, audio_status)
                item = QListWidgetItem(
                    f"[{seg.id}] {seg.start:.2f}s ~ {seg.end:.2f}s | "
                    f"{play_icon} {play_status} | {audio_icon} {audio_status}")
                item.setData(Qt.UserRole, seg.id)
                self.segment_list.addItem(item)
        self._segment_list_refreshing = False
        if self.selected_segment_id is not None:
            for i in range(self.segment_list.count()):
                item = self.segment_list.item(i)
                if item and item.data(Qt.UserRole) == self.selected_segment_id:
                    self.segment_list.setCurrentItem(item)
                    break
        self.refresh_segment_highlight()
        if keep_scroll:
            QTimer.singleShot(
                0, lambda v=old_scroll: self.segment_list.verticalScrollBar().setValue(v))

    def get_segment_play_status(self, seg: Segment) -> str:
        if self.recorder.is_recording and self.recording_segment_id == seg.id:
            return "录音中"
        if self.player_mode == "segment_loop" and self.selected_segment_id == seg.id:
            return "循环中"
        if self.player_mode == "playing_recording" and self.selected_segment_id == seg.id:
            return "播放录音中"
        if self.player_mode in ("compare_original", "compare_recording") \
                and self.selected_segment_id == seg.id:
            return "对比中"
        return "空闲"

    def get_segment_audio_status(self, seg: Segment) -> str:
        return "已录音" if seg.audio_path and os.path.exists(seg.audio_path) else "未录音"

    def status_to_icons(self, play_status: str, audio_status: str):
        play_icon_map  = {"循环中": "▶", "录音中": "●", "播放录音中": "♪",
                          "对比中": "⇄", "空闲": "○"}
        audio_icon_map = {"已录音": "✓", "未录音": "○"}
        return play_icon_map.get(play_status, "○"), audio_icon_map.get(audio_status, "○")

    # ──────────────────────────────────────────────────────────── recording ──

    def toggle_recording(self):
        if not self.recorder.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        seg = self.get_selected_segment()
        if not seg:
            self.set_status("请先选择片段")
            return
        self.stop_segment_loop(silent=True)
        self.player.stop()
        self.record_audio_player.stop()
        self.player_mode = "recording"
        self.recording_segment_id = seg.id
        out_dir = self.get_recording_folder()
        os.makedirs(out_dir, exist_ok=True)
        self.recording_output_path = os.path.join(out_dir, f"segment_{seg.id}.wav")
        self.recorder.start()
        self.btn_record_toggle.setText("⏹  停止录音")
        self.refresh_segment_list(keep_scroll=True)
        self.refresh_highlights()
        self.set_status("录音中")

    def stop_recording(self):
        if not self.recorder.is_recording:
            return
        ok = self.recorder.stop(self.recording_output_path or "")
        self.btn_record_toggle.setText("⏺  开始录音")
        if ok and self.recording_segment_id is not None:
            seg = self.get_segment_by_id(self.recording_segment_id)
            if seg:
                seg.audio_path = self.recording_output_path or ""
            self.auto_save_project()
            self.recording_segment_id = None
            self.player_mode = "idle"
            self.refresh_segment_list(keep_scroll=True)
            self.refresh_highlights()
            self._warn_unsaved_project()
            if self.recording_output_path and os.path.exists(self.recording_output_path):
                self.play_selected_audio()
        else:
            self.set_status("未录到有效音频")
            self.recording_segment_id = None

    def play_selected_audio(self):
        seg = self.get_selected_segment()
        if not seg:
            self.set_status("请先选择片段")
            return
        if not seg.audio_path or not os.path.exists(seg.audio_path):
            self.set_status("当前片段没有录音文件")
            return
        self.stop_segment_loop(silent=True)
        self.player.stop()
        self.record_audio_player.stop()
        self.player_mode = "playing_recording"
        self.record_audio_player.setSource(QUrl.fromLocalFile(seg.audio_path))
        self.record_audio_player.play()
        self.refresh_segment_list(keep_scroll=True)
        self.refresh_highlights()
        self.set_status("播放录音中")

    def toggle_play_compare(self):
        if self.player_mode in ("compare_original", "compare_recording"):
            self.stop_play_compare()
        else:
            self.play_compare()

    def stop_play_compare(self, silent: bool = False):
        if self.player_mode not in ("compare_original", "compare_recording"):
            return
        self.player.stop()
        self.record_audio_player.stop()
        self.player_mode = "idle"
        self.segment_loop_end_ms = None
        self.compare_original_seg = None
        self.btn_play_compare.setText("⇌  播放对比")
        self.refresh_segment_list(keep_scroll=True)
        self.refresh_highlights()
        if not silent:
            self.set_status("已停止对比")

    def play_compare(self):
        seg = self.get_selected_segment()
        if not seg:
            self.set_status("请先选择片段")
            return
        if not seg.audio_path or not os.path.exists(seg.audio_path):
            self.set_status("当前片段没有录音文件")
            return
        self.stop_segment_loop(silent=True)
        self.player.stop()
        self.record_audio_player.stop()
        self.compare_original_seg = seg
        self.player_mode          = "compare_original"
        self.segment_loop_end_ms  = int(seg.end * 1000)
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(self.project.media_path)))
        self.player.setPosition(int(seg.start * 1000))
        self.player.play()
        self.btn_play_compare.setText("⏹  停止对比")
        self.refresh_segment_list(keep_scroll=True)
        self.refresh_highlights()
        self.set_status("播放对比：原音中")

    def get_recording_folder(self):
        if self.current_project_path:
            base = os.path.splitext(os.path.basename(self.current_project_path))[0]
            return os.path.join(os.path.dirname(self.current_project_path),
                                f"{base}_recordings")
        return os.path.join(tempfile.gettempdir(), "EnglishDubApp_Recordings")

    def open_recording_folder(self):
        folder = self.get_recording_folder()
        os.makedirs(folder, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            self.set_status(f"无法打开录音文件夹：{e}")


    def merge_all_recordings(self):
        """按片段顺序合并所有已录音文件，导出为一个完整 WAV。"""
        recorded_segs = [
            seg for seg in sorted(self.project.segments, key=lambda s: s.id)
            if seg.audio_path and os.path.exists(seg.audio_path)
        ]
        if not recorded_segs:
            self.set_status("没有可合并的录音")
            return

        out_dir = self.get_recording_folder()
        default_name = "merged_recording.wav"
        if self.current_project_path:
            base = os.path.splitext(os.path.basename(self.current_project_path))[0]
            default_name = f"{base}_merged.wav"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存合并录音", os.path.join(out_dir, default_name),
            "WAV 音频文件 (*.wav)")
        if not save_path:
            return
        if not save_path.endswith(".wav"):
            save_path += ".wav"

        self.set_status("合并中…")
        self.btn_merge_recordings.setEnabled(False)
        QApplication.processEvents()

        try:
            arrays = []
            target_sr = None
            for seg in recorded_segs:
                data, sr = sf.read(seg.audio_path, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if target_sr is None:
                    target_sr = sr
                elif sr != target_sr:
                    ratio = target_sr / sr
                    new_len = int(len(data) * ratio)
                    data = np.interp(
                        np.linspace(0, len(data) - 1, new_len),
                        np.arange(len(data)), data
                    ).astype(np.float32)
                arrays.append(data)

            merged = np.concatenate(arrays)
            sf.write(save_path, merged, target_sr, subtype="PCM_16")

            n = len(recorded_segs)
            total_s = len(merged) / target_sr
            m, s = divmod(int(total_s), 60)
            self.set_status(f"合并完成：已合并 {n} 段（{m:02d}:{s:02d}）→ {os.path.basename(save_path)}")
        except Exception as e:
            self.set_status(f"合并录音失败：{e}")
        finally:
            self.btn_merge_recordings.setEnabled(True)

    def auto_save_segment_text(self):
        if self._ui_lock or self.selected_segment_id is None:
            return
        seg = self.get_segment_by_id(self.selected_segment_id)
        if seg:
            seg.text = self.text_editor.toPlainText().strip()
            self.refresh_segment_list(keep_scroll=True)
            self.refresh_highlights()
            self.auto_save_project()

    # ──────────────────────────────────────────────────────── overlap search ──

    def _find_best_overlap_idx(self, items, ref) -> Optional[int]:
        best_idx, best_score = None, -999999.0
        for i, item in enumerate(items):
            overlap = min(item.end, ref.end) - max(item.start, ref.start)
            if overlap > best_score:
                best_score, best_idx = overlap, i
        return best_idx

    def find_best_subtitle_index_for_segment(self, seg) -> Optional[int]:
        if not self.subtitle_entries or not seg:
            return None
        return self._find_best_overlap_idx(self.subtitle_entries, seg)

    def find_best_segment_id_for_subtitle(self, sub) -> Optional[int]:
        if not self.project.segments:
            return None
        idx = self._find_best_overlap_idx(self.project.segments, sub)
        return self.project.segments[idx].id if idx is not None else None

    # ─────────────────────────────────────────────────────────────────── TTS ──

    def tts_split_sentences(self):
        """按标点自动断句，将文本框内容替换为每行一句。"""
        raw = self.tts_text_edit.toPlainText()
        if not raw.strip():
            self.tts_status_label.setText("文本框为空，请先粘贴文字")
            return

        # 以中英文句末标点为切分点，保留标点在句尾
        # 支持：。！？…… . ! ?  以及全半角变体
        parts = re.split(r'(?<=[。！？…\.!?])\s*', raw)
        lines = [p.strip() for p in parts if p.strip()]

        if not lines:
            self.tts_status_label.setText("未检测到标点，请手动换行后生成")
            return

        self.tts_text_edit.setPlainText("\n".join(lines))
        self._apply_line_spacing(self.tts_text_edit, 130)
        self.tts_status_label.setText(f'已断句 {len(lines)} 行，可手动调整后点\u300c生成\u300d')

    def tts_generate(self):
        raw   = self.tts_text_edit.toPlainText()
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if not lines:
            self.tts_status_label.setText("请先输入文本（每行一句）")
            return
        if self._tts_worker and self._tts_worker.isRunning():
            self.tts_status_label.setText("正在生成中，请稍候…")
            return

        _, voice_id = EDGE_TTS_VOICES[self.combo_tts_voice.currentIndex()]
        tts_dir     = os.path.join(self.get_recording_folder(), "tts")
        os.makedirs(tts_dir, exist_ok=True)
        name        = hashlib.md5(f"{voice_id}:{'|'.join(lines)}".encode()).hexdigest()[:12]
        out_path    = os.path.join(tts_dir, f"tts_{name}.wav")
        timing_path = os.path.join(tts_dir, f"tts_{name}_timing.json")

        self.btn_tts_generate.setEnabled(False)
        self.btn_tts_play.setEnabled(False)
        self.tts_status_label.setText(f"生成中（{len(lines)} 句）…")

        self._tts_lines  = lines
        self._tts_worker = TtsWorker(lines, voice_id, out_path, timing_path)
        self._tts_worker.finished.connect(self._on_tts_finished)
        self._tts_worker.error.connect(self._on_tts_error)
        self._tts_worker.start()

    def _on_tts_finished(self, audio_path: str, timing_path: str):
        self._tts_audio_path = audio_path
        self.btn_tts_generate.setEnabled(True)
        self.btn_tts_play.setEnabled(True)
        self._update_export_btn()
        self.tts_status_label.setText("生成完成，正在载入练习…")

        try:
            subtitles = self._build_subtitles_from_timing(timing_path, self._tts_lines)
        except Exception as e:
            self.tts_status_label.setText(f"时间戳解析失败：{e}")
            subtitles = []

        self.load_media_path(audio_path)

        if subtitles:
            self.subtitle_entries          = subtitles
            self.project.subtitle_entries  = subtitles[:]
            self.render_subtitle_list()
            self.segment_generate_mode     = "subtitles"
            self.update_generate_mode_ui()
            self.rebuild_segments_from_subtitles()
            self._update_export_btn()
            self.tts_status_label.setText(f"已载入 {len(subtitles)} 句，可开始跟读练习")
            self._warn_unsaved_project()
        else:
            self.tts_status_label.setText("音频已载入，但未能解析句子时间戳")

    def _build_subtitles_from_timing(
            self, timing_path: str, lines: List[str]) -> List[SubtitleEntry]:
        with open(timing_path, "r", encoding="utf-8") as f:
            sentences = json.load(f)
        if not sentences:
            return []

        def to_sec(ns100: int) -> float:
            return ns100 / 10_000_000.0

        subtitles = []
        for i, sent in enumerate(sentences):
            start = to_sec(sent["offset"])
            end   = to_sec(sent["offset"] + sent["duration"]) + 0.3
            if i + 1 < len(sentences):
                end = min(end, to_sec(sentences[i + 1]["offset"]))
            text = lines[i] if i < len(lines) else sent["text"]
            subtitles.append(SubtitleEntry(
                start=round(start, 3), end=round(end, 3), text=text))
        return subtitles

    def _on_tts_error(self, msg: str):
        self.btn_tts_generate.setEnabled(True)
        print(f"[TTS ERROR] {msg}")
        short = msg[:120] + "…" if len(msg) > 120 else msg
        self.tts_status_label.setText(f"生成失败：{short}")

    def export_subtitle_srt(self):
        """将当前字幕列表导出为标准 SRT 文件。"""
        if not self.subtitle_entries:
            self.set_status("没有可导出的字幕，请先生成或识别字幕")
            return

        # 默认文件名：项目名或媒体文件名
        default_name = "subtitle"
        if self.current_project_path:
            default_name = os.path.splitext(
                os.path.basename(self.current_project_path))[0]
        elif self.project.media_path:
            default_name = os.path.splitext(
                os.path.basename(self.project.media_path))[0]

        path, _ = QFileDialog.getSaveFileName(
            self, "导出字幕", default_name + ".srt", "SRT 字幕文件 (*.srt)")
        if not path:
            return
        if not path.endswith(".srt"):
            path += ".srt"

        try:
            def fmt(seconds: float) -> str:
                h  = int(seconds // 3600)
                m  = int((seconds % 3600) // 60)
                s  = int(seconds % 60)
                ms = int(round((seconds % 1) * 1000))
                if ms == 1000:
                    ms = 0; s += 1
                if s == 60:
                    s = 0; m += 1
                if m == 60:
                    m = 0; h += 1
                return f"{h:02}:{m:02}:{s:02},{ms:03}"

            with open(path, "w", encoding="utf-8") as f:
                for i, sub in enumerate(self.subtitle_entries, 1):
                    f.write(f"{i}\n")
                    f.write(f"{fmt(sub.start)} --> {fmt(sub.end)}\n")
                    f.write(f"{sub.text}\n\n")

            self.set_status(f"字幕已导出：{os.path.basename(path)}")
        except Exception as e:
            self.set_status(f"导出失败：{e}")

    def tts_play(self):
        if not self._tts_audio_path or not os.path.exists(self._tts_audio_path):
            self.tts_status_label.setText("没有可播放的音频，请先生成")
            return
        self.stop_segment_loop(silent=True)
        self.player.stop()
        self.record_audio_player.stop()
        self.player_mode = "idle"
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(self._tts_audio_path)))
        self.player.setPosition(0)
        self.player.play()
        self.set_status("播放 TTS 音频中")

    # ──────────────────────────────────────────────────────────── close ──

    def closeEvent(self, event):
        if self._asr_worker and self._asr_worker.isRunning():
            self._asr_worker.quit()
            self._asr_worker.wait(2000)
        if self._tts_worker and self._tts_worker.isRunning():
            self._tts_worker.quit()
            self._tts_worker.wait(2000)
        if self.recorder.is_recording:
            try:
                self.recorder.stop(self.recording_output_path or "recording.wav")
            except Exception:
                pass
        self.auto_save_project()
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    QTimer.singleShot(0, w.reset_splitter_equal)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
