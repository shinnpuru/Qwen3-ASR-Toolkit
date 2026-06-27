#!/usr/bin/env python3
"""
Qwen3-ASR CLI — 本地 llama.cpp ASR 转录 + SRT 字幕生成工具
"""

import argparse
import os
import sys
import tempfile
import concurrent.futures
import json
from datetime import timedelta
from tqdm import tqdm

# ── 注入环境变量让动态库能找到 ──
_BUNDLED_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
os.environ['PATH'] = _BUNDLED_DIR + os.pathsep + os.environ.get('PATH', '')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR CLI — 本地 llama.cpp ASR 转录 + SRT 字幕生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s -i speech.wav -m qwen3-asr
  %(prog)s -i video.mp4 -m qwen3-asr --llamacpp-url http://localhost:8080 -srt
  %(prog)s -i lecture.wav -c "量子物理术语" -srt -s
        """,
    )
    parser.add_argument("-i", "--input-file", required=True, help="输入音频/视频文件路径")
    parser.add_argument("-m", "--model", default="qwen3-asr", help="llama.cpp 中的模型名称 (默认: qwen3-asr)")
    parser.add_argument("--llamacpp-url", default="http://localhost:8080", help="llama.cpp server URL (默认: http://localhost:8080)")
    parser.add_argument("-c", "--context", default="", help="ASR 上下文提示文本")
    parser.add_argument("-j", "--num-threads", type=int, default=4, help="并行线程数 (默认: 4)")
    parser.add_argument("-d", "--vad-segment-threshold", type=int, default=120, help="VAD 分割目标时长/秒 (默认: 120)")
    parser.add_argument("-t", "--tmp-dir", default=None, help="临时文件目录 (默认: 系统临时目录)")
    parser.add_argument("-srt", "--save-srt", action="store_true", help="生成 SRT 字幕文件")
    parser.add_argument("-s", "--silence", action="store_true", help="静默模式，减少终端输出")
    parser.add_argument("-o", "--output", default=None, help="输出文件路径 (默认: 与输入文件同名)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 延迟导入（PyInstaller 兼容）
    from qwen3_asr_toolkit.llamacpp_asr import LlamaCppASR
    from qwen3_asr_toolkit.audio_tools import load_audio, process_vad, save_audio_file, WAV_SAMPLE_RATE
    import srt as srt_lib
    from silero_vad import load_silero_vad

    input_file = args.input_file
    context = args.context
    llamacpp_url = args.llamacpp_url
    model = args.model
    num_threads = args.num_threads
    vad_segment_threshold = args.vad_segment_threshold
    tmp_dir = args.tmp_dir or tempfile.mkdtemp(prefix="qwen3-asr-")
    save_srt = args.save_srt
    silence = args.silence

    # 验证输入文件
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}", file=sys.stderr)
        sys.exit(1)

    # 初始化 ASR 客户端
    asr_client = LlamaCppASR(llamacpp_url=llamacpp_url, model=model)

    # 加载音频
    if not silence:
        print(f"📂 加载音频: {input_file}")
    wav = load_audio(input_file)
    duration = len(wav) / WAV_SAMPLE_RATE
    if not silence:
        print(f"⏱️  音频时长: {duration:.2f}s")

    # VAD 分割
    segments = []
    if duration >= 180:
        if not silence:
            print("🔧 音频超过3分钟，使用 Silero VAD 分割...")
        try:
            vad_model = load_silero_vad(onnx=True)
            segments = process_vad(wav, vad_model, segment_threshold_s=vad_segment_threshold)
        except Exception as e:
            if not silence:
                print(f"⚠️  VAD 失败 ({e})，使用等长分割")
            max_chunk = vad_segment_threshold * WAV_SAMPLE_RATE
            for start in range(0, len(wav), max_chunk):
                end = min(start + max_chunk, len(wav))
                if end - start > 0:
                    segments.append((start, end, wav[start:end]))
    else:
        segments = [(0, len(wav), wav)]

    if not silence:
        print(f"📦 分割为 {len(segments)} 段")

    # 保存临时音频文件
    wav_paths = []
    for idx, (_, _, wav_data) in enumerate(segments):
        wav_path = os.path.join(tmp_dir, f"seg_{idx}.wav")
        save_audio_file(wav_data, wav_path)
        wav_paths.append(wav_path)

    # 并行 ASR 调用
    results = []
    # 修复 macOS multiprocessing 资源竞争
    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_map = {
            executor.submit(asr_client.asr, path, context): idx
            for idx, path in enumerate(wav_paths)
        }
        if not silence:
            pbar = tqdm(total=len(future_map), desc="🎤 ASR 转录中")
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                language, recog_text = future.result()
                results.append((idx, recog_text, language))
            except Exception as e:
                results.append((idx, f"[ERROR: {e}]", "Unknown"))
            if not silence:
                pbar.update(1)
        if not silence:
            pbar.close()

    # 按原始顺序拼接
    results.sort(key=lambda x: x[0])
    languages = [r[2] for r in results if r[2] != "Not Supported" and r[2] != "Unknown"]
    primary_lang = languages[0] if languages else "Unknown"
    full_text = " ".join(r[1] for r in results)

    # 从原始结果中提取语言信息
    import re
    lang_match = re.search(r'language (\w+)<asr_text>', full_text)
    if lang_match:
        primary_lang_raw = lang_match.group(1)
        if primary_lang_raw not in ("Not", "Unk"):
            primary_lang = primary_lang_raw
    full_text = re.sub(r'language \w+<asr_text>\s*', '', full_text)
    full_text = re.sub(r'</asr_text>\s*', '', full_text)
    full_text = full_text.strip()

    if not silence:
        print(f"\n🌐 检测语言: {primary_lang}")
        print(f"📝 转录结果:\n{full_text}\n")

    # 确定输出文件路径
    if args.output:
        output_base = os.path.splitext(args.output)[0]
    else:
        output_base = os.path.splitext(input_file)[0]
    txt_path = output_base + ".txt"

    # 保存 TXT
    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Language: {primary_lang}\n")
        f.write(full_text + "\n")
    print(f"✅ 转录已保存: {txt_path}")

    # 保存 SRT
    if save_srt:
        subtitles = []
        def clean_srt_text(t):
            t = re.sub(r'language \w+<asr_text>\s*', '', t)
            t = re.sub(r'</asr_text>\s*', '', t)
            return t.strip()

        for idx, (start_sample, end_sample, _), (_, text, _) in zip(
            range(len(segments)), segments, results
        ):
            sub = srt_lib.Subtitle(
                index=idx + 1,
                start=timedelta(seconds=start_sample / WAV_SAMPLE_RATE),
                end=timedelta(seconds=end_sample / WAV_SAMPLE_RATE),
                content=clean_srt_text(text),
            )
            subtitles.append(sub)
        srt_path = output_base + ".srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_lib.compose(subtitles))
        print(f"✅ 字幕已保存: {srt_path}")

    # 清理临时文件
    for path in wav_paths:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
