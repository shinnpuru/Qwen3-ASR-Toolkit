from qwen3_asr_toolkit.llamacpp_asr import LlamaCppASR, language_code_mapping


class QwenASR(LlamaCppASR):
    """Backward-compatible class name for the Qwen3 ASR interface.

    Uses a local llama.cpp server for ASR via an OpenAI-compatible API.
    """
    pass


if __name__ == "__main__":
    qwen_asr = QwenASR(llamacpp_url="http://localhost:8080", model="qwen3-asr")
    asr_text = qwen_asr.asr(wav_url="/path/to/your/wav_file.wav")
    print(asr_text)
