import os
import base64
import time
import random
import requests
from openai import OpenAI

MAX_API_RETRY = 10
API_RETRY_SLEEP = (1, 2)


language_code_mapping = {
    "ar": "Arabic",
    "zh": "Chinese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish"
}


class LlamaCppASR:
    def __init__(self, llamacpp_url: str = "http://localhost:8080", model: str = "qwen3-asr"):
        self.llamacpp_url = llamacpp_url
        self.model = model
        self.client = OpenAI(base_url=f"{llamacpp_url}/v1", api_key="not-needed")

    def post_text_process(self, text, threshold=20):
        def fix_char_repeats(s, thresh):
            res = []
            i = 0
            n = len(s)
            while i < n:
                count = 1
                while i + count < n and s[i + count] == s[i]:
                    count += 1

                if count > thresh:
                    res.append(s[i])
                    i += count
                else:
                    res.append(s[i:i + count])
                    i += count
            return ''.join(res)

        def fix_pattern_repeats(s, thresh, max_len=20):
            n = len(s)
            min_repeat_chars = thresh * 2
            if n < min_repeat_chars:
                return s

            i = 0
            result = []
            while i <= n - min_repeat_chars:
                found = False
                for k in range(1, max_len + 1):
                    if i + k * thresh > n:
                        break

                    pattern = s[i:i + k]

                    valid = True
                    for rep in range(1, thresh):
                        start_idx = i + rep * k
                        if s[start_idx:start_idx + k] != pattern:
                            valid = False
                            break

                    if valid:
                        total_rep = thresh
                        end_index = i + thresh * k
                        while end_index + k <= n and s[end_index:end_index + k] == pattern:
                            total_rep += 1
                            end_index += k

                        result.append(pattern)
                        result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                        i = n
                        found = True
                        break

                if found:
                    break
                else:
                    result.append(s[i])
                    i += 1

            if not found:
                result.append(s[i:])
            return ''.join(result)

        text = fix_char_repeats(text, threshold)
        return fix_pattern_repeats(text, threshold)

    def _read_audio_base64(self, audio_path: str) -> str:
        if audio_path.startswith(("http://", "https://")):
            response = requests.get(audio_path, timeout=60)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")

        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def asr(self, wav_url: str, context: str = ""):
        if not wav_url.startswith(("http://", "https://")):
            assert os.path.exists(wav_url), f"{wav_url} not exists!"

        audio_data = self._read_audio_base64(wav_url)

        for attempt in range(MAX_API_RETRY):
            try:
                messages = [
                    {"role": "system", "content": context},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Transcribe this audio"},
                        {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}}
                    ]}
                ]

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )

                recog_text = response.choices[0].message.content
                if recog_text is None:
                    recog_text = ""

                language = "Not Supported"

                return language, self.post_text_process(recog_text)
            except Exception as e:
                try:
                    print(f"Retry {attempt + 1}...  {wav_url}\n{response}")
                except Exception:
                    print(f"Retry {attempt + 1}...  {wav_url}\n{e}")
            time.sleep(random.uniform(*API_RETRY_SLEEP))
        raise Exception(f"{wav_url} task failed!")


if __name__ == "__main__":
    asr = LlamaCppASR(llamacpp_url="http://localhost:8080", model="qwen3-asr")
    asr_text = asr.asr(wav_url="/path/to/your/wav_file.wav")
    print(asr_text)
