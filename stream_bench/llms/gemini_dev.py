import os

from google import genai
from google.genai import types

from stream_bench.llms.utils import retry_with_exponential_backoff

class GeminiDev:
    SAFETY_SETTINGS = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    def __init__(self, model_name: str = "gemini-1.0-pro") -> None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    @staticmethod
    def _extract_text(response) -> str:
        if response.text:
            return response.text
        chunks = []
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(text)
        return "".join(chunks)

    def _build_config(self, max_tokens: int, temperature: float, top_p: float, top_k: int):
        thinking_config = None
        if self.model_name.startswith("gemini-3"):
            # Gemini 3 uses hidden thinking tokens by default, which can starve short code / SQL outputs.
            thinking_config = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(
            max_output_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            safety_settings=self.SAFETY_SETTINGS,
            thinking_config=thinking_config,
        )

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, max_tokens=512, temperature=0.0, top_p=1, top_k=1) -> tuple[str, dict]:
        """Returns the tuple of the response text as well as other detailed info."""
        res = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._build_config(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ),
        )
        output_text = self._extract_text(res)
        input_tokens = self.client.models.count_tokens(model=self.model_name, contents=prompt).total_tokens
        output_tokens = getattr(getattr(res, "usage_metadata", None), "candidates_token_count", None)
        if output_tokens is None:
            output_tokens = (
                self.client.models.count_tokens(model=self.model_name, contents=output_text).total_tokens
                if output_text else 0
            )
        res_info = {
            "input": prompt,
            "output": output_text,
            "num_input_tokens": input_tokens,
            "num_output_tokens": output_tokens,
            "logprobs": []  # NOTE: currently the Gemini API does not provide logprobs
        }
        return output_text, res_info
