import os
from openai import OpenAI

from stream_bench.llms.base import LLM
from .utils import retry_with_exponential_backoff

class OpenAIChat(LLM):
    TOP_LOGPROBS = 1

    def __init__(self, model_name='gpt-3.5-turbo-0125') -> None:
        api_key = (
            os.getenv('OAI_KEY')
            or os.getenv('OPENAI_API_KEY')
            or os.getenv('DASHSCOPE_API_KEY')
        )
        if not api_key:
            raise KeyError("Missing API key. Set one of OAI_KEY, OPENAI_API_KEY, or DASHSCOPE_API_KEY.")

        params = {'api_key': api_key}
        base_url = (
            os.getenv('CUSTOM_API_URL')
            or os.getenv('OPENAI_BASE_URL')
            or os.getenv('DASHSCOPE_BASE_URL')
        )
        if base_url:
            params['base_url'] = base_url
        self.client = OpenAI(**params)
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, max_tokens: int = 1024, temperature=0.0, **kwargs) -> tuple[str, dict]:
        request_kwargs = dict(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            logprobs=True,
            top_logprobs=self.TOP_LOGPROBS,
            **kwargs
        )

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            error_text = str(exc).lower()
            if "logprobs" not in error_text:
                raise
            request_kwargs.pop("logprobs", None)
            request_kwargs.pop("top_logprobs", None)
            response = self.client.chat.completions.create(**request_kwargs)

        res_text = response.choices[0].message.content
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        logprobs = []
        choice_logprobs = getattr(response.choices[0], "logprobs", None)
        if choice_logprobs is not None and getattr(choice_logprobs, "content", None) is not None:
            log_prob_seq = choice_logprobs.content
            logprobs = [
                [{"token": pos_info.token, "logprob": pos_info.logprob} for pos_info in position.top_logprobs]
                for position in log_prob_seq
            ]
            if completion_tokens is None:
                completion_tokens = len(log_prob_seq)

        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": prompt_tokens,
            "num_output_tokens": completion_tokens,
            "logprobs": logprobs,
        }
        return res_text, res_info

if __name__ == "__main__":
    from pprint import pprint
    llm = OpenAIChat()
    res_text, res_info = llm(prompt="Say apple!")
    print(res_text)
    print()
    pprint(res_info)
