import json
import textwrap
import evaluate
from datasets import Dataset
from colorama import Fore, Style

from stream_bench.llms.oai_chat import OpenAIChat
from stream_bench.benchmarks.base import Bench
from stream_bench.benchmarks.utils import strip_all_lines, extract_json_string

class GSM8KBench(Bench):
    """Benchmark for the GSM8K dataset."""
    DATASET_PATH = "appier-ai-research/robust-finetuning"
    DATASET_NAME = "gsm8k"
    EVAL_LLM = "gpt-4o-mini-2024-07-18"  # for extracting the answer from LLMs' raw output

    def __init__(
        self,
        split: str = "test",
        seed: int = 42,
        feedback: str = "correctness",
        max_samples: int | None = None,
        streambench_window_size: int = 5,
        **kwargs
    ) -> None:
        super().__init__({})
        self.split = split
        self.seed = seed
        self.feedback = feedback
        self.max_samples = max_samples
        if streambench_window_size <= 0:
            raise ValueError("streambench_window_size must be a positive integer")
        self.streambench_window_size = streambench_window_size
        self.eval_func = evaluate.load("exact_match")
        self.llm = OpenAIChat(model_name=self.EVAL_LLM)
        self.window_scores = []
        self.window_deltas = []

    def get_dataset(self) -> Dataset:
        dataset = self.dataset[self.split].shuffle(seed=self.seed)
        if self.max_samples is not None:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))
        return dataset

    @staticmethod
    def get_zeroshot_prompt(question: str) -> str:
        return question

    def get_input(self, row: dict) -> dict:
        row_input = dict()
        row_input["question"] = row["question"]
        # Keep GT-agent output compatible with postprocess_generation(),
        # which expects to extract the final numeric answer from text/JSON.
        row_input["label_text"] = json.dumps({"answer": self.get_output(row)["answer"]})
        row_input["prompt_zeroshot"] = self.get_zeroshot_prompt(row["question"])
        return row_input

    def get_output(self, row: dict) -> dict:
        rna = row["answer"].split("####")
        rationale = rna[0].strip()
        answer = rna[1].strip()
        if ',' in answer:
            answer = ''.join(answer.split(','))
        return {"rationale": rationale, "answer": answer}

    def get_metrics(self) -> dict:
        metrics = self.eval_func.compute(
            predictions=self.predictions,
            references=self.references,
            ignore_punctuation=True
        )
        metrics.update({
            "streambench_window_size": self.streambench_window_size,
            "streambench_num_windows": len(self.window_scores),
            "streambench_window_scores": self.window_scores,
            "streambench_window_deltas": self.window_deltas,
        })
        if self.window_scores:
            metrics["streambench_last_window_em"] = self.window_scores[-1]
            metrics["streambench_best_window_em"] = max(self.window_scores)
        if self.window_deltas:
            metrics["streambench_last_window_delta"] = self.window_deltas[-1]
            metrics["streambench_max_window_gain"] = max(self.window_deltas)
            metrics["streambench_max_window_drop"] = min(self.window_deltas)
        return metrics

    def compute_streambench(self, predictions: list[str], references: list[str]) -> float:
        return self.eval_func.compute(
            predictions=predictions,
            references=references,
            ignore_punctuation=True
        )["exact_match"]

    def should_evaluate_window(self) -> bool:
        num_samples = len(self.predictions)
        return num_samples > 0 and (num_samples % self.streambench_window_size == 0)

    def evaluate_recent_window(self) -> dict | None:
        if not self.should_evaluate_window():
            return None

        window_end = len(self.predictions)
        window_start = window_end - self.streambench_window_size
        window_score = self.compute_streambench(
            predictions=self.predictions[window_start:window_end],
            references=self.references[window_start:window_end]
        )

        self.window_scores.append(window_score)
        metrics = {
            "streambench_window_closed": 1,
            "streambench_window_index": len(self.window_scores) - 1,
            "streambench_window_start": window_start,
            "streambench_window_end": window_end - 1,
            "streambench_window_em": window_score,
        }
        if len(self.window_scores) >= 2:
            window_delta = self.window_scores[-1] - self.window_scores[-2]
            self.window_deltas.append(window_delta)
            metrics["streambench_window_delta"] = window_delta
        return metrics

    def postprocess_generation(self, res: str, idx: int = -1) -> str:
        # Parse out the answer with a cheap LLM
        prompt_extract = strip_all_lines("""\
        The following text is an LLM's response to a math question:

        Text (enclosed in triple quotes): '''{text}'''

        Extract the answer from the text (only extract the digits, potentially the sign if the number is negative), and provide it in the following JSON format:
        {{"answer": "<digits>"}}""")
        prompt = prompt_extract.format(text=res)
        answer_str, _ = self.llm(prompt)
        answer_json = extract_json_string(answer_str)
        try:
            answer = json.loads(answer_json)["answer"]
        except (json.JSONDecodeError, KeyError) as e:
            print(Fore.RED + str(e) + Style.RESET_ALL)
            answer = res
        return answer

    def process_results(
        self,
        prediction: str,
        label: dict,
        return_details: bool = True,
        **kwargs
    ) -> bool | dict:
        answer = label["answer"]
        correct = self.eval_func.compute(
            predictions=[prediction],
            references=[answer]
        )["exact_match"]
        self.n_correct += correct
        self.predictions.append(prediction)
        self.references.append(answer)
        if return_details:
            metrics = {
                "correct": int(correct),
                "n_correct": self.n_correct,
                "streambench_window_closed": 0,
                "rolling_em": self.get_metrics()["exact_match"]
            }
            window_metrics = self.evaluate_recent_window()
            if window_metrics is not None:
                metrics.update(window_metrics)
            return metrics
        return bool(correct)

    def give_feedback(self, model_output: str, row: dict, res: dict) -> tuple[bool, dict]:
        return True, {}
