import argparse
import csv
import json
import os
import ssl
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import error, request


API_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_GENERATION_BATCH_SIZE = 100
DEFAULT_SCORING_BATCH_SIZE = 5
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_RETRY_LIMIT = 5


class DatasetGenerationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic investigation issues and batch-score them with the OpenAI Responses API."
    )
    parser.add_argument("--count", type=int, default=1000, help="Number of unique issues to generate.")
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=DEFAULT_GENERATION_BATCH_SIZE,
        help="Number of issues to request per generation call.",
    )
    parser.add_argument(
        "--scoring-batch-size",
        type=int,
        default=DEFAULT_SCORING_BATCH_SIZE,
        help="Number of issues to score per scoring call.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Maximum number of concurrent scoring requests.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model name. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort sent to the Responses API.",
    )
    parser.add_argument(
        "--out",
        default="issue_priority_dataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--retry-limit",
        type=int,
        default=DEFAULT_RETRY_LIMIT,
        help="Maximum retry attempts for a failed generation or scoring request.",
    )
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetGenerationError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetGenerationError(f"Invalid JSON in {path}: {exc}") from exc


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise DatasetGenerationError(f"Missing required file: {path}") from exc


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


class OpenAIResponsesClient:
    def __init__(self, api_key: str, model: str, reasoning_effort: str, retry_limit: int) -> None:
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.retry_limit = retry_limit
        self.ssl_context = self._build_ssl_context()

    def create_structured_response(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt,
                        }
                    ],
                },
            ],
            "reasoning": {
                "effort": self.reasoning_effort,
            },
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema["name"],
                    "schema": schema["schema"],
                    "strict": schema.get("strict", True),
                }
            },
        }
        response = self._post_json(payload)
        return self._extract_structured_output(response)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.retry_limit + 1):
            req = request.Request(API_URL, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=180, context=self.ssl_context) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                status = exc.code
                raw_payload = exc.read().decode("utf-8", errors="replace")
                last_error = DatasetGenerationError(
                    f"Responses API request failed with HTTP {status}: {raw_payload}"
                )
                if status in {408, 409, 429, 500, 502, 503, 504} and attempt < self.retry_limit:
                    self._sleep_for_retry(attempt)
                    continue
                raise last_error from exc
            except error.URLError as exc:
                last_error = DatasetGenerationError(f"Responses API connection failed: {exc}")
                if attempt < self.retry_limit:
                    self._sleep_for_retry(attempt)
                    continue
                raise last_error from exc

        raise last_error or DatasetGenerationError("Responses API request failed without an explicit error.")

    @staticmethod
    def _sleep_for_retry(attempt: int) -> None:
        time.sleep(min(2 ** (attempt - 1), 16))

    @staticmethod
    def _build_ssl_context() -> ssl.SSLContext:
        candidate_paths = [
            os.getenv("OPENAI_CA_BUNDLE"),
            os.getenv("SSL_CERT_FILE"),
            ssl.get_default_verify_paths().cafile,
            r"C:\msys64\usr\ssl\cert.pem",
        ]

        for candidate in candidate_paths:
            if candidate and Path(candidate).exists():
                return ssl.create_default_context(cafile=candidate)

        return ssl.create_default_context()

    @staticmethod
    def _extract_structured_output(response: dict[str, Any]) -> dict[str, Any]:
        if response.get("status") == "incomplete":
            raise DatasetGenerationError(f"Response incomplete: {response.get('incomplete_details')}")
        if response.get("error"):
            raise DatasetGenerationError(f"Response returned an error: {response['error']}")

        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            try:
                return json.loads(output_text)
            except json.JSONDecodeError as exc:
                raise DatasetGenerationError(f"Invalid JSON in output_text: {exc}") from exc

        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    try:
                        return json.loads(content["text"])
                    except json.JSONDecodeError as exc:
                        raise DatasetGenerationError(f"Invalid JSON in response content: {exc}") from exc

        raise DatasetGenerationError("No structured JSON content found in Responses API reply.")


class IssueDatasetBuilder:
    def __init__(
        self,
        client: OpenAIResponsesClient,
        generation_prompt: str,
        scoring_prompt: str,
        generation_schema: dict[str, Any],
        scoring_schema: dict[str, Any],
        generation_batch_size: int,
        scoring_batch_size: int,
        max_concurrency: int,
        retry_limit: int,
    ) -> None:
        self.client = client
        self.generation_prompt = generation_prompt
        self.scoring_prompt = scoring_prompt
        self.generation_schema = generation_schema
        self.scoring_schema = scoring_schema
        self.generation_batch_size = generation_batch_size
        self.scoring_batch_size = scoring_batch_size
        self.max_concurrency = max_concurrency
        self.retry_limit = retry_limit
        self._print_lock = threading.Lock()

    def generate_dataset(self, count: int) -> list[dict[str, Any]]:
        issues = self._generate_issues(count)
        scores = self._score_issues(issues)
        return [
            {
                "issue_id": issue_index + 1,
                "issue_description": issue_text,
                "priority_score": scores[issue_index],
            }
            for issue_index, issue_text in enumerate(issues)
        ]

    def _generate_issues(self, count: int) -> list[str]:
        unique_issues: list[str] = []
        seen: set[str] = set()
        batch_number = 0

        while len(unique_issues) < count:
            remaining = count - len(unique_issues)
            requested = min(self.generation_batch_size, remaining)
            batch_number += 1
            self._log(f"Generating issue batch {batch_number} for {requested} items.")
            response_data = self.client.create_structured_response(
                system_prompt=self._generation_system_prompt(requested),
                user_prompt=f"Generate {requested} issue descriptions.",
                schema=self.generation_schema,
            )
            batch_issues = self._validate_generated_issues(response_data)

            added = 0
            for issue in batch_issues:
                if issue not in seen:
                    seen.add(issue)
                    unique_issues.append(issue)
                    added += 1
                    if len(unique_issues) == count:
                        break

            if added == 0:
                raise DatasetGenerationError(
                    "Generation batch produced no new unique issues. Stop to avoid an infinite retry loop."
                )

        return unique_issues

    def _generation_system_prompt(self, batch_size: int) -> str:
        return (
            f"{self.generation_prompt}\n\n"
            "Output contract:\n"
            "Return a single JSON object matching the provided schema.\n"
            'The object must have an "issues" key whose value is an array of strings.\n'
            f"Generate exactly {batch_size} issue descriptions in that array."
        )

    def _validate_generated_issues(self, response_data: dict[str, Any]) -> list[str]:
        issues = response_data.get("issues")
        if not isinstance(issues, list):
            raise DatasetGenerationError("Generation response is missing an 'issues' array.")

        cleaned: list[str] = []
        for item in issues:
            if not isinstance(item, str):
                raise DatasetGenerationError("Generation response contained a non-string issue.")
            normalized = " ".join(item.strip().split())
            if normalized:
                cleaned.append(normalized)

        if not cleaned:
            raise DatasetGenerationError("Generation response contained no usable issues.")
        return cleaned

    def _score_issues(self, issues: list[str]) -> list[int]:
        score_batches = chunked(issues, self.scoring_batch_size)
        final_scores: list[int | None] = [None] * len(issues)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {}
            for batch_number, issue_batch in enumerate(score_batches, start=1):
                start_index = (batch_number - 1) * self.scoring_batch_size
                future = executor.submit(self._score_single_batch, batch_number, start_index, issue_batch)
                futures[future] = (batch_number, start_index, len(issue_batch))

            for future in as_completed(futures):
                batch_number, start_index, batch_len = futures[future]
                batch_scores = future.result()
                if len(batch_scores) != batch_len:
                    raise DatasetGenerationError(
                        f"Scoring batch {batch_number} returned {len(batch_scores)} scores for {batch_len} issues."
                    )
                for offset, score in enumerate(batch_scores):
                    final_scores[start_index + offset] = score

        unresolved = [index for index, score in enumerate(final_scores) if score is None]
        if unresolved:
            raise DatasetGenerationError(f"Missing final scores for indexes: {unresolved}")

        return [int(score) for score in final_scores]

    def _score_single_batch(self, batch_number: int, start_index: int, issues: list[str]) -> list[int]:
        for attempt in range(1, self.retry_limit + 1):
            self._log(
                f"Scoring batch {batch_number} with {len(issues)} issues "
                f"(attempt {attempt}/{self.retry_limit})."
            )
            response_data = self.client.create_structured_response(
                system_prompt=self._scoring_system_prompt(len(issues)),
                user_prompt=self._scoring_user_prompt(issues),
                schema=self.scoring_schema,
            )

            try:
                batch_scores = self._validate_scoring_response(response_data, len(issues))
                self._log(
                    f"Scoring batch {batch_number} completed for global rows "
                    f"{start_index + 1}-{start_index + len(issues)}."
                )
                return batch_scores
            except DatasetGenerationError:
                if attempt >= self.retry_limit:
                    raise
                time.sleep(min(2 ** (attempt - 1), 16))

        raise DatasetGenerationError(f"Scoring batch {batch_number} exhausted retry attempts.")

    def _scoring_system_prompt(self, batch_size: int) -> str:
        return (
            f"{self.scoring_prompt}\n\n"
            "Output contract:\n"
            "Return one JSON object matching the provided schema.\n"
            'Return a "scores" array with exactly one object per submitted issue.\n'
            'Each object must contain "issue_index" and "score".\n'
            f"Cover every issue index from 0 to {batch_size - 1} exactly once."
        )

    @staticmethod
    def _scoring_user_prompt(issues: list[str]) -> str:
        lines = ["Score every issue in this batch.", "Batch items:"]
        lines.extend(f"{index}: {issue}" for index, issue in enumerate(issues))
        return "\n".join(lines)

    @staticmethod
    def _validate_scoring_response(response_data: dict[str, Any], batch_size: int) -> list[int]:
        scores = response_data.get("scores")
        if not isinstance(scores, list):
            raise DatasetGenerationError("Scoring response is missing a 'scores' array.")
        if len(scores) != batch_size:
            raise DatasetGenerationError(
                f"Scoring response returned {len(scores)} items for a batch of {batch_size}."
            )

        indexed_scores: dict[int, int] = {}
        for item in scores:
            if not isinstance(item, dict):
                raise DatasetGenerationError("Scoring response contained a non-object score entry.")
            issue_index = item.get("issue_index")
            score = item.get("score")
            if not isinstance(issue_index, int):
                raise DatasetGenerationError("Scoring response contained a non-integer issue_index.")
            if not isinstance(score, int):
                raise DatasetGenerationError("Scoring response contained a non-integer score.")
            if issue_index < 0 or issue_index >= batch_size:
                raise DatasetGenerationError(f"Scoring response issue_index out of range: {issue_index}")
            if score < 0 or score > 100:
                raise DatasetGenerationError(f"Scoring response score out of range: {score}")
            if issue_index in indexed_scores:
                raise DatasetGenerationError(f"Duplicate issue_index in scoring response: {issue_index}")
            indexed_scores[issue_index] = score

        missing = [index for index in range(batch_size) if index not in indexed_scores]
        if missing:
            raise DatasetGenerationError(f"Scoring response omitted issue indexes: {missing}")

        return [indexed_scores[index] for index in range(batch_size)]

    def _log(self, message: str) -> None:
        with self._print_lock:
            print(message, file=sys.stderr)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["issue_id", "issue_description", "priority_score"])
        writer.writeheader()
        writer.writerows(rows)


def validate_args(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise DatasetGenerationError("--count must be greater than 0.")
    if args.generation_batch_size <= 0:
        raise DatasetGenerationError("--generation-batch-size must be greater than 0.")
    if args.scoring_batch_size <= 0:
        raise DatasetGenerationError("--scoring-batch-size must be greater than 0.")
    if args.max_concurrency <= 0:
        raise DatasetGenerationError("--max-concurrency must be greater than 0.")
    if args.retry_limit <= 0:
        raise DatasetGenerationError("--retry-limit must be greater than 0.")


def main() -> int:
    args = parse_args()
    validate_args(args)

    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise DatasetGenerationError("OPENAI_API_KEY is not set. Add it to .env or the environment.")

    generation_prompt = load_text(root / "generate_issues_prompt.txt")
    scoring_prompt = load_text(root / "rate_issues_prompt.txt")
    generation_schema = load_json(root / "issues_schema.json")
    scoring_schema = load_json(root / "issues_scoring.json")

    client = OpenAIResponsesClient(
        api_key=api_key,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        retry_limit=args.retry_limit,
    )
    builder = IssueDatasetBuilder(
        client=client,
        generation_prompt=generation_prompt,
        scoring_prompt=scoring_prompt,
        generation_schema=generation_schema,
        scoring_schema=scoring_schema,
        generation_batch_size=args.generation_batch_size,
        scoring_batch_size=args.scoring_batch_size,
        max_concurrency=args.max_concurrency,
        retry_limit=args.retry_limit,
    )

    rows = builder.generate_dataset(args.count)
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = root / output_path
    write_csv(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetGenerationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
