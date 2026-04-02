import argparse
import csv
import json
import os
import shutil
import ssl
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import error, request


API_URL = "https://api.openai.com/v1/responses"
BUCKET_ORDER = ["0-9", "10-39", "40-69", "70-89", "90-100"]
DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_GENERATION_BATCH_SIZE = 50
DEFAULT_SCORING_BATCH_SIZE = 5
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_RETRY_LIMIT = 5
DEFAULT_DATASET_PATH = "issue_priority_dataset.csv"
DEFAULT_BACKUP_DIR = "backups"
DEFAULT_BOOTSTRAP_COUNT = 1000

PROMPT_FILES = {
    "general": "generate_issues_prompt.txt",
    "minimal": "generate_minimal_severity_issues_prompt.txt",
    "low": "generate_low_severity_issues_prompt.txt",
    "medium": "generate_medium_severity_issues_prompt.txt",
    "critical": "generate_critical_severity_issues_prompt.txt",
}

PROFILE_BY_BUCKET = {
    "0-9": "minimal",
    "10-39": "low",
    "40-69": "medium",
    "70-89": "general",
    "90-100": "critical",
}


class DatasetGenerationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append synthetic issues to a dataset while balancing score buckets with OpenAI Responses API calls."
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="CSV dataset path. Existing files are backed up and appended to, never rewritten.",
    )
    parser.add_argument(
        "--backup-dir",
        default=DEFAULT_BACKUP_DIR,
        help="Directory for timestamped full-copy backups created before each append.",
    )
    parser.add_argument(
        "--target-bucket-count",
        type=int,
        help="Target count for each score bucket. Defaults to the current largest bucket in the dataset.",
    )
    parser.add_argument(
        "--bootstrap-count",
        type=int,
        default=DEFAULT_BOOTSTRAP_COUNT,
        help="Fallback row count when creating a new dataset from scratch.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=DEFAULT_GENERATION_BATCH_SIZE,
        help="Maximum number of issues to request per generation call.",
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
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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


def bucket_for_score(score: int) -> str:
    if 0 <= score <= 9:
        return "0-9"
    if 10 <= score <= 39:
        return "10-39"
    if 40 <= score <= 69:
        return "40-69"
    if 70 <= score <= 89:
        return "70-89"
    if 90 <= score <= 100:
        return "90-100"
    raise DatasetGenerationError(f"Score out of range: {score}")


def empty_bucket_counter() -> Counter[str]:
    return Counter({bucket: 0 for bucket in BUCKET_ORDER})


def choose_bucket_to_fill(counts: Counter[str], target_bucket_count: int) -> str | None:
    best_bucket = None
    best_deficit = 0
    for bucket in BUCKET_ORDER:
        deficit = target_bucket_count - counts[bucket]
        if deficit > best_deficit:
            best_deficit = deficit
            best_bucket = bucket
    return best_bucket


def timestamp_for_filename() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def validate_args(args: argparse.Namespace) -> None:
    if args.target_bucket_count is not None and args.target_bucket_count <= 0:
        raise DatasetGenerationError("--target-bucket-count must be greater than 0.")
    if args.bootstrap_count <= 0:
        raise DatasetGenerationError("--bootstrap-count must be greater than 0.")
    if args.generation_batch_size <= 0:
        raise DatasetGenerationError("--generation-batch-size must be greater than 0.")
    if args.scoring_batch_size <= 0:
        raise DatasetGenerationError("--scoring-batch-size must be greater than 0.")
    if args.max_concurrency <= 0:
        raise DatasetGenerationError("--max-concurrency must be greater than 0.")
    if args.retry_limit <= 0:
        raise DatasetGenerationError("--retry-limit must be greater than 0.")


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
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            "reasoning": {"effort": self.reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema["name"],
                    "schema": schema["schema"],
                    "strict": schema.get("strict", True),
                }
            },
        }
        return self._extract_structured_output(self._post_json(payload))

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


class DatasetStore:
    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path

    def exists(self) -> bool:
        return self.dataset_path.exists()

    def load_state(self) -> tuple[int, Counter[str]]:
        counts = empty_bucket_counter()
        max_issue_id = 0
        if not self.exists():
            return max_issue_id, counts

        with self.dataset_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected = {"issue_id", "issue_description", "priority_score"}
            if set(reader.fieldnames or []) != expected:
                raise DatasetGenerationError(
                    f"Dataset columns must be {sorted(expected)}, got {reader.fieldnames}"
                )
            for row in reader:
                issue_id = int(row["issue_id"])
                score = int(row["priority_score"])
                max_issue_id = max(max_issue_id, issue_id)
                counts[bucket_for_score(score)] += 1
        return max_issue_id, counts

    def create_backup(self, backup_dir: Path) -> Path | None:
        if not self.exists():
            return None
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{self.dataset_path.stem}.{timestamp_for_filename()}{self.dataset_path.suffix}"
        shutil.copy2(self.dataset_path, backup_path)
        return backup_path

    def append_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.exists()
        with self.dataset_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["issue_id", "issue_description", "priority_score"])
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


class IssueDatasetBuilder:
    def __init__(
        self,
        client: OpenAIResponsesClient,
        prompts: dict[str, str],
        scoring_prompt: str,
        generation_schema: dict[str, Any],
        scoring_schema: dict[str, Any],
        generation_batch_size: int,
        scoring_batch_size: int,
        max_concurrency: int,
        retry_limit: int,
    ) -> None:
        self.client = client
        self.prompts = prompts
        self.scoring_prompt = scoring_prompt
        self.generation_schema = generation_schema
        self.scoring_schema = scoring_schema
        self.generation_batch_size = generation_batch_size
        self.scoring_batch_size = scoring_batch_size
        self.max_concurrency = max_concurrency
        self.retry_limit = retry_limit
        self._print_lock = threading.Lock()

    def bootstrap_rows(self, count: int, next_issue_id: int) -> list[dict[str, Any]]:
        generated = self._generate_issues("general", count)
        scores = self._score_issues(generated)
        rows = []
        for issue_text, score in zip(generated, scores):
            rows.append(
                {
                    "issue_id": next_issue_id,
                    "issue_description": issue_text,
                    "priority_score": score,
                }
            )
            next_issue_id += 1
        return rows

    def build_rows_for_balance(
        self,
        current_counts: Counter[str],
        target_bucket_count: int,
        next_issue_id: int,
    ) -> tuple[list[dict[str, Any]], Counter[str]]:
        accepted_rows: list[dict[str, Any]] = []
        updated_counts = Counter(current_counts)
        stalled_attempts = Counter()

        while True:
            bucket = choose_bucket_to_fill(updated_counts, target_bucket_count)
            if bucket is None:
                return accepted_rows, updated_counts

            deficit = target_bucket_count - updated_counts[bucket]
            profile = PROFILE_BY_BUCKET[bucket]
            request_count = min(self.generation_batch_size, max(deficit, self.scoring_batch_size))
            self._log(
                f"Targeting bucket {bucket} with profile {profile}; "
                f"current={updated_counts[bucket]} target={target_bucket_count} request={request_count}."
            )

            issues = self._generate_issues(profile, request_count)
            scores = self._score_issues(issues)

            useful_rows = 0
            for issue_text, score in zip(issues, scores):
                scored_bucket = bucket_for_score(score)
                if updated_counts[scored_bucket] >= target_bucket_count:
                    continue
                accepted_rows.append(
                    {
                        "issue_id": next_issue_id,
                        "issue_description": issue_text,
                        "priority_score": score,
                    }
                )
                updated_counts[scored_bucket] += 1
                next_issue_id += 1
                useful_rows += 1

            if useful_rows == 0:
                stalled_attempts[bucket] += 1
                if stalled_attempts[bucket] >= self.retry_limit:
                    raise DatasetGenerationError(
                        f"No usable rows were produced for bucket {bucket} after {self.retry_limit} attempts."
                    )
            else:
                stalled_attempts[bucket] = 0

    def _generate_issues(self, profile: str, count: int) -> list[str]:
        response_data = self.client.create_structured_response(
            system_prompt=self._generation_system_prompt(profile, count),
            user_prompt=f"Generate {count} issue descriptions.",
            schema=self.generation_schema,
        )
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
        return cleaned[:count]

    def _generation_system_prompt(self, profile: str, count: int) -> str:
        prompt = self.prompts[profile]
        return (
            f"{prompt}\n\n"
            "Output contract:\n"
            "Return a single JSON object matching the provided schema.\n"
            'The object must have an "issues" key whose value is an array of strings.\n'
            f"Generate exactly {count} issue descriptions in that array."
        )

    def _score_issues(self, issues: list[str]) -> list[int]:
        batches = chunked(issues, self.scoring_batch_size)
        final_scores: list[int | None] = [None] * len(issues)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {}
            for batch_number, issue_batch in enumerate(batches, start=1):
                start_index = (batch_number - 1) * self.scoring_batch_size
                future = executor.submit(self._score_single_batch, batch_number, issue_batch)
                futures[future] = (start_index, len(issue_batch))

            for future in as_completed(futures):
                start_index, batch_len = futures[future]
                batch_scores = future.result()
                if len(batch_scores) != batch_len:
                    raise DatasetGenerationError(
                        f"Scoring batch returned {len(batch_scores)} scores for {batch_len} issues."
                    )
                for offset, score in enumerate(batch_scores):
                    final_scores[start_index + offset] = score

        if any(score is None for score in final_scores):
            raise DatasetGenerationError("Missing scores after scoring completed.")
        return [int(score) for score in final_scores]

    def _score_single_batch(self, batch_number: int, issues: list[str]) -> list[int]:
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
                return self._validate_scoring_response(response_data, len(issues))
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


def bucket_counts_for_rows(rows: list[dict[str, Any]]) -> Counter[str]:
    counts = empty_bucket_counter()
    for row in rows:
        counts[bucket_for_score(int(row["priority_score"]))] += 1
    return counts


def main() -> int:
    args = parse_args()
    validate_args(args)

    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise DatasetGenerationError("OPENAI_API_KEY is not set. Add it to .env or the environment.")

    prompts = {name: load_text(root / filename) for name, filename in PROMPT_FILES.items()}
    scoring_prompt = load_text(root / "rate_issues_prompt.txt")
    generation_schema = load_json(root / "issues_schema.json")
    scoring_schema = load_json(root / "issues_scoring.json")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    backup_dir = Path(args.backup_dir)
    if not backup_dir.is_absolute():
        backup_dir = root / backup_dir

    store = DatasetStore(dataset_path)
    max_issue_id, current_counts = store.load_state()
    next_issue_id = max_issue_id + 1

    client = OpenAIResponsesClient(
        api_key=api_key,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        retry_limit=args.retry_limit,
    )
    builder = IssueDatasetBuilder(
        client=client,
        prompts=prompts,
        scoring_prompt=scoring_prompt,
        generation_schema=generation_schema,
        scoring_schema=scoring_schema,
        generation_batch_size=args.generation_batch_size,
        scoring_batch_size=args.scoring_batch_size,
        max_concurrency=args.max_concurrency,
        retry_limit=args.retry_limit,
    )

    if store.exists():
        current_max_bucket = max(current_counts.values()) if current_counts else 0
        target_bucket_count = args.target_bucket_count or current_max_bucket
        if target_bucket_count < current_max_bucket:
            raise DatasetGenerationError(
                "--target-bucket-count cannot be less than the current largest bucket count."
            )
        rows_to_append, final_counts = builder.build_rows_for_balance(
            current_counts=current_counts,
            target_bucket_count=target_bucket_count,
            next_issue_id=next_issue_id,
        )
    else:
        target_bucket_count = args.target_bucket_count
        rows_to_append = builder.bootstrap_rows(args.bootstrap_count, next_issue_id)
        final_counts = bucket_counts_for_rows(rows_to_append)

    if not rows_to_append:
        print("No rows needed; dataset already satisfies the target.")
        return 0

    backup_path = store.create_backup(backup_dir)
    store.append_rows(rows_to_append)

    print(f"Appended {len(rows_to_append)} rows to {dataset_path}")
    if backup_path:
        print(f"Backup created at {backup_path}")
    if target_bucket_count is not None:
        for bucket in BUCKET_ORDER:
            print(f"{bucket}: {final_counts[bucket]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetGenerationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
