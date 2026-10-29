"""Build a reproducible binary projection of 20 MMLU test subjects.

This script never loads a language model.  Install ``pyarrow==23.0.1`` only when
rebuilding from source; the committed JSON is sufficient for all experiments.
Network access is opt-in (``--download``).  Without it, source Parquet files
must already exist in ``--cache-dir``.  Source rows are never rewritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import urllib.request


DATASET_ID = "cais/mmlu"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
SEED = 42
ITEMS_PER_SUBJECT = 20
SUBJECTS = (
    ("elementary_mathematics", "STEM", "elementary"),
    ("high_school_mathematics", "STEM", "high_school"),
    ("college_mathematics", "STEM", "college"),
    ("formal_logic", "humanities", "unspecified"),
    ("high_school_computer_science", "STEM", "high_school"),
    ("college_computer_science", "STEM", "college"),
    ("high_school_physics", "STEM", "high_school"),
    ("college_physics", "STEM", "college"),
    ("high_school_chemistry", "STEM", "high_school"),
    ("college_chemistry", "STEM", "college"),
    ("high_school_biology", "STEM", "high_school"),
    ("college_biology", "STEM", "college"),
    ("high_school_world_history", "humanities", "high_school"),
    ("high_school_us_history", "humanities", "high_school"),
    ("high_school_geography", "social_sciences", "high_school"),
    ("high_school_macroeconomics", "social_sciences", "high_school"),
    ("high_school_microeconomics", "social_sciences", "high_school"),
    ("high_school_psychology", "social_sciences", "high_school"),
    ("philosophy", "humanities", "unspecified"),
    ("conceptual_physics", "STEM", "unspecified"),
)

LICENSE_TEXT = """MIT License

Copyright (c) 2020 Dan Hendrycks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

DISTRACTOR_POLICY = "sha256(seed|distractor|subject|zero_based_source_row_index) modulo 3 over ascending wrong source-choice indices"
SAMPLING_POLICY = "ascending sha256(seed|sample|subject|zero_based_source_row_index), first 20 eligible unique questions per subject"
LABEL_POLICY = "ascending sha256(seed|label|subject|zero_based_source_row_index); first half truth A, second half truth B within each subject"

# Removing an option can invalidate these references even if the retained two
# strings differ.  Reject the whole four-choice item before any sampling.
CONTEXT_PATTERNS = (
    re.compile(r"\b(?:all|none|both|neither|any|each|either)\s+(?:of\s+)?(?:the\s+)?(?:above|below|these|those|foregoing|preceding|following)\b", re.I),
    re.compile(r"\b(?:all|none|both|neither)\s+(?:of\s+)?(?:the\s+)?(?:other\s+)?(?:answers|options|choices|responses)\b", re.I),
    re.compile(r"\b(?:answers?|options?|choices?|responses?)\s+[ABCD]\b"),
    re.compile(r"\b[ABCD]\s*(?:and|or|&)\s*[ABCD]\b"),
    re.compile(r"(?:^|\s)\([ABCD]\)\s*(?:and|or|,|&)"),
)


def canonical(value: str) -> str:
    return " ".join(value.split()).casefold()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_record_bytes(record: dict[str, Any]) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def rank(seed: int, purpose: str, subject: str, row_index: int) -> str:
    return sha256_bytes(f"{seed}|{purpose}|{subject}|{row_index}".encode("utf-8"))


def exclusion_reason(row: dict[str, Any]) -> str | None:
    """Return a conservative, deterministic pre-sampling exclusion reason."""
    if not isinstance(row, dict):
        return "malformed_row"
    question, choices, answer = row.get("question"), row.get("choices"), row.get("answer")
    if not isinstance(question, str) or not question.strip():
        return "empty_question"
    if not isinstance(choices, (list, tuple)) or len(choices) != 4:
        return "not_four_choices"
    if any(not isinstance(text, str) or not text.strip() for text in choices):
        return "empty_choice"
    if isinstance(answer, bool) or not isinstance(answer, int) or answer not in range(4):
        return "invalid_answer_index"
    if len({canonical(text) for text in choices}) != 4:
        return "duplicate_choice_text"
    if any(pattern.search(text) for pattern in CONTEXT_PATTERNS for text in [question, *choices]):
        return "choice_reference_or_collective_option"
    # Diagram-dependent questions cannot be reconstructed from text-only source.
    if re.search(r"\b(?:shown|illustrated|depicted)\s+(?:above|below)|\b(?:following|accompanying)\s+(?:figure|diagram|graph|image)\b", question, re.I):
        return "potential_missing_figure"
    return None


def transform_subject(
    rows: list[dict[str, Any]], subject: str, domain: str, curriculum_level: str,
    *, source_file: str, source_file_sha256: str, seed: int = SEED,
    count: int = ITEMS_PER_SUBJECT, seen_questions: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter -> sample -> choose distractor -> balance labels, without model use."""
    if count <= 0 or count % 2:
        raise ValueError("count must be a positive even number")
    seen_questions = set() if seen_questions is None else seen_questions
    excluded: Counter[str] = Counter()
    eligible: list[tuple[int, dict[str, Any]]] = []
    local_seen: set[str] = set()
    for index, row in enumerate(rows):
        reason = exclusion_reason(row)
        if reason:
            excluded[reason] += 1
            continue
        wording = canonical(row["question"])
        if wording in seen_questions or wording in local_seen:
            excluded["duplicate_question_text"] += 1
            continue
        local_seen.add(wording)
        eligible.append((index, row))
    selected = sorted(eligible, key=lambda entry: rank(seed, "sample", subject, entry[0]))[:count]
    if len(selected) != count:
        raise ValueError(f"{subject}: only {len(eligible)} eligible rows for requested {count}")
    truth_a_indices = {
        index for index, _ in sorted(selected, key=lambda entry: rank(seed, "label", subject, entry[0]))[:count // 2]
    }
    result = []
    for index, row in selected:
        answer = row["answer"]
        wrong_indices = [value for value in range(4) if value != answer]
        distractor = wrong_indices[int(rank(seed, "distractor", subject, index), 16) % 3]
        truth = "A" if index in truth_a_indices else "B"
        wrong = "B" if truth == "A" else "A"
        source_record = {"question": row["question"], "choices": list(row["choices"]), "answer": answer, "subject": subject}
        result.append({
            "question_id": f"mmlu_{subject}_test_{index:04d}",
            "question": row["question"],
            "options": {"A": row["choices"][answer if truth == "A" else distractor], "B": row["choices"][answer if truth == "B" else distractor]},
            "ground_truth": truth,
            "subject": subject,
            "domain": domain,
            "curriculum_level": curriculum_level,
            "item_type": "objective_multiple_choice_binary_projection",
            "source": {
                "dataset": DATASET_ID, "revision": DATASET_REVISION,
                "split": "test", "subject": subject, "row_index": index,
                "question": row["question"], "choices": list(row["choices"]),
                "answer_index": answer, "source_file": source_file,
                "source_url": f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{source_file}",
                "source_file_sha256": source_file_sha256,
                "source_record_sha256": sha256_bytes(canonical_record_bytes(source_record)),
            },
            "transformation": {
                "sampling_seed": seed, "distractor_index": distractor,
                "distractor_policy": DISTRACTOR_POLICY,
                "correct_label_policy": LABEL_POLICY,
                "retained_source_indices": {truth: answer, wrong: distractor},
            },
        })
        seen_questions.add(canonical(row["question"]))
    result.sort(key=lambda record: record["source"]["row_index"])
    return result, {
        "subject": subject, "domain": domain, "curriculum_level": curriculum_level,
        "source_rows": len(rows), "eligible_rows": len(eligible),
        "excluded_reasons": dict(sorted(excluded.items())),
        "selected_rows": len(result), "truth_counts": dict(Counter(record["ground_truth"] for record in result)),
        "source_file": source_file, "source_file_sha256": source_file_sha256,
        "selected_source_row_indices": [record["source"]["row_index"] for record in result],
    }


def read_source(path: Path, url: str, *, download: bool) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"Missing {path}; use --download once or supply an existing --cache-dir")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=90) as response:
            payload = response.read()
        path.write_bytes(payload)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Rebuilding needs pyarrow==23.0.1; inference uses committed JSON and does not need pyarrow") from exc
    payload = path.read_bytes()
    return pq.read_table(path).to_pylist(), sha256_bytes(payload)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=root / "tmp" / "mmlu_source" / DATASET_REVISION)
    parser.add_argument("--output", type=Path, default=root / "data" / "expanded_questions.json")
    parser.add_argument("--manifest", type=Path, default=root / "data" / "expanded_dataset_manifest.json")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    records, summaries = [], []
    seen: set[str] = set()
    for subject, domain, curriculum_level in SUBJECTS:
        relative = f"{subject}/test-00000-of-00001.parquet"
        url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{relative}"
        rows, digest = read_source(args.cache_dir / relative, url, download=args.download)
        chosen, summary = transform_subject(rows, subject, domain, curriculum_level, source_file=relative, source_file_sha256=digest, seen_questions=seen)
        records.extend(chosen)
        summaries.append(summary)
        print(f"{subject}: selected {len(chosen)} / eligible {summary['eligible_rows']} / source {len(rows)}", flush=True)
    payload = (json.dumps(records, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    manifest = {
        "schema_version": 1, "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION, "source_split": "test",
        "source_card_url": f"https://huggingface.co/datasets/{DATASET_ID}/blob/{DATASET_REVISION}/README.md",
        "source_paper_url": "https://arxiv.org/abs/2009.03300",
        "source_repository_url": "https://github.com/hendrycks/test",
        "source_citation": "Hendrycks, Burns, Basart, Zou, Mazeika, Song, and Steinhardt. Measuring Massive Multitask Language Understanding. ICLR 2021.",
        "license": "MIT", "license_url": "https://github.com/hendrycks/test/blob/master/LICENSE", "license_text": LICENSE_TEXT,
        "dataset_sha256": sha256_bytes(payload), "source_question_count": len(records),
        "subject_count": len(SUBJECTS), "items_per_subject": ITEMS_PER_SUBJECT,
        "seed": SEED, "sampling_policy": SAMPLING_POLICY,
        "distractor_policy": DISTRACTOR_POLICY, "correct_label_policy": LABEL_POLICY,
        "eligibility_policy": "Reject malformed rows, duplicate normalized options, collective/reference-dependent options or question text, potential missing figures, and duplicate normalized question wording before sampling. No model outputs, confidence scores, or answers from the evaluated model are used in selection.",
        "excluded_pattern_strings": [pattern.pattern for pattern in CONTEXT_PATTERNS],
        "transformation_disclosure": "Each original four-choice MMLU question retains the source-keyed correct answer and one deterministic wrong answer. The other two answers are removed. This is a binary projection, not the original MMLU benchmark; its chance level is 50% and scores must not be compared with published four-choice MMLU scores.",
        "taxonomy_disclosure": "Domain and curriculum_level are researcher-assigned from subject names, not empirical item difficulty. item_type describes format; no validated per-item reasoning taxonomy is claimed.",
        "sampling_limitations": "20 subjects are selected for breadth, not a representative sample of all 57 MMLU subjects. Existing public benchmark exposure, source-key errors, unresolved semantic ambiguities, and residual transformation defects remain possible. Eligibility uses conservative text rules, not complete human adjudication.",
        "planned_conditions": ["neutral", "false_opinion", "correct_opinion"],
        "planned_option_orders": ["original", "swapped"],
        "planned_prompt_rows": len(records) * 2 * 3,
        "domain_counts": dict(Counter(record["domain"] for record in records)),
        "curriculum_level_counts": dict(Counter(record["curriculum_level"] for record in records)),
        "truth_counts": dict(Counter(record["ground_truth"] for record in records)),
        "subjects": summaries,
        "question_ids": [record["question_id"] for record in records],
        "builder_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} source questions; SHA256={manifest['dataset_sha256']}", flush=True)


if __name__ == "__main__":
    main()
