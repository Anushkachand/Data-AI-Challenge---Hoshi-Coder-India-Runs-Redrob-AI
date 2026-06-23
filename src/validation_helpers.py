"""
validation_helpers.py
Helper functions to validate the generated candidate ranking output.
Follows PEP 8 style guide.
"""

import csv
import re
from pathlib import Path


def verify_ranking_output(csv_path):
    """
    Validates a submission CSV file against Redrob hackathon rules.
    Returns:
        is_valid (bool): True if submission is valid, False otherwise.
        errors (list): List of error messages.
    """
    errors = []
    path = Path(csv_path)

    if not path.exists():
        return False, [f"File {csv_path} does not exist."]

    required_header = ["candidate_id", "rank", "score", "reasoning"]
    candidate_id_pattern = re.compile(r"^CAND_[0-9]{7}$")
    expected_rows = 100

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return False, ["File is empty."]

            if header != required_header:
                errors.append(
                    f"Header row must be exactly: {','.join(required_header)}. "
                    f"Found: {','.join(header)}"
                )

            data_rows = []
            for row in reader:
                if any(cell.strip() for cell in row):
                    data_rows.append(row)
    except Exception as e:
        return False, [f"Failed to read file: {e}"]

    n = len(data_rows)
    if n != expected_rows:
        errors.append(f"Expected exactly {expected_rows} data rows, found {n}.")

    seen_ids = set()
    seen_ranks = set()
    by_rank = []

    for i, cells in enumerate(data_rows):
        row_num = 2 + i
        if len(cells) != len(required_header):
            errors.append(
                f"Row {row_num}: expected {len(required_header)} columns, got {len(cells)}."
            )
            continue

        row = dict(zip(required_header, cells))
        cid = row["candidate_id"].strip()
        rank_s = row["rank"].strip()
        score_s = row["score"].strip()
        reasoning = row["reasoning"].strip()

        # Validate candidate ID
        if not cid:
            errors.append(f"Row {row_num}: candidate_id is required.")
        elif not candidate_id_pattern.match(cid):
            errors.append(f"Row {row_num}: candidate_id format must be CAND_XXXXXXX.")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'.")
        else:
            seen_ids.add(cid)

        # Validate rank
        try:
            rank = int(rank_s)
            if str(rank) != rank_s:
                raise ValueError
            if not 1 <= rank <= 100:
                errors.append(f"Row {row_num}: rank must be 1 to 100.")
            elif rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}.")
            else:
                seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {row_num}: rank must be an integer (1-100).")
            rank = None

        # Validate score
        try:
            score = float(score_s)
        except ValueError:
            errors.append(f"Row {row_num}: score must be a float.")
            score = None

        # Validate reasoning
        if not reasoning:
            errors.append(f"Row {row_num}: reasoning cannot be empty.")
        else:
            # Check sentence counts (approximate via periods not between digits)
            sentences = [s for s in re.split(r'(?<!\d)\.(?!\d)', reasoning) if s.strip()]
            if len(sentences) > 2:
                errors.append(
                    f"Row {row_num}: reasoning must be 1-2 sentences. Found {len(sentences)}."
                )

        if rank is not None and score is not None and cid:
            by_rank.append((rank, score, cid))

    # Verify all ranks present
    missing_ranks = set(range(1, 101)) - seen_ranks
    if missing_ranks:
        errors.append(f"Each rank 1-100 must appear once. Missing: {sorted(missing_ranks)}")

    # Sort by rank and verify monotonic score
    by_rank.sort(key=lambda x: x[0])
    for i in range(len(by_rank) - 1):
        r1, s1, _ = by_rank[i]
        r2, s2, _ = by_rank[i + 1]
        if s1 < s2:
            errors.append(
                f"Scores must be non-increasing by rank: rank {r1} ({s1}) < rank {r2} ({s2})."
            )

    # Tie break check (ID ascending)
    for i in range(len(by_rank) - 1):
        r1, s1, c1 = by_rank[i]
        r2, s2, c2 = by_rank[i + 1]
        if s1 == s2 and c1 > c2:
            errors.append(
                f"Tie-break violation at ranks {r1} and {r2}: candidate_id ascending order "
                f"({c1} should come before {c2})."
            )

    return len(errors) == 0, errors
