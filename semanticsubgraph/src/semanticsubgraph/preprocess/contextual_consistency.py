# semanticsubgraph/preprocess/contextual_consistency.py

from typing import List, Tuple
from .models import RawRecord, PreprocessActions
from .utils import (
    extract_and_normalize_slug,
)
from .utils import (
    text_metadata_distances_below_threshold,
)


def validate_input(records: List[RawRecord]) -> List[RawRecord]:
    if not isinstance(records, list):
        raise TypeError("contextual_consistency: input must be a list of RawRecord")

    for i, rec in enumerate(records):
        if not isinstance(rec, RawRecord):
            raise TypeError(
                f"contextual_consistency: item at index {i} is not a RawRecord"
            )

    return records


def validate_output(records: List[RawRecord]) -> List[RawRecord]:
    if not isinstance(records, list):
        raise TypeError("contextual_consistency: output must be a list of RawRecord")

    for i, rec in enumerate(records):
        if not isinstance(rec, RawRecord):
            raise TypeError(
                f"contextual_consistency: output item at index {i} is not a RawRecord"
            )

    return records

def calculate(
    records: List[RawRecord],
    embeddings,
    similarity_threshold: float = 0.15,
    min_length: int = 40,
    metadata_key: str = "slug",
    debug: bool = False,
) -> Tuple[List[RawRecord], PreprocessActions]:
    """
    Contextual consistency logic (generalized):
      - Filter out summaries that are too short
      - Compute text–metadata embedding similarity for each record
      - Deactivate records whose similarity is below threshold
      - Optional debug mode prints offenders
    """

    # -------------------------
    # 1. Length-based filtering
    # -------------------------
    too_short_ids = [
        str(r.raw_id)
        for r in records
        if len(r.text.strip()) < min_length
    ]


    remaining = [
        r for r in records
        if str(r.raw_id) not in too_short_ids
    ]

    # -------------------------
    # 2. Compute low-similarity records
    # -------------------------
    bad_similarity = text_metadata_distances_below_threshold(
        remaining,
        embeddings=embeddings,
        metadata_key=metadata_key,
        threshold=similarity_threshold,
    )

    bad_similarity_ids = list(bad_similarity.keys())

    # -------------------------
    # Combine all bad IDs
    # -------------------------
    all_bad_ids = sorted(set(too_short_ids + bad_similarity_ids))

    # -------------------------
    # Debug mode: print offenders
    # -------------------------
    if debug:
        print("\n=== Contextual Consistency Debug ===")
        print(f"Total offenders: {len(all_bad_ids)}")
        print(f"Metadata key used: '{metadata_key}'\n")

        # Length offenders
        if too_short_ids:
            print("Too-short summaries:")
            for r in records:
                if str(r.raw_id) in too_short_ids:
                    summary = r.text.strip()
                    print(f"  - raw_id={r.raw_id} (len={len(summary)})")
            print()

        # Similarity offenders
        if bad_similarity_ids:
            print(f"Low text–{metadata_key} similarity:")
            for r in remaining:
                rid = str(r.raw_id)
                if rid in bad_similarity:
                    sim = bad_similarity[rid]
                    meta_val = r.metadata.get(metadata_key, "")
                    summary = r.text.strip()
                    short_summary = summary[:120] + ("..." if len(summary) > 120 else "")
                    print(f"  - raw_id={rid} sim={sim:.3f}")
                    print(f"      {metadata_key}: {meta_val}")
                    print(f"      summary: {short_summary}")
            print()

        print("=== End Debug ===\n")

    # -------------------------
    # Filter out inconsistent records
    # -------------------------
    filtered = [
        r for r in records
        if str(r.raw_id) not in all_bad_ids
    ]

    # -------------------------
    # Build PreprocessActions
    # -------------------------
    actions = PreprocessActions(
        deactivate_raw_ids=all_bad_ids,
        notes=[
            f"contextual_consistency: removed {len(all_bad_ids)} inconsistent summaries",
            f"contextual_consistency: similarity threshold={similarity_threshold}",
            f"contextual_consistency: min_length={min_length}",
            f"contextual_consistency: metadata_key={metadata_key}",
        ],
    )

    return filtered, actions

def run(
    records: List[RawRecord],
    embeddings:any,
    similarity_threshold: float = 0.15,
    min_length: int = 40,
    metadata_key: str = "slug",
    debug: bool = False,
) -> Tuple[List[RawRecord], PreprocessActions]:
    """
    Standardized entry point for all preprocessing modules.
    """
    validated_in = validate_input(records)
    calculated_records, actions = calculate(
        validated_in,
        embeddings,
        similarity_threshold=similarity_threshold,
        min_length=min_length,
        metadata_key=metadata_key,
        debug=debug,
    )
    validated_out = validate_output(calculated_records)
    return validated_out, actions
