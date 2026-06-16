# semanticsubgraph/preprocess/dedupe.py

from typing import List, Tuple
from pydantic import ValidationError
from .models import RawRecord, PreprocessActions

# Shared utils
from .utils import (
    rawrecordlist_to_id_text_map,
    rawrecords_grouped,
    remove_by_raw_id,
    flatten_list_of_lists,
    cluster_strings_adaptive,
)

def validate_input(records: List[RawRecord]) -> List[RawRecord]:
    if not isinstance(records, list):
        raise TypeError("dedupe: input must be a list of RawRecord")

    for i, rec in enumerate(records):
        if not isinstance(rec, RawRecord):
            raise TypeError(f"dedupe: item at index {i} is not a RawRecord")

    return records


def calculate(
    records: List[RawRecord],
    metadata_fields: List[str] | None = None,
    global_threshold: float = 0.75,
    metadata_threshold: float = 0.70,
) -> Tuple[List[RawRecord], PreprocessActions]:
    """
    Core dedupe logic:
      - Always run global clustering
      - Only run metadata-level clustering if metadata_fields is provided
    """

    # -------------------------
    # Stage 1: Global clustering
    # -------------------------
    raw_text_dict = rawrecordlist_to_id_text_map(records)

    global_clusters = cluster_strings_adaptive(
        raw_text_dict,
        threshold=global_threshold,
    )

    to_deactivate = flatten_list_of_lists(
        [c[-1:] for c in global_clusters if len(c) > 1]
    )

    filtered_records = remove_by_raw_id(
        records,
        [str(i) for i in to_deactivate],
    )

    # -------------------------
    # Stage 2: Metadata-level clustering (ONLY if provided)
    # -------------------------
    if metadata_fields:
        for field in metadata_fields:

            grouped = rawrecords_grouped(
                filtered_records,
                metadata_name=field,
            )

            for _, id_text_map in grouped.items():

                clusters = cluster_strings_adaptive(
                    id_text_map,
                    threshold=metadata_threshold,
                )

                group_deactivate = flatten_list_of_lists(
                    [c[-1:] for c in clusters if len(c) > 1]
                )

                to_deactivate.extend(group_deactivate)

            filtered_records = remove_by_raw_id(
                filtered_records,
                [str(i) for i in to_deactivate],
            )

    # -------------------------
    # Final filtered list
    # -------------------------
    final_filtered = remove_by_raw_id(
        filtered_records,
        [str(i) for i in to_deactivate],
    )

    # -------------------------
    # Build PreprocessActions
    # -------------------------
    actions = PreprocessActions(
        deactivate_raw_ids=[str(i) for i in sorted(set(to_deactivate))],
        notes=[
            f"dedupe: removed {len(set(to_deactivate))} duplicates",
            f"dedupe: global threshold={global_threshold}, metadata threshold={metadata_threshold}",
            f"dedupe: metadata fields={metadata_fields or 'NONE (global only)'}",
        ],
    )

    return final_filtered, actions


def validate_output(records: List[RawRecord]) -> List[RawRecord]:
    if not isinstance(records, list):
        raise TypeError("dedupe: output must be a list of RawRecord")

    for i, rec in enumerate(records):
        if not isinstance(rec, RawRecord):
            raise TypeError(f"dedupe: output item at index {i} is not a RawRecord")

    return records


def run(
    records: List[RawRecord],
    metadata_fields: List[str] | None = None,
    global_threshold: float = 0.75,
    metadata_threshold: float = 0.60,
) -> Tuple[List[RawRecord], PreprocessActions]:
    """
    Standardized entry point for all preprocessing modules.
    """
    validated_in = validate_input(records)
    calculated_records, actions = calculate(
        validated_in,
        metadata_fields=metadata_fields,
        global_threshold=global_threshold,
        metadata_threshold=metadata_threshold,
    )
    validated_out = validate_output(calculated_records)
    return validated_out, actions
