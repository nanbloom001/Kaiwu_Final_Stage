#!/usr/bin/env python3
"""Pure save-trigger helpers shared by the workflow and local tests."""

SAVE_DEDUP_DELTA_S = 60.0


def should_deduplicate_save(
    delta_since_save: float,
    *,
    first_save_this_session: bool,
) -> bool:
    """Return whether a save trigger is covered by the prior save."""
    return (
        not first_save_this_session
        and float(delta_since_save) <= SAVE_DEDUP_DELTA_S
    )
