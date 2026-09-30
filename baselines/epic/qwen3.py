"""EPIC/LegoLink initial-token selection, matching the upstream e2e protocol."""
def selected_positions(plan, width=4):
    if width < 0:
        raise ValueError('EPIC initial-token budget must be nonnegative')
    artifact_positions = {i for s in plan.segments for i in range(s.start, s.end)}
    selected = set(range(len(plan.token_ids))) - artifact_positions
    for s in plan.segments:
        # The first module has no preceding context and needs no repair.
        if s.start != 0:
            selected.update(range(s.start, min(s.end, s.start + width)))
    return sorted(selected)
