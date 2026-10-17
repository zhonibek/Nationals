from .corpus_warm import anchor_rows, validate_definition
from .expert_data import iter_demonstrations


SOURCE_TRANSFORM = "deadline-context-v1"
TARGET_TRANSFORM = "long-history-deadline-v1"
SOURCE_SIZE = 145
TARGET_SIZE = 565
FRAME_SIZE = 35


def history_definition(frames, source_definition=None):
    if type(frames) is not int or frames not in (4, 16):
        raise ValueError("Explicit original four or sixteen public history frames required")
    if frames == 4:
        return None
    validate_definition(source_definition)
    if source_definition["featureTransform"] != SOURCE_TRANSFORM or source_definition["featureSize"] != SOURCE_SIZE:
        raise ValueError("Actual original four-frame public corpus required")
    return {"method": "original-public-long-history-view-v1", "sourceManifestSha256": source_definition["manifestSha256"],
        "sourceFeatureTransform": SOURCE_TRANSFORM, "targetFeatureTransform": TARGET_TRANSFORM,
        "sourceFeatureSize": SOURCE_SIZE, "targetFeatureSize": TARGET_SIZE,
        "sourceHistoryFrames": 4, "targetHistoryFrames": 16, "frameSize": FRAME_SIZE,
        "rows": source_definition["rowCount"], "padding": "Repeat only this world's first actually measured public frame",
        "admission": "Every original four-frame window must exactly reconstruct from its own chronological current frames and constant initial context",
        "labels": "Every original actually applied teacher action unchanged, chronological source-world boundaries retained",
        "accounting": "No new native query or invented label from deriving a view; original measurement cost and every source world retained separately",
        "scope": "Explicit recorded-public-history view, not producer metadata rewriting, private input, altered physics, reward or movement qualification"}


def extend_records(records):
    expanded = []
    for record in records:
        frames = []
        context = None
        rows = []
        for row in record["rows"]:
            values = row["modelInput"]
            if len(values) != SOURCE_SIZE:
                raise ValueError("Original145-feature chronological row required")
            if context is None:
                context = list(values[140:144])
            if values[140:144] != context:
                raise ValueError("Each recorded world must retain its exact initial public context")
            frames.append(list(values[105:140]))
            short = frames[-4:]
            expected = [item for frame in [frames[0]] * (4 - len(short)) + short for item in frame] + context + [values[-1]]
            if expected != values:
                raise ValueError("Every source history window must match the actual ordered public frame sequence")
            history = frames[-16:]
            inputs = [item for frame in [frames[0]] * (16 - len(history)) + history for item in frame] + context + [values[-1]]
            rows.append({"modelInput": inputs, "teacherAction": list(row["teacherAction"])})
        expanded.append({**record, "rows": rows})
    return expanded


def extend_corpus(directory, corpus):
    definition = history_definition(16, corpus["definition"])
    if corpus.get("portability") is not None:
        raise ValueError("Long-history derivation currently requires the exact original producing runtime")
    records = list(iter_demonstrations(directory))
    if [row for record in records for row in record["rows"]] != corpus["rows"]:
        raise ValueError("Every exact original admitted source row required before deriving a view")
    expanded = extend_records(records)
    rows = [row for record in expanded for row in record["rows"]]
    if len(rows) != corpus["definition"]["rowCount"]:
        raise ValueError("No original source row may be omitted or duplicated")
    return {**corpus, "rows": rows, "anchorRows": anchor_rows(expanded), "historyView": definition}
