import json
from pathlib import Path
import tempfile

from .model import validate_network_depth, validate_network_width


def atomic_actor_json(filename, value):
    if (value.get("schemaVersion") != 1 or value.get("policyContractVersion") != 1 or
            value.get("algorithm") not in ("ppo-beta-reference-v1", "td3-beta-mean-reference-v1") or
            value.get("observationSize") != 34 or value.get("actionSize") != 4):
        raise ValueError("Motion actor contract required for actor publication")
    validate_network_width(value.get("width"))
    depth = validate_network_depth(value.get("hiddenLayers", 2))
    maximum_bytes = (24 if depth == 10 else 8) * 1024 * 1024
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=filename.parent,
                                         prefix=filename.name + ".", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            encoded_bytes = 1
            for chunk in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(value):
                encoded_bytes += len(chunk.encode("utf-8"))
                if encoded_bytes > maximum_bytes:
                    raise ValueError("Learning artifact exceeds bounded reader size")
                output.write(chunk)
            output.write("\n")
        temporary.replace(filename)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
