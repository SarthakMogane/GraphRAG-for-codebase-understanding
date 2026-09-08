from src.services.scout.deep_scout import RepoScoutResult as ScoutResult
import dataclasses

def _serialize_scout(result: ScoutResult) -> dict:
    """Convert RepoScoutResult dataclass tree to a JSON-safe dict."""
    def _convert(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if hasattr(obj, "value"):   # Enum → string
            return obj.value
        return obj
    return _convert(result)