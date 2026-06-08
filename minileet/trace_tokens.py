from __future__ import annotations

from minileet.dsl import TraceEvent


def encode_event(event: TraceEvent) -> list[str]:
    tokens = [f"EVENT:{event.kind}"]
    for key in sorted(event.data):
        tokens.append(f"{key}={repr(event.data[key])}")
    return tokens


def encode_trace(trace: tuple[TraceEvent, ...]) -> list[str]:
    tokens: list[str] = []
    for event in trace:
        encoded = encode_event(event)
        tokens.append(f"BEGIN:{len(encoded)}")
        tokens.extend(encoded)
    return tokens


def decode_trace(tokens: list[str]) -> tuple[TraceEvent, ...]:
    events: list[TraceEvent] = []
    i = 0
    while i < len(tokens):
        marker = tokens[i]
        if not marker.startswith("BEGIN:"):
            raise ValueError(f"bad trace marker {marker}")
        n = int(marker.split(":", 1)[1])
        parts = tokens[i + 1 : i + 1 + n]
        if len(parts) != n or not parts[0].startswith("EVENT:"):
            raise ValueError("bad trace event")
        kind = parts[0].split(":", 1)[1]
        data = {}
        for item in parts[1:]:
            key, value = item.split("=", 1)
            data[key] = eval(value, {"__builtins__": {}}, {})
        events.append(TraceEvent(kind, data))
        i += 1 + n
    return tuple(events)


def divergence_window(trace: tuple[TraceEvent, ...], index: int, radius: int = 2) -> tuple[TraceEvent, ...]:
    start = max(0, index - radius)
    end = min(len(trace), index + radius + 1)
    return trace[start:end]
