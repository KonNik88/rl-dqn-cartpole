from __future__ import annotations


class LinearSchedule:
    """
    Универсальная линейная шкала.
    """
    def __init__(self, start: float, end: float, steps: int):
        self.start = float(start)
        self.end = float(end)
        self.steps = max(1, int(steps))
        self.t = 0

    def step(self, n: int = 1) -> None:
        self.t += int(n)

    def value(self, t: int | None = None) -> float:
        if t is None:
            t = self.t
        frac = min(1.0, max(0.0, float(t) / float(self.steps)))
        return self.start + (self.end - self.start) * frac

LinearScheduler = LinearSchedule