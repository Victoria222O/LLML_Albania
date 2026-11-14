import math
from collections import deque

# Baseline (mean, std) of each channel on normal data.
_STATS = {
    "temperature_c": (13.00, 0.372),
    "humidity_pct": (72.02, 0.58),
    "vibration_g": (0.1198, 0.0100),
    "acoustic_db": (32.098, 0.350),
    "air_pressure_kpa": (101.503, 0.099),
    "co2_ppm": (510.3, 10.5),
}
_KEYS = list(_STATS)
W = 6          # window for drift detection
DRIFT_T = 3.0  # threshold on windowed |mean z|
POINT_T = 7.0
COLL_T = 6.0
CLIP = 6.0


class Detector:
    def __init__(self):
        self.hist = {k: deque(maxlen=W) for k in _KEYS}
        self.coll = deque(maxlen=3)
        self.ew = {k: 0.0 for k in _KEYS}

    def update(self, reading: dict) -> float:
        z = {}
        for k in _KEYS:
            try:
                v = float(reading.get(k, _STATS[k][0]))
                if not math.isfinite(v):
                    v = _STATS[k][0]
            except (TypeError, ValueError):
                v = _STATS[k][0]
            z[k] = (v - _STATS[k][0]) / _STATS[k][1]
            self.hist[k].append(max(-CLIP, min(CLIP, z[k])))

        # point: one huge spike on a channel
        s_point = max(abs(x) for x in z.values()) / POINT_T

        # collective: vibration + acoustic + co2 elevated together
        c = sum(min(max(z[k], 0), 7.0) for k in ("vibration_g", "acoustic_db", "co2_ppm"))
        self.coll.append(c)
        cm = sum(self.coll) / len(self.coll)
        n_el = sum(1 for k in ("vibration_g", "acoustic_db", "co2_ppm") if z[k] > 1.5)
        s_coll = cm / (3 * COLL_T / 3 * 2.0) if n_el >= 2 else 0.0

        # drift: sustained windowed mean shift on any channel
        s_drift = 0.0
        for k in _KEYS:
            h = self.hist[k]
            if len(h) >= W:
                m = sum(h) / len(h)
                if m * z[k] <= 0 or abs(z[k]) < 1.0:
                    continue
                m = abs(m)
                s_drift = max(s_drift, m / DRIFT_T)

        s = max(s_point, s_coll, s_drift)
        return float(min(1.0, max(0.0, 0.75 * s)))