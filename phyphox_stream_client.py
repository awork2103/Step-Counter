import numpy as np
import requests


class PhyphoxStreamClient:
    def __init__(self, base_url, time_buffer=None, acc_buffers=None):
        self.base_url = str(base_url).rstrip("/")
        self.time_buffer = time_buffer
        self.acc_buffers = acc_buffers
        self._last_time = -np.inf
        self._configure_buffers()

    def _configure_buffers(self):
        response = requests.get(self.base_url + "/config", timeout=5)
        response.raise_for_status()
        config = response.json()

        if self.time_buffer is not None and self.acc_buffers is not None:
            return

        buffers = []

        for item in config.get("buffers", []):
            name = item.get("name")
            if isinstance(name, str) and name not in buffers:
                buffers.append(name)

        for inp in config.get("inputs", []):
            for out in inp.get("outputs", []):
                for v in out.values():
                    if isinstance(v, str) and v not in buffers:
                        buffers.append(v)

        for ex in config.get("export", []):
            for src in ex.get("sources", []):
                name = src.get("buffer")
                if isinstance(name, str) and name not in buffers:
                    buffers.append(name)

        buffers_lower = {b.lower(): b for b in buffers}

        def find_first(candidates):
            for c in candidates:
                if c.lower() in buffers_lower:
                    return buffers_lower[c.lower()]
            return None

        if self.time_buffer is None:
            self.time_buffer = find_first([
                "time", "t", "acc_time", "linacc_time", "linearacc_time"
            ])

        if self.acc_buffers is None:
            ax = find_first(["ax", "accx", "acc_x", "linaccx", "x"])
            ay = find_first(["ay", "accy", "acc_y", "linaccy", "y"])
            az = find_first(["az", "accz", "acc_z", "linaccz", "z"])
            if ax and ay and az:
                self.acc_buffers = [ax, ay, az]

        if self.time_buffer is None or self.acc_buffers is None or len(self.acc_buffers) != 3:
            raise RuntimeError(
                "Could not auto-detect phyphox buffers from /config.\n"
                f"Available buffers: {buffers}\n"
                "Pass --time-buffer and --acc-buffers explicitly."
            )

        print(f"[INFO] Auto-detected buffers: time={self.time_buffer}, acc={self.acc_buffers}")

    def get_new_data(self):
        names = [self.time_buffer] + list(self.acc_buffers)
        query = "&".join([f"{name}=full" for name in names])

        response = requests.get(self.base_url + "/get?" + query, timeout=5)
        response.raise_for_status()
        payload = response.json()

        if "buffer" not in payload:
            return {
                "time": np.empty(0, dtype=float),
                "acc": np.empty((0, 3), dtype=float),
            }

        def extract_series(name):
            buf = payload["buffer"].get(name, {})
            if "buffer" in buf:
                values = buf["buffer"]
            elif "data" in buf:
                values = buf["data"]
            else:
                values = []
            return np.asarray(values, dtype=float)

        t = extract_series(self.time_buffer)
        ax = extract_series(self.acc_buffers[0])
        ay = extract_series(self.acc_buffers[1])
        az = extract_series(self.acc_buffers[2])

        n = min(t.size, ax.size, ay.size, az.size)
        if n <= 0:
            return {
                "time": np.empty(0, dtype=float),
                "acc": np.empty((0, 3), dtype=float),
            }

        t = t[:n]
        acc = np.column_stack((ax[:n], ay[:n], az[:n]))

        mask = t > self._last_time
        t = t[mask]
        acc = acc[mask]

        if t.size > 0:
            self._last_time = float(t[-1])

        return {
            "time": t,
            "acc": acc,
        }
