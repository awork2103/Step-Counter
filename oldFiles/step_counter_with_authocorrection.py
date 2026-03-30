import numpy as np


class StepCounter:
    """
    One step counter class for both offline and real-time usage.
    You can add any other attributes you need to the class. But you should not change the interface of the class.
    """

    def __init__(self):
        """
        Initialize the step counter.
        """
        # Short moving average for noise reduction on acceleration magnitude.
        self.smooth_window_sec = 0.10
        # Slow moving average to remove orientation / carrying offset changes.
        self.baseline_window_sec = 0.85

        # History used for adaptive thresholding on the detrended signal.
        self.threshold_history_sec = 1.30
        self.threshold_percentile = 64.0
        self.min_threshold = 0.04

        # Local valley and prominence check.
        self.prominence_window_sec = 0.24
        self.min_prominence = 0.075

        # Minimum spacing between steps.
        self.min_step_interval_sec = 0.32

        # Autocorrelation-based cadence correction.
        self.acf_min_period_sec = 0.32
        self.acf_max_period_sec = 0.95
        self.acf_window_sec = 3.0
        self.acf_min_strength = 0.05
        self.acf_reject_ratio = 0.58

        self._time_eps = 1e-9
        self.reset()

    def reset(self) -> None:
        """
        Reset internal state such as buffers and cumulative count.
        After reset(), total_steps should be 0.
        """
        self.total_steps = 0

        self._all_times = []
        self._all_smoothed = []
        self._all_baselines = []
        self._all_detrended = []
        self._all_thresholds = []
        self._committed_step_timestamps = []

        self._raw_time_window = []
        self._raw_mag_window = []

        self._baseline_time_window = []
        self._baseline_value_window = []

        self._detect_time_window = []
        self._detect_value_window = []

        self._prev2 = None
        self._prev1 = None
        self._last_step_time = -np.inf
        self._last_input_time = -np.inf

    def update(self, data_chunk: dict) -> dict:
        """
        Real-time update: process a chunk of new samples.

        Input
          data_chunk["time"] : numpy.ndarray with shape (M,) [required]
          data_chunk["acc"]  : numpy.ndarray with shape (M, 3) in m/s^2 [required]
          data_chunk["gyro"] : numpy.ndarray with shape (M, 3) in rad/s [optional]
          data_chunk["mag"]  : numpy.ndarray with shape (M, 3) in uT [optional]
          Chunks arrive sequentially.

        Output (must contain all keys)
          {
            "new_steps": int,
            "total_steps": int,
            "new_step_timestamps": np.ndarray,  # shape (K,), float seconds
            "diagnostics": dict
          }
        """
        time_arr, acc_arr = self._validate_required_inputs(data_chunk)
        if time_arr.size == 0:
            return {
                "new_steps": 0,
                "total_steps": int(self.total_steps),
                "new_step_timestamps": np.empty(0, dtype=float),
                "diagnostics": self._build_diagnostics(),
            }

        new_steps = self._process_stream(time_arr, acc_arr)
        return {
            "new_steps": int(new_steps.size),
            "total_steps": int(self.total_steps),
            "new_step_timestamps": new_steps,
            "diagnostics": self._build_diagnostics(),
        }

    def run_offline(self, data: dict) -> dict:
        """
        Offline processing: process a full recording.

        Input
          data["time"] : numpy.ndarray with shape (N,) [required]
          data["acc"]  : numpy.ndarray with shape (N, 3) in m/s^2 [required]
          data["gyro"] : numpy.ndarray with shape (N, 3) in rad/s [optional]
          data["mag"]  : numpy.ndarray with shape (N, 3) in uT [optional]

        Output format for grading (must contain all keys)
          {
            "step_count": int,
            "step_timestamps": np.ndarray,  # shape (K,), float seconds
            "diagnostics": dict
          }

        Requirements on output:
          - "step_count" must be a Python int and must be >= 0.
          - "step_timestamps" must be a 1D NumPy array of dtype float with shape (K,).
            Each entry is a timestamp in seconds. If your algorithm does not produce
            timestamps, return an empty array with shape (0,) rather than None.
          - "diagnostics" must be a Python dict. It may be empty.
        """
        time_arr, acc_arr = self._validate_required_inputs(data)
        temp_counter = StepCounter()
        step_timestamps = temp_counter._process_stream(time_arr, acc_arr, finalize=True)
        return {
            "step_count": int(step_timestamps.size),
            "step_timestamps": step_timestamps,
            "diagnostics": temp_counter._build_diagnostics(),
        }

    def _validate_required_inputs(self, data: dict):
        if "time" not in data or "acc" not in data:
            raise KeyError("Input dictionary must contain 'time' and 'acc'.")

        time_arr = np.asarray(data["time"], dtype=float)
        acc_arr = np.asarray(data["acc"], dtype=float)

        if time_arr.ndim != 1:
            raise ValueError("data['time'] must be a 1D array.")
        if acc_arr.ndim != 2 or acc_arr.shape[1] != 3:
            raise ValueError("data['acc'] must have shape (N, 3).")
        if acc_arr.shape[0] != time_arr.shape[0]:
            raise ValueError("data['time'] and data['acc'] must contain the same number of samples.")
        if time_arr.size == 0:
            return np.empty(0, dtype=float), np.empty((0, 3), dtype=float)

        finite_mask = np.isfinite(time_arr) & np.all(np.isfinite(acc_arr), axis=1)
        time_arr = time_arr[finite_mask]
        acc_arr = acc_arr[finite_mask]
        if time_arr.size == 0:
            return np.empty(0, dtype=float), np.empty((0, 3), dtype=float)

        order = np.argsort(time_arr, kind="mergesort")
        time_arr = time_arr[order]
        acc_arr = acc_arr[order]

        unique_mask = np.ones(time_arr.shape[0], dtype=bool)
        unique_mask[1:] = np.diff(time_arr) > self._time_eps
        return time_arr[unique_mask], acc_arr[unique_mask]

    def _process_stream(self, time_arr, acc_arr, finalize=False):
        new_step_timestamps = []

        for t, acc in zip(time_arr, acc_arr):
            t = float(t)
            if t <= self._last_input_time + self._time_eps:
                continue
            self._last_input_time = t

            # Orientation-robust magnitude from tri-axis acceleration.
            mag = float(np.sqrt(np.sum(np.asarray(acc, dtype=float) ** 2)))
            smoothed, baseline, detrended, threshold, valley = self._update_signal_windows(t, mag)

            self._all_times.append(t)
            self._all_smoothed.append(smoothed)
            self._all_baselines.append(baseline)
            self._all_detrended.append(detrended)
            self._all_thresholds.append(threshold)

            current = {
                "time": t,
                "value": detrended,
                "threshold": threshold,
                "valley": valley,
                "strength": max(0.0, detrended - valley),
            }

            if self._prev2 is not None and self._prev1 is not None:
                if self._is_step_candidate(self._prev2, self._prev1, current):
                    step_time = float(self._prev1["time"])
                    if self._passes_autocorrect(step_time):
                        self._committed_step_timestamps.append(step_time)
                        self._last_step_time = step_time
                        new_step_timestamps.append(step_time)

            self._prev2 = self._prev1
            self._prev1 = current

        if finalize and self._prev2 is not None and self._prev1 is not None:
            tail_probe = {
                "time": self._prev1["time"] + self._time_eps,
                "value": -np.inf,
                "threshold": self._prev1["threshold"],
                "valley": self._prev1["valley"],
                "strength": self._prev1["strength"],
            }
            if self._is_step_candidate(self._prev2, self._prev1, tail_probe):
                step_time = float(self._prev1["time"])
                if self._passes_autocorrect(step_time):
                    self._committed_step_timestamps.append(step_time)
                    self._last_step_time = step_time
                    new_step_timestamps.append(step_time)

        self.total_steps = int(len(self._committed_step_timestamps))
        return np.asarray(new_step_timestamps, dtype=float)

    def _update_signal_windows(self, current_time, current_mag):
        self._raw_time_window.append(current_time)
        self._raw_mag_window.append(current_mag)
        self._drop_old(
            self._raw_time_window,
            self._raw_mag_window,
            current_time - self.smooth_window_sec,
        )
        smoothed = float(np.mean(self._raw_mag_window))

        self._baseline_time_window.append(current_time)
        self._baseline_value_window.append(smoothed)
        self._drop_old(
            self._baseline_time_window,
            self._baseline_value_window,
            current_time - self.baseline_window_sec,
        )
        baseline = float(np.mean(self._baseline_value_window))
        detrended = float(smoothed - baseline)

        self._detect_time_window.append(current_time)
        self._detect_value_window.append(detrended)
        self._drop_old(
            self._detect_time_window,
            self._detect_value_window,
            current_time - max(self.threshold_history_sec, self.acf_window_sec),
        )

        recent_vals = [
            v for tt, v in zip(self._detect_time_window, self._detect_value_window)
            if tt >= current_time - self.threshold_history_sec - self._time_eps
        ]
        recent_vals = np.asarray(recent_vals, dtype=float)
        if recent_vals.size == 0:
            threshold = self.min_threshold
        else:
            threshold = float(np.percentile(recent_vals, self.threshold_percentile))
            threshold = max(self.min_threshold, threshold)

        valley_start = current_time - self.prominence_window_sec
        valley_values = [
            v for tt, v in zip(self._detect_time_window, self._detect_value_window)
            if tt >= valley_start - self._time_eps
        ]
        valley = float(np.min(valley_values)) if valley_values else detrended
        return smoothed, baseline, detrended, threshold, valley

    def _drop_old(self, time_window, value_window, min_time_to_keep):
        while time_window and time_window[0] < min_time_to_keep - self._time_eps:
            time_window.pop(0)
            value_window.pop(0)

    def _is_step_candidate(self, prev2, prev1, current):
        is_peak = prev1["value"] >= prev2["value"] and prev1["value"] > current["value"]
        high_enough = prev1["value"] > prev1["threshold"]
        prominent = (prev1["value"] - prev1["valley"]) >= self.min_prominence
        return bool(is_peak and high_enough and prominent)

    def _passes_autocorrect(self, step_time):
        gap = float(step_time - self._last_step_time)
        if gap < self.min_step_interval_sec:
            return False

        expected_interval = self._estimate_expected_interval()
        if expected_interval is None:
            return True

        dynamic_floor = max(self.min_step_interval_sec, self.acf_reject_ratio * expected_interval)
        return bool(gap >= dynamic_floor)

    def _estimate_expected_interval(self):
        if len(self._detect_time_window) < 12:
            return None

        time_arr = np.asarray(self._detect_time_window, dtype=float)
        value_arr = np.asarray(self._detect_value_window, dtype=float)
        keep_mask = time_arr >= (time_arr[-1] - self.acf_window_sec - self._time_eps)
        time_arr = time_arr[keep_mask]
        value_arr = value_arr[keep_mask]

        if time_arr.size < 12:
            return None

        dt = float(np.median(np.diff(time_arr)))
        if (not np.isfinite(dt)) or dt <= 0.0:
            return None

        x = value_arr - np.mean(value_arr)
        power = float(np.dot(x, x))
        if power <= 1e-10:
            return None

        acf = np.correlate(x, x, mode="full")
        acf = acf[acf.size // 2 :]
        acf = acf / acf[0]

        min_lag = max(1, int(round(self.acf_min_period_sec / dt)))
        max_lag = min(len(acf) - 1, int(round(self.acf_max_period_sec / dt)))
        if max_lag <= min_lag:
            return None

        search = acf[min_lag : max_lag + 1]
        lag = int(min_lag + np.argmax(search))
        if acf[lag] < self.acf_min_strength:
            return None
        return float(lag * dt)

    def _build_diagnostics(self):
        return {
            "smoothed_magnitude": np.asarray(self._all_smoothed, dtype=float),
            "baseline": np.asarray(self._all_baselines, dtype=float),
            "detrended_signal": np.asarray(self._all_detrended, dtype=float),
            "threshold": np.asarray(self._all_thresholds, dtype=float),
            "num_detected_steps": int(len(self._committed_step_timestamps)),
        }
