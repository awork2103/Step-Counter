import matplotlib.pyplot as plt
import numpy as np


class RealtimeStepPlotter:
    def __init__(self, step_counter, window_seconds=12.0):
        self.step_counter = step_counter
        self.window_seconds = float(window_seconds)

    def run_phyphox(self, client, poll_interval=0.05):
        self.step_counter.reset()
        plt.ion()
        fig, ax = plt.subplots(figsize=(11, 6))
        try:
            while plt.fignum_exists(fig.number):
                chunk = client.get_new_data()
                result = self.step_counter.update(chunk)
                self._redraw(ax, result, title="Real-time Step Counter (phyphox)")
                plt.pause(max(0.001, poll_interval))
        finally:
            plt.ioff()
            if plt.fignum_exists(fig.number):
                plt.show()

    def run_replay(self, data, chunk_size=5, speedup=1.0):
        self.step_counter.reset()
        time_arr = np.asarray(data["time"], dtype=float)
        acc_arr = np.asarray(data["acc"], dtype=float)

        plt.ion()
        fig, ax = plt.subplots(figsize=(11, 6))
        try:
            i = 0
            while i < len(time_arr) and plt.fignum_exists(fig.number):
                j = min(i + int(chunk_size), len(time_arr))
                chunk = {
                    "time": time_arr[i:j],
                    "acc": acc_arr[i:j],
                }
                result = self.step_counter.update(chunk)
                self._redraw(ax, result, title="Replay Step Counter")

                if j < len(time_arr):
                    dt = max(0.001, float(time_arr[j - 1] - time_arr[i]) / max(speedup, 1e-6))
                    plt.pause(dt)
                else:
                    plt.pause(0.001)
                i = j
        finally:
            plt.ioff()
            if plt.fignum_exists(fig.number):
                plt.show()

    def _redraw(self, ax, result, title):
        ax.clear()

        t = np.asarray(self.step_counter._all_times, dtype=float)
        raw_like = np.asarray(self.step_counter._all_smoothed, dtype=float)
        thr = np.asarray(self.step_counter._all_thresholds, dtype=float)
        steps = np.asarray(self.step_counter._committed_step_timestamps, dtype=float)

        if t.size == 0:
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Acceleration magnitude")
            ax.text(0.02, 0.95, "Waiting for data...", transform=ax.transAxes, va="top")
            return

        t_end = float(t[-1])
        t_start = max(float(t[0]), t_end - self.window_seconds)
        mask = t >= t_start

        ax.plot(t[mask], raw_like[mask], label="Smoothed |acc|")
        ax.plot(t[mask], thr[mask], label="Threshold")

        if steps.size > 0:
            step_mask = steps >= t_start
            steps_win = steps[step_mask]
            if steps_win.size > 0:
                y_steps = np.interp(steps_win, t, raw_like)
                ax.plot(steps_win, y_steps, linestyle="None", marker="o", label="Detected steps")

        ax.set_xlim(t_start, max(t_end, t_start + 1e-3))
        ymin = float(np.min(raw_like[mask])) if np.any(mask) else float(np.min(raw_like))
        ymax = float(np.max(raw_like[mask])) if np.any(mask) else float(np.max(raw_like))
        pad = max(0.2, 0.1 * (ymax - ymin + 1e-6))
        ax.set_ylim(ymin - pad, ymax + pad)

        latest_step = "None"
        if steps.size > 0:
            latest_step = f"{steps[-1]:.3f}s"

        immediate = int(result["new_steps"])
        cumulative = int(result["total_steps"])

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration magnitude")
        ax.legend(loc="upper left")

        info = (
            f"Immediate step count: {immediate}\n"
            f"Cumulative total: {cumulative}\n"
            f"Latest detected step: {latest_step}"
        )
        ax.text(
            0.98,
            0.98,
            info,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
