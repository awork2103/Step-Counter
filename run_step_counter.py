import argparse
from pathlib import Path

import pandas as pd
from step_counter import StepCounter
from step_counter_metrics import StepCounterErrorMetrics

DEFAULT_BASE_URL = "http://172.20.10.1"


def load_csv(filepath):
    df = pd.read_csv(filepath)

    cols = list(df.columns)
    norm = {c: c.strip().lower() for c in cols}

    time_col = None
    x_col = None
    y_col = None
    z_col = None

    for original, c in norm.items():
        if time_col is None and "time" in c:
            time_col = original

        if x_col is None and (
            "linear acceleration x" in c
            or c == "ax"
            or "acc x" in c
            or "acceleration x" in c
        ):
            x_col = original

        if y_col is None and (
            "linear acceleration y" in c
            or c == "ay"
            or "acc y" in c
            or "acceleration y" in c
        ):
            y_col = original

        if z_col is None and (
            "linear acceleration z" in c
            or c == "az"
            or "acc z" in c
            or "acceleration z" in c
        ):
            z_col = original

    if time_col is None or x_col is None or y_col is None or z_col is None:
        raise ValueError(
            "Could not find required columns.\n"
            f"Found columns: {list(df.columns)}"
        )

    time = df[time_col].to_numpy(dtype=float)
    acc = df[[x_col, y_col, z_col]].to_numpy(dtype=float)

    return {
        "time": time,
        "acc": acc,
    }


def read_ground_truth(filepath):
    df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip().str.replace('"', "", regex=False)

    ground_truth = {}

    for _, row in df.iterrows():
        filename = str(row["filename"])
        step_count = int(row["step_count"])
        location = row["location"] if "location" in row and pd.notna(row["location"]) else None
        if location is not None:
            location = str(location).strip()
        ground_truth[filename] = {
            "step_count": step_count,
            "location": location,
        }

    return ground_truth


def lookup_ground_truth(ground_truth, csv_file):
    path = Path(csv_file)
    candidates = [
        str(csv_file),
        str(csv_file).replace("\\", "/"),
        path.name,
        path.stem,
    ]

    trailing_digits = "".join(ch for ch in path.stem if ch.isdigit())
    if trailing_digits:
        candidates.append(str(int(trailing_digits)))

    for candidate in candidates:
        if candidate in ground_truth:
            return ground_truth[candidate]

    return None


def print_sample_error_metrics(metrics):
    print(f"Signed error (pred - gt): {metrics['signed_error']}")
    print(f"Absolute error: {metrics['absolute_error']}")
    if metrics["percentage_error"] is None:
        print("Percentage error: N/A (ground truth is 0)")
        print("Absolute percentage error: N/A (ground truth is 0)")
    else:
        print(f"Percentage error: {metrics['percentage_error']:.2f}%")
        print(f"Absolute percentage error: {metrics['absolute_percentage_error']:.2f}%")


def print_summary_error_metrics(summary):
    print("==== Error Summary ====")
    print(f"Files evaluated: {summary['num_samples']}")
    print(f"Mean error (bias): {summary['mean_error']:.2f}")
    print(f"Mean absolute error (MAE): {summary['mean_absolute_error']:.2f}")
    print(f"Root mean squared error (RMSE): {summary['root_mean_squared_error']:.2f}")
    if summary["mape"] is None:
        print("Mean absolute percentage error (MAPE): N/A")
    else:
        print(f"Mean absolute percentage error (MAPE): {summary['mape']:.2f}%")
    print("=======================\n")


def run_offline_batch(ground_truth_csv="data/ground_truth.csv", prefix="data/data", suffix=".csv"):
    file_counter = 0
    ground_truth = read_ground_truth(ground_truth_csv)
    metrics_calculator = StepCounterErrorMetrics()
    predicted_steps = []
    ground_truth_steps = []

    for i in range(1, 100):
        file_counter += 1
        csv_file = f"{prefix}{file_counter}{suffix}"

        try:
            data = load_csv(csv_file)

            step_counter = StepCounter()
            result = step_counter.run_offline(data)

            gt = lookup_ground_truth(ground_truth, csv_file)

            print(f"==== Step Counter Result {file_counter} ====")
            print(f"Step count: {result['step_count']}")
            if gt is not None and gt.get("location"):
                print(f"Location: {gt['location']}")
            print(f"Ground truth: {gt['step_count'] if gt is not None else 'N/A'}")
            if gt is not None:
                sample_metrics = metrics_calculator.calculate_sample_metrics(
                    result["step_count"],
                    gt["step_count"],
                )
                predicted_steps.append(result["step_count"])
                ground_truth_steps.append(gt["step_count"])
                print_sample_error_metrics(sample_metrics)
            print("=============================\n")
        except FileNotFoundError:
            break

    if predicted_steps:
        summary = metrics_calculator.calculate_summary_metrics(
            predicted_steps,
            ground_truth_steps,
        )
        print_summary_error_metrics(summary)


def run_offline_single(csv_file, ground_truth_step_count=None, ground_truth_csv=None):
    data = load_csv(csv_file)
    step_counter = StepCounter()
    result = step_counter.run_offline(data)
    metrics_calculator = StepCounterErrorMetrics()

    gt_record = None
    if ground_truth_step_count is None and ground_truth_csv is not None:
        ground_truth = read_ground_truth(ground_truth_csv)
        gt_record = lookup_ground_truth(ground_truth, csv_file)
    elif ground_truth_step_count is not None:
        gt_record = {
            "step_count": int(ground_truth_step_count),
            "location": None,
        }

    print("==== Step Counter Result ====")
    print(f"Step count: {result['step_count']}")
    print(f"Detected steps: {len(result['step_timestamps'])}")
    print("First 10 timestamps:", result["step_timestamps"][:10])
    if gt_record is not None and gt_record.get("location"):
        print(f"Location: {gt_record['location']}")
    if gt_record is not None:
        print(f"Ground truth: {gt_record['step_count']}")
        sample_metrics = metrics_calculator.calculate_sample_metrics(
            result["step_count"],
            gt_record["step_count"],
        )
        print_sample_error_metrics(sample_metrics)
    print("=============================\n")


def run_online_phyphox(
    base_url=DEFAULT_BASE_URL,
    time_buffer=None,
    acc_buffers=None,
    poll_interval=0.05,
    window_seconds=12.0,
):
    step_counter = StepCounter()
    step_counter.run_online(
        base_url=base_url,
        time_buffer=time_buffer,
        acc_buffers=acc_buffers,
        poll_interval=poll_interval,
        window_seconds=window_seconds,
    )


def run_replay(csv_file, chunk_size=5, speedup=1.0, window_seconds=12.0):
    data = load_csv(csv_file)
    step_counter = StepCounter()
    step_counter.run_replay(
        data=data,
        chunk_size=chunk_size,
        speedup=speedup,
        window_seconds=window_seconds,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run offline or online step counting.")
    subparsers = parser.add_subparsers(dest="mode")

    offline_batch = subparsers.add_parser(
        "offline-batch",
        help="Run offline batch evaluation over data/data1.csv, data/data2.csv, ..."
    )
    offline_batch.add_argument("--ground-truth", default="data/ground_truth.csv")
    offline_batch.add_argument("--prefix", default="data/data")
    offline_batch.add_argument("--suffix", default=".csv")

    offline_one = subparsers.add_parser(
        "offline",
        help="Run offline step counting for one CSV file."
    )
    offline_one.add_argument("--csv", required=True)
    offline_one.add_argument("--ground-truth", type=int, default=None)
    offline_one.add_argument("--ground-truth-csv", default=None)

    online = subparsers.add_parser(
        "online",
        help="Run real-time phyphox streaming and visualization."
    )
    online.add_argument("--base-url", default=None, help="Example: http://172.20.10.1")
    online.add_argument("--time-buffer", default=None)
    online.add_argument("--acc-buffers", nargs=3, default=None, metavar=("AX", "AY", "AZ"))
    online.add_argument("--poll-interval", type=float, default=0.05)
    online.add_argument("--window-seconds", type=float, default=12.0)

    replay = subparsers.add_parser(
        "replay",
        help="Replay a CSV through the same real-time visualization."
    )
    replay.add_argument("--csv", required=True)
    replay.add_argument("--chunk-size", type=int, default=5)
    replay.add_argument("--speedup", type=float, default=1.0)
    replay.add_argument("--window-seconds", type=float, default=12.0)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode is None:
        run_offline_batch()
        return

    if args.mode == "offline-batch":
        run_offline_batch(
            ground_truth_csv=args.ground_truth,
            prefix=args.prefix,
            suffix=args.suffix,
        )

    elif args.mode == "offline":
        run_offline_single(
            args.csv,
            ground_truth_step_count=args.ground_truth,
            ground_truth_csv=args.ground_truth_csv,
        )

    elif args.mode == "online":
        import requests

        base_url = args.base_url if args.base_url else DEFAULT_BASE_URL
        print(f"[INFO] Using phyphox base URL: {base_url}")

        try:
            requests.get(base_url + "/config", timeout=2).raise_for_status()
        except Exception:
            print("[ERROR] Cannot connect to phyphox.")
            print("Make sure:")
            print("- Phone and laptop are on same WiFi")
            print("- Phyphox remote access is enabled")
            print("- IP address is correct")
            return

        run_online_phyphox(
            base_url=base_url,
            time_buffer=args.time_buffer,
            acc_buffers=args.acc_buffers,
            poll_interval=args.poll_interval,
            window_seconds=args.window_seconds,
        )

    elif args.mode == "replay":
        run_replay(
            csv_file=args.csv,
            chunk_size=args.chunk_size,
            speedup=args.speedup,
            window_seconds=args.window_seconds,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
