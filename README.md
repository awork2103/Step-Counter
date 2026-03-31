# Step Counter

This project runs a step counter in two main ways:

- `offline`: read acceleration data from CSV files and count steps
- `online`: stream live data from `phyphox` and show the real-time step counter plot

## Files

- [run_step_counter.py](/c:/Users/User/Documents/GitHub/Step-Counter/run_step_counter.py): main entry point
- [step_counter.py](/c:/Users/User/Documents/GitHub/Step-Counter/step_counter.py): step counting logic
- [phyphox_stream_client.py](/c:/Users/User/Documents/GitHub/Step-Counter/phyphox_stream_client.py): live phyphox streaming client
- [data/](/c:/Users/User/Documents/GitHub/Step-Counter/data): sample offline CSV files

## Prerequisites

Install Python 3, then install the packages used by the project:

```powershell
pip install numpy pandas matplotlib requests
```

## Datasets

Datasets are located in the `/data` folder. Here are the details for all the datasets:
| Dataset | Duration (s) | Location                                              | Steps |
|----|-------------|--------------------------------------------------------|-------|
| 1  | 60          | HKU Underground Station (Exit C tunnel)               | 105   |
| 2  | 30          | HKU Underground Station (Exit C tunnel)               | 54    |
| 3  | 60          | HKU Main Building corridor outside MB167 (walking up and down)            | 101   |
| 4  | 30          | HKU Main Building corridor outside MB167 (walking up and down)            | 40    |
| 5  | 60          | Inside HKU student flats dormitory                              | 72    |
| 6  | 30          | Inside HKU student flats dormitory                              | 42    |

## How To Run

Run commands from the project root:

```powershell
cd c:\Users\User\Documents\GitHub\Step-Counter
```

### Offline mode

Run all CSV files:

```powershell
python run_step_counter.py
```

Run one CSV file:

```powershell
python run_step_counter.py offline --csv data/data1.csv
```

Run one CSV file and compare it against a known ground-truth step count:

```powershell
python run_step_counter.py offline --csv data/data1.csv --ground-truth 105
```

Run one CSV file and look up the ground-truth value from a CSV file:

```powershell
python run_step_counter.py offline --csv data/data1.csv --ground-truth-csv data/ground_truth.csv
```

Run the batch evaluation over `data/data1.csv`, `data/data2.csv`, and so on:

```powershell
python run_step_counter.py offline-batch
```

You can also point batch mode at a different dataset prefix or ground-truth file:

```powershell
python run_step_counter.py offline-batch --ground-truth data/ground_truth.csv --prefix data/data --suffix .csv
```

Note: if you run `python run_step_counter.py` with no subcommand, the script defaults to `offline-batch`.

### Offline error metrics

Offline evaluation now reports error metrics when ground-truth step counts are available.
If the ground-truth CSV also contains a `location` column, the location is printed for each evaluated test case.

Per-file metrics:

- `signed_error = predicted_steps - ground_truth_steps`
- `absolute_error = |predicted_steps - ground_truth_steps|`
- `percentage_error = 100 * signed_error / ground_truth_steps`
- `absolute_percentage_error = 100 * absolute_error / ground_truth_steps`

If the ground-truth value is `0`, the percentage-based metrics are shown as `N/A`.

Batch summary metrics:

- `mean_error (bias)`: average of all signed errors
- `mean_absolute_error (MAE)`: average of all absolute errors
- `root_mean_squared_error (RMSE)`: square root of the average squared error
- `mean_absolute_percentage_error (MAPE)`: average of all absolute percentage errors for files with non-zero ground truth

In formula form, for `N` evaluated files:

```text
ME   = (1/N) * sum(predicted_i - ground_truth_i)
MAE  = (1/N) * sum(|predicted_i - ground_truth_i|)
RMSE = sqrt((1/N) * sum((predicted_i - ground_truth_i)^2))
MAPE = (1/N_nonzero) * sum(100 * |predicted_i - ground_truth_i| / ground_truth_i)
```

The implementation for these calculations is in `step_counter_metrics.py`.

### Online mode

`online` mode connects to a phone running `phyphox`, checks the remote interface, then starts the live plot:

```powershell
python run_step_counter.py online
```

If auto-detection does not find the right phyphox buffers, you can pass them yourself:

```powershell
python run_step_counter.py online --base-url http://192.168.1.20:8080 --time-buffer time --acc-buffers ax ay az
```

Useful optional flags:

```powershell
python run_step_counter.py online --base-url http://192.168.1.20:8080 --poll-interval 0.05 --window-seconds 12
```

## Changing The Online Stream URL

The script uses this default value in [run_step_counter.py](/c:/Users/User/Documents/GitHub/Step-Counter/run_step_counter.py):

```python
DEFAULT_BASE_URL = "http://172.20.10.1"
```

You have two ways to change the online stream address:

1. Edit `DEFAULT_BASE_URL` in `run_step_counter.py` if you want a new default every time.
2. Pass your own URL with `--base-url` if you only want to override it for one run.

Examples:

```powershell
python run_step_counter.py online --base-url http://172.20.10.1
python run_step_counter.py online --base-url http://192.168.1.20:8080
python run_step_counter.py online --base-url http://10.0.0.15
```

## iOS Note

For iOS phyphox streaming, the port usually does not need to be included because it streams on port `80`.

That means this is fine:

```powershell
python run_step_counter.py online --base-url http://192.168.1.20
```

Instead of:

```powershell
python run_step_counter.py online --base-url http://192.168.1.20:80
```

## Troubleshooting

If online mode cannot connect, check the following:

- the phone and computer are on the same Wi-Fi network
- phyphox remote access is enabled on the phone
- the IP address in `DEFAULT_BASE_URL` or `--base-url` is correct
- required Python packages are installed

If you see `ModuleNotFoundError`, install the missing package with `pip install ...` and run the command again.
