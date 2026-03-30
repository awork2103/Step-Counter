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

## How To Run

Run commands from the project root:

```powershell
cd c:\Users\User\Documents\GitHub\Step-Counter
```

### Offline mode

Run one CSV file:

```powershell
python run_step_counter.py offline --csv data/data1.csv
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
