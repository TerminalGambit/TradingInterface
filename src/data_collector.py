"""
BTC Price Data Collector for Training
Runs independently to collect historical data 24/7.
Saves hourly files for ML training.
Press Ctrl+C to stop.
"""

import requests
import time
import json
import gzip
from datetime import datetime
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "fetch_interval": 1.0,          # 1 request/sec - safe rate limit
    "save_interval": 60,            # Save to file every 60 seconds
    "compress_after_hours": 24,     # Compress files older than 24 hours
    "max_uncompressed_files": 48,   # Keep last 48 hours uncompressed
}

BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"
SYMBOL = "BTCUSDT"
DATA_DIR = Path(__file__).parent / "training_data"
STATE_FILE = DATA_DIR / "collector_state.json"


def ensure_data_dir():
    """Create training_data directory if it doesn't exist."""
    DATA_DIR.mkdir(exist_ok=True)


def get_hourly_filename(dt: datetime = None) -> Path:
    """Get filename for the current hour."""
    if dt is None:
        dt = datetime.now()
    return DATA_DIR / f"btc_{dt.strftime('%Y_%m_%d_%H')}.json"


def fetch_btc_price() -> dict:
    """Fetch current BTC price from Binance API."""
    response = requests.get(BINANCE_API_URL, params={"symbol": SYMBOL}, timeout=5)
    response.raise_for_status()
    data = response.json()
    return {
        "timestamp": datetime.now().isoformat(),
        "epoch": time.time(),
        "price": float(data["price"])
    }


def load_hourly_data(filepath: Path) -> list:
    """Load data from hourly file (supports both json and gzip)."""
    if not filepath.exists():
        # Check for compressed version
        gz_path = filepath.with_suffix(".json.gz")
        if gz_path.exists():
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        return []

    with open(filepath, "r") as f:
        return json.load(f)


def save_hourly_data(filepath: Path, data: list):
    """Save data to hourly file."""
    with open(filepath, "w") as f:
        json.dump(data, f)


def load_state() -> dict:
    """Load collector state (for resume capability)."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_tick_epoch": 0, "total_ticks": 0}


def save_state(state: dict):
    """Save collector state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def compress_old_files():
    """Compress files older than configured hours."""
    now = datetime.now()
    json_files = sorted(DATA_DIR.glob("btc_*.json"))

    # Keep only recent files uncompressed
    files_to_compress = json_files[:-CONFIG["max_uncompressed_files"]] if len(json_files) > CONFIG["max_uncompressed_files"] else []

    for filepath in files_to_compress:
        gz_path = filepath.with_suffix(".json.gz")
        if not gz_path.exists():
            print(f"\nCompressing {filepath.name}...")
            with open(filepath, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    f_out.write(f_in.read())
            filepath.unlink()  # Remove original after compression
            print(f"  -> {gz_path.name}")


def get_stats() -> dict:
    """Get statistics about collected data."""
    json_files = list(DATA_DIR.glob("btc_*.json"))
    gz_files = list(DATA_DIR.glob("btc_*.json.gz"))

    total_ticks = 0
    for f in json_files:
        try:
            data = load_hourly_data(f)
            total_ticks += len(data)
        except:
            pass

    return {
        "uncompressed_files": len(json_files),
        "compressed_files": len(gz_files),
        "total_files": len(json_files) + len(gz_files),
        "estimated_ticks": total_ticks + (len(gz_files) * 3600),  # Estimate ~3600 ticks/hour
    }


def run_collector():
    """Main collection loop."""
    ensure_data_dir()

    state = load_state()
    print("=" * 50)
    print("BTC Data Collector for Training")
    print("=" * 50)
    print(f"Data directory: {DATA_DIR}")
    print(f"Fetch interval: {CONFIG['fetch_interval']}s")
    print(f"Previous session ticks: {state['total_ticks']}")

    stats = get_stats()
    print(f"\nExisting data:")
    print(f"  Uncompressed files: {stats['uncompressed_files']}")
    print(f"  Compressed files: {stats['compressed_files']}")
    print(f"  Estimated total ticks: {stats['estimated_ticks']:,}")
    print("\nPress Ctrl+C to stop.\n")

    current_hour = datetime.now().hour
    current_file = get_hourly_filename()
    hourly_data = load_hourly_data(current_file)
    last_save_time = time.time()
    session_ticks = 0

    try:
        while True:
            # Fetch new price
            try:
                new_data = fetch_btc_price()
                hourly_data.append(new_data)
                session_ticks += 1
                state["last_tick_epoch"] = new_data["epoch"]
                state["total_ticks"] += 1
            except requests.RequestException as e:
                print(f"\n[Error] API request failed: {e}")
                time.sleep(5)  # Wait before retry
                continue

            # Check if hour changed -> rotate file
            now = datetime.now()
            if now.hour != current_hour:
                # Save current hour's data
                save_hourly_data(current_file, hourly_data)
                print(f"\n[Rotate] Saved {len(hourly_data)} ticks to {current_file.name}")

                # Start new hour
                current_hour = now.hour
                current_file = get_hourly_filename()
                hourly_data = load_hourly_data(current_file)

                # Compress old files
                compress_old_files()

            # Periodic save
            current_time = time.time()
            if current_time - last_save_time >= CONFIG["save_interval"]:
                save_hourly_data(current_file, hourly_data)
                save_state(state)
                last_save_time = current_time

            # Status line
            print(f"\r[{new_data['timestamp'][11:19]}] ${new_data['price']:,.2f} | "
                  f"Hour: {len(hourly_data)} | Session: {session_ticks} | Total: {state['total_ticks']:,}",
                  end="", flush=True)

            time.sleep(CONFIG["fetch_interval"])

    except KeyboardInterrupt:
        print(f"\n\n[Stopping] Saving data...")
        save_hourly_data(current_file, hourly_data)
        save_state(state)
        print(f"Saved {len(hourly_data)} ticks to {current_file.name}")
        print(f"Session ticks: {session_ticks}")
        print(f"Total ticks: {state['total_ticks']:,}")

    except Exception as e:
        print(f"\n\n[Error] {e}")
        save_hourly_data(current_file, hourly_data)
        save_state(state)
        raise


if __name__ == "__main__":
    run_collector()
