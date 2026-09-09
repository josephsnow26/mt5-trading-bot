from datetime import datetime, timezone
import MetaTrader5 as mt5
from decouple import config
import pandas as pd
import os


if not mt5.initialize(
    path=config("MT5_PATHWAY"),
    login=int(config("MT5_USERNAME")),
    password=config("MT5_PASSWORD"),
    server=config("MT5_SERVER"),
):
    print("MT5 init failed:", mt5.last_error())
    exit()

print("MT5 connected:", mt5.account_info().server)

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL = "XCUUSDm"

# Sept 4 2026 NFP — real release 12:30:00 UTC. Window padded a few minutes
# either side so we can see the calm-before / spike / settle-after shape,
# not just the release instant itself.
START = datetime(2026, 9, 4, 12, 25, 0, tzinfo=timezone.utc)
END   = datetime(2026, 9, 4, 12, 35, 0, tzinfo=timezone.utc)

# ── Select symbol ─────────────────────────────────────────────────────────────
if not mt5.symbol_select(SYMBOL, True):
    print(f"Failed to select {SYMBOL}:", mt5.last_error())
    mt5.shutdown()
    exit()

info = mt5.symbol_info(SYMBOL)
if info is None:
    print(f"Symbol {SYMBOL} not found on this broker")
    mt5.shutdown()
    exit()

print(f"Symbol OK: {SYMBOL}")
print(f"Digits: {info.digits}  Point: {info.point}")

# ── Pull ticks ────────────────────────────────────────────────────────────────
# COPY_TICKS_ALL = both bid/ask quote ticks AND trade ticks. This is the raw
# feed — every price update MT5's terminal has cached locally for this window,
# not a compressed bar. Same silent-partial-result risk as copy_rates_* — if
# the terminal's local tick history doesn't cover this window yet, this can
# come back empty or short rather than erroring, so the row-count/range print
# below matters, don't skip reading it.
print(f"\nFetching ticks {START} -> {END}...")
ticks = mt5.copy_ticks_range(SYMBOL, START, END, mt5.COPY_TICKS_ALL)

if ticks is None or len(ticks) == 0:
    print("copy_ticks_range returned nothing:", mt5.last_error())
    print(
        "If this is empty, MT5's local tick cache likely doesn't reach back "
        "to this window yet. Fix: open XAUUSDm's chart in the terminal, "
        "scroll back through Sept 4 2026 manually to force the broker to "
        "backfill tick history, then re-run this script."
    )
    mt5.shutdown()
    exit()

df = pd.DataFrame(ticks)
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
df["time_msc"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)

print(f"\nFetched {len(df)} ticks")
print(f"Range: {df['time_msc'].iloc[0]} -> {df['time_msc'].iloc[-1]}")
print(f"\nColumns: {list(df.columns)}")
print(df[["time_msc", "bid", "ask", "last", "volume", "flags"]].head(20).to_string())

# ── Gap check — same idea as the M1 export, but the threshold means something
# different here: a gap of even a few hundred milliseconds during the release
# second is the actual thing we're looking for (it's the "no liquidity at
# that price" gap), so this just flags it loudly rather than needing a
# separate holiday-aware exception list like the M1 gap check does.
diffs_ms = df["time_msc"].diff().dt.total_seconds() * 1000
GAP_THRESHOLD_MS = 200  # flag any tick-to-tick gap wider than this
gap_mask = diffs_ms > GAP_THRESHOLD_MS
gaps = df.loc[gap_mask]

if gaps.empty:
    print(f"\nNo tick gaps wider than {GAP_THRESHOLD_MS}ms in this window.")
else:
    print(f"\n*** {len(gaps)} tick gap(s) wider than {GAP_THRESHOLD_MS}ms — these are the real liquidity-gap candidates ***")
    for idx in gaps.index:
        prev_bid = df["bid"].iloc[idx - 1]
        prev_ask = df["ask"].iloc[idx - 1]
        this_bid = df["bid"].iloc[idx]
        this_ask = df["ask"].iloc[idx]
        gap_start = df["time_msc"].iloc[idx - 1]
        gap_end = df["time_msc"].iloc[idx]
        print(
            f"  {gap_start} -> {gap_end}  ({(gap_end-gap_start).total_seconds()*1000:.0f}ms)  "
            f"bid {prev_bid}->{this_bid}  ask {prev_ask}->{this_ask}"
        )

# ── Save ──────────────────────────────────────────────────────────────────────
save_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(save_dir, exist_ok=True)

filename = f"{SYMBOL}_ticks_{START.strftime('%Y%m%d_%H%M')}_{END.strftime('%H%M')}.csv"
save_path = os.path.join(save_dir, filename)

df.to_csv(save_path, index=False)
print(f"\nSaved -> {save_path}")

mt5.shutdown()