# Weekday popular MT5 symbols
def get_symbols():
    symbols = ["EURUSDm", "GBPUSDm"]

    symbols_dict = {
        "EURUSDm": {"sl_pips": 9, "tp_pips": 200, "trail_start": 20, "trail_step": 2},
        "GBPUSDm": {"sl_pips": 9, "tp_pips": 200, "trail_start": 20, "trail_step": 2},
    }
    details = {"symbols": symbols, "symbols_dict": symbols_dict}
    return details
