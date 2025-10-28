# Weekday popular MT5 symbols
def get_symbols():
    symbols = ["EURUSDm", "GBPUSDm", "USDJPYm", "EURGBPm"]

    symbols_dict = {
        "EURUSDm": {"sl_pips": 15, "tp_pips": 30, "trail_start": 5, "trail_step": 20},
        "GBPUSDm": {"sl_pips": 15, "tp_pips": 30, "trail_start": 7, "trail_step": 22},
        "USDJPYm": {"sl_pips": 10, "tp_pips": 20, "trail_start": 5, "trail_step": 10},
        "EURGBPm": {"sl_pips": 20, "tp_pips": 60, "trail_start": 8, "trail_step": 22},
    }
    details = {
        "symbols":symbols,
        "symbols_dict": symbols_dict
    }
    return details