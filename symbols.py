# Weekday popular MT5 symbols
def get_symbols():
    symbols = [
        "EURUSDm",
        "GBPUSDm",
        "USDJPYm",
        "AUDUSDm",
        "USDCADm",
        "EURJPYm",
        "GBPJPYm",   # Added
        "AUDJPYm",   # Added
        "CADJPYm",   # Added
        "CHFJPYm",   # Added
        "NZDJPYm",   # Added
    ]

    symbols_dict = {
        "EURUSDm": {"sl_pips": 15, "tp_pips": 60, "trail_start": 16, "trail_step": 2},
        "GBPUSDm": {"sl_pips": 15, "tp_pips": 60, "trail_start": 16, "trail_step": 2},
        "USDJPYm": {"sl_pips": 15, "tp_pips": 60, "trail_start": 16, "trail_step": 2},
        "AUDUSDm": {"sl_pips": 15, "tp_pips": 60, "trail_start": 16, "trail_step": 2},
        "USDCADm": {"sl_pips": 15, "tp_pips": 60, "trail_start": 16, "trail_step": 2},
        "EURJPYm": {"sl_pips": 15, "tp_pips": 60, "trail_start": 16, "trail_step": 2},

        # New JPY pairs
        "GBPJPYm": {"sl_pips": 20, "tp_pips": 80, "trail_start": 20, "trail_step": 3},
        "AUDJPYm": {"sl_pips": 18, "tp_pips": 70, "trail_start": 18, "trail_step": 3},
        "CADJPYm": {"sl_pips": 15, "tp_pips": 60, "trail_start": 16, "trail_step": 2},
        "CHFJPYm": {"sl_pips": 20, "tp_pips": 75, "trail_start": 20, "trail_step": 3},
        "NZDJPYm": {"sl_pips": 18, "tp_pips": 65, "trail_start": 18, "trail_step": 2},
    }

    return {"symbols": symbols, "symbols_dict": symbols_dict}
