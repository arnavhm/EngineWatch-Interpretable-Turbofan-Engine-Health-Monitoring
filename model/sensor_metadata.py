"""
Sensor metadata derived from Saxena et al. (2008) Table 2 (C-MAPSS outputs).

Keyed by short sensor ID (s1–s21). Active=False sensors are flat under
single-condition datasets (FD001/FD003) — they carry no degradation signal
and are excluded from selected_sensors / PCA. They are still enumerated here
so the Engine Health Map can correctly resolve every module, including those
whose sensors are all flat (burner, epr).

No Streamlit imports. Safe to import from both api/ and app/.
"""

SENSOR_METADATA: dict[str, dict] = {
    "s1":  {"symbol": "T2",        "description": "Total temperature at fan inlet",    "module": "fan",    "units": "°R",      "active": False},
    "s2":  {"symbol": "T24",       "description": "Total temperature at LPC outlet",   "module": "lpc",    "units": "°R",      "active": True},
    "s3":  {"symbol": "T30",       "description": "Total temperature at HPC outlet",   "module": "hpc",    "units": "°R",      "active": True},
    "s4":  {"symbol": "T50",       "description": "Total temperature at LPT outlet",   "module": "lpt",    "units": "°R",      "active": True},
    "s5":  {"symbol": "P2",        "description": "Pressure at fan inlet",             "module": "fan",    "units": "psia",    "active": False},
    "s6":  {"symbol": "P15",       "description": "Total pressure in bypass-duct",     "module": "bypass", "units": "psia",    "active": True},
    "s7":  {"symbol": "P30",       "description": "Total pressure at HPC outlet",      "module": "hpc",    "units": "psia",    "active": True},
    "s8":  {"symbol": "Nf",        "description": "Physical fan speed",                "module": "fan",    "units": "rpm",     "active": True},
    "s9":  {"symbol": "Nc",        "description": "Physical core speed",               "module": "core",   "units": "rpm",     "active": True},
    "s10": {"symbol": "epr",       "description": "Engine pressure ratio (P50/P2)",    "module": "epr",    "units": "—",       "active": False},
    "s11": {"symbol": "Ps30",      "description": "Static pressure at HPC outlet",     "module": "hpc",    "units": "psia",    "active": True},
    "s12": {"symbol": "phi",       "description": "Ratio of fuel flow to Ps30",        "module": "hpc",    "units": "pps/psi", "active": True},
    "s13": {"symbol": "NRf",       "description": "Corrected fan speed",               "module": "fan",    "units": "rpm",     "active": True},
    "s14": {"symbol": "NRc",       "description": "Corrected core speed",              "module": "core",   "units": "rpm",     "active": True},
    "s15": {"symbol": "BPR",       "description": "Bypass ratio",                      "module": "bypass", "units": "—",       "active": True},
    "s16": {"symbol": "farB",      "description": "Burner fuel-air ratio",             "module": "burner", "units": "—",       "active": False},
    "s17": {"symbol": "htBleed",   "description": "Bleed enthalpy",                    "module": "hpc",    "units": "—",       "active": True},
    "s18": {"symbol": "Nf_dmd",    "description": "Demanded fan speed",                "module": "fan",    "units": "rpm",     "active": False},
    "s19": {"symbol": "PCNfR_dmd", "description": "Demanded corrected fan speed",      "module": "fan",    "units": "rpm",     "active": False},
    "s20": {"symbol": "W31",       "description": "HPT coolant bleed",                 "module": "hpt",    "units": "lbm/s",   "active": True},
    "s21": {"symbol": "W32",       "description": "LPT coolant bleed",                 "module": "lpt",    "units": "lbm/s",   "active": True},
}

MODULE_DISPLAY_NAMES: dict[str, str] = {
    "fan":    "Fan",
    "lpc":    "LPC",
    "hpc":    "HPC",
    "hpt":    "HPT",
    "lpt":    "LPT",
    "bypass": "Bypass Duct",
    "core":   "Core Shaft",
    "burner": "Burner",
    "epr":    "EPR / Overall",
}
