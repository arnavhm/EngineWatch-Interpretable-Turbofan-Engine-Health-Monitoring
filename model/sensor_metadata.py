"""
Sensor metadata derived from Saxena et al. (2008) Table 2 (C-MAPSS outputs).

Two exports — both derived from the single SENSOR_METADATA dict:

  SENSOR_METADATA  — keyed by short sensor ID (s1–s21).
                     Consumed by model/predict.py and the Engine Health Map.
                     Includes inactive sensors so every engine module can be
                     resolved on the SVG diagram (burner, epr, etc.).

  SENSOR_CATALOG   — keyed by preprocessed column name (sensor_2 … sensor_21).
                     Consumed by app/components/sensor_panel.py.
                     Active sensors with a confirmed signal direction only.
                     s6 (P15) omitted: active but no confirmed direction.

No Streamlit imports. Safe to import from both api/ and app/.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class SensorMeta(TypedDict, total=False):
    """Full sensor record. All fields present on active sensors; inactive
    sensors carry empty-string / None / False for the interpretability fields."""

    symbol: str
    description: str
    module: str            # Lowercase engine module key, e.g. "hpc"
    module_display: str    # Override display name where catalog differed from MODULE_DISPLAY_NAMES
    units: str
    active: bool
    signal_direction: Optional[str]  # "rising" | "falling" | None for inactive/ambiguous
    confirmed: bool        # True = monotonic RUL correlation confirmed in Phase 1
    explanation: str       # Plain English for maintenance personnel
    layman_text: str       # One-sentence glance summary for non-experts


SENSOR_METADATA: dict[str, SensorMeta] = {
    "s1": {
        "symbol": "T2",
        "description": "Total temperature at fan inlet",
        "module": "fan",
        "units": "°R",
        "active": False,
        "signal_direction": None,
        "confirmed": False,
        "explanation": "",
    },
    "s2": {
        "symbol": "T24",
        "description": "Total temperature at LPC outlet",
        "module": "lpc",
        "units": "°R",
        "active": True,
        "signal_direction": "falling",
        "confirmed": False,
        "explanation": (
            "Outlet temperature of the low-pressure compressor. "
            "Falls slightly as LPC compression efficiency degrades — the "
            "compressor does less work on the incoming air. "
            "Informational: individual correlation is weaker than T30 or T50, "
            "but it contributes to the fan-axis health index."
        ),
        "layman_text": "How hot the air is after the first (low-pressure) compressor. Drifts down a little as that compressor wears.",
    },
    "s3": {
        "symbol": "T30",
        "description": "Total temperature at HPC outlet",
        "module": "hpc",
        "units": "°R",
        "active": True,
        "signal_direction": "rising",
        "confirmed": True,
        "explanation": (
            "Outlet temperature of the high-pressure compressor. "
            "One of the strongest confirmed degradation signals in CMAPSS: "
            "as HPC blade erosion or fouling reduces isentropic efficiency, "
            "more shaft work is dissipated as heat, driving T30 up. "
            "A sustained rise at constant thrust is a direct indicator of "
            "compressor health deterioration."
        ),
        "layman_text": "How hot the air is after the main (high-pressure) compressor. Climbs as the compressor wears — one of the clearest warning signs.",
    },
    "s4": {
        "symbol": "T50",
        "description": "Total temperature at LPT outlet",
        "module": "lpt",
        "units": "°R",
        "active": True,
        "signal_direction": "rising",
        "confirmed": True,
        "explanation": (
            "Exhaust gas temperature at the low-pressure turbine exit. "
            "Rises as turbine efficiency falls — degraded turbine blades "
            "extract less energy from the hot gas, leaving more thermal energy "
            "in the exhaust. Confirmed strong correlation; elevated T50 also "
            "accelerates thermal fatigue in downstream components."
        ),
        "layman_text": "Exhaust temperature leaving the turbine. Rises as the turbine wears and wastes more heat out the back.",
    },
    "s5": {
        "symbol": "P2",
        "description": "Pressure at fan inlet",
        "module": "fan",
        "units": "psia",
        "active": False,
        "signal_direction": None,
        "confirmed": False,
        "explanation": "",
    },
    "s6": {
        "symbol": "P15",
        "description": "Total pressure in bypass-duct",
        "module": "bypass",
        "units": "psia",
        "active": True,
        "signal_direction": None,   # Active but no confirmed monotonic direction
        "confirmed": False,
        "explanation": (
            "Air pressure in the bypass duct. Monitored for completeness, "
            "but in this dataset it has no clear, repeatable trend with wear, "
            "so it is not used as a degradation indicator."
        ),
        "layman_text": (
            "Air pressure in the bypass duct. Watched but not a reliable "
            "wear signal here, so it does not drive the health score."
        ),
    },
    "s7": {
        "symbol": "P30",
        "description": "Total pressure at HPC outlet",
        "module": "hpc",
        "units": "psia",
        "active": True,
        "signal_direction": "rising",
        "confirmed": True,
        "explanation": (
            "Total pressure delivered by the high-pressure compressor. "
            "Rises with degradation as the control system commands higher "
            "compressor work to compensate for efficiency loss and maintain "
            "thrust. Confirmed monotonic trend; a rising P30 at constant "
            "thrust set-point is a direct measure of HPC margin erosion."
        ),
        "layman_text": "Air pressure built up by the main compressor. Creeps up as the engine works harder to hold thrust.",
    },
    "s8": {
        "symbol": "Nf",
        "description": "Physical fan speed",
        "module": "fan",
        "units": "rpm",
        "active": True,
        "signal_direction": "rising",
        "confirmed": True,
        "explanation": (
            "Actual fan shaft rotational speed. Increases to compensate for "
            "aerodynamic efficiency losses as fan blades erode or accumulate "
            "foreign object damage. A higher physical fan speed at the same "
            "thrust level means the fan is working harder to move the same "
            "mass of air — a confirmed sign of fan degradation."
        ),
        "layman_text": "How fast the big front fan is spinning. Speeds up to make up for worn, less-efficient blades.",
    },
    "s9": {
        "symbol": "Nc",
        "description": "Physical core speed",
        "module": "core",
        "units": "rpm",
        "active": True,
        "signal_direction": "falling",
        "confirmed": False,
        "explanation": (
            "Actual rotational speed of the high-pressure spool (core). "
            "The engine control system largely holds this steady under "
            "constant thrust demand, so changes are subtle. "
            "Informational: most degradation signal for the core is better "
            "captured by the corrected speed (NRc) and HPC temperatures."
        ),
        "layman_text": "How fast the engine core is spinning. Held fairly steady by the controller, so it moves only a little.",
    },
    "s10": {
        "symbol": "epr",
        "description": "Engine pressure ratio (P50/P2)",
        "module": "epr",
        "units": "—",
        "active": False,
        "signal_direction": None,
        "confirmed": False,
        "explanation": "",
    },
    "s11": {
        "symbol": "Ps30",
        "description": "Static pressure at HPC outlet",
        "module": "hpc",
        "units": "psia",
        "active": True,
        "signal_direction": "rising",
        "confirmed": True,
        "explanation": (
            "Static (wall) pressure at the HPC exit stage. "
            "Rises alongside total pressure as the compressor compensates "
            "for efficiency loss. Confirmed strong individual correlation; "
            "deviations from the expected Ps30–P30 relationship can indicate "
            "specific stage stall or compressor tip-clearance degradation."
        ),
        "layman_text": "Wall pressure at the main compressor exit. Rises with wear, right alongside P30.",
    },
    "s12": {
        "symbol": "phi",
        "description": "Ratio of fuel flow to Ps30",
        "module": "hpc",
        "module_display": "Combustor",  # phi is a combustor parameter; override HPC display
        "units": "pps/psi",
        "active": True,
        "signal_direction": "falling",
        "confirmed": True,
        "explanation": (
            "Fuel flow normalised by HPC outlet static pressure. "
            "Confirmed monotonic trend with RUL: as HPC static pressure (Ps30) "
            "rises with compressor degradation faster than fuel flow changes, "
            "this ratio falls. A steady decrease signals increasing HPC workload "
            "and is one of the cleaner individual indicators in FD001."
        ),
        "layman_text": "Fuel used per unit of compressor pressure. Slides down as the compressor has to work harder for the same fuel.",
    },
    "s13": {
        "symbol": "NRf",
        "description": "Corrected fan speed",
        "module": "fan",
        "units": "rpm",
        "active": True,
        "signal_direction": "rising",
        "confirmed": True,
        "explanation": (
            "Fan speed normalised to standard inlet temperature and pressure. "
            "Correcting for operating conditions isolates the true aerodynamic "
            "state of the fan. A rising NRf at constant corrected thrust means "
            "the fan is spinning faster to compensate for efficiency loss — "
            "confirmed as one of the top degradation indicators in FD001."
        ),
        "layman_text": "Fan speed adjusted for weather and altitude. A cleaner read on the fan working harder as it wears.",
    },
    "s14": {
        "symbol": "NRc",
        "description": "Corrected core speed",
        "module": "core",
        "units": "rpm",
        "active": True,
        "signal_direction": "rising",
        "confirmed": False,
        "explanation": (
            "Core (HPC) shaft speed normalised to standard inlet conditions. "
            "Increases modestly as the HPC compensates for efficiency loss, "
            "but the control system moderates this change. "
            "Informational: individual correlation is weaker than HPC pressures "
            "and temperatures; most useful in multi-condition datasets where "
            "regime normalisation separates throttle from degradation effects."
        ),
        "layman_text": "Core speed adjusted for weather and altitude. Edges up as the core compensates for wear.",
    },
    "s15": {
        "symbol": "BPR",
        "description": "Bypass ratio",
        "module": "bypass",
        "units": "—",
        "active": True,
        "signal_direction": "falling",
        "confirmed": True,
        "explanation": (
            "Fraction of total airflow that bypasses the core and goes "
            "directly to the nozzle. Falls as fan efficiency degrades and "
            "less air is accelerated through the bypass duct. "
            "Confirmed: strong monotonic correlation with RUL, especially "
            "in single-condition datasets (FD001, FD003)."
        ),
        "layman_text": "Share of air going around the core instead of through it. Drops as the fan loses efficiency.",
    },
    "s16": {
        "symbol": "farB",
        "description": "Burner fuel-air ratio",
        "module": "burner",
        "units": "—",
        "active": False,
        "signal_direction": None,
        "confirmed": False,
        "explanation": "",
    },
    "s17": {
        "symbol": "htBleed",
        "description": "Bleed enthalpy",
        "module": "hpc",
        "units": "—",
        "active": True,
        "signal_direction": "rising",
        "confirmed": True,
        "explanation": (
            "Thermal energy content of bleed air extracted from the HPC for "
            "cooling (turbine blades, nacelle) and cabin pressurisation. "
            "Rises as HPC outlet temperature rises with degradation — hotter "
            "bleed air carries more enthalpy. Confirmed monotonic trend; "
            "persistent rise indicates increasing thermal load on downstream "
            "cooling systems."
        ),
        "layman_text": "Heat carried by the air bled off for cooling. Rises as the compressor runs hotter with wear.",
    },
    "s18": {
        "symbol": "Nf_dmd",
        "description": "Demanded fan speed",
        "module": "fan",
        "units": "rpm",
        "active": False,
        "signal_direction": None,
        "confirmed": False,
        "explanation": "",
    },
    "s19": {
        "symbol": "PCNfR_dmd",
        "description": "Demanded corrected fan speed",
        "module": "fan",
        "units": "rpm",
        "active": False,
        "signal_direction": None,
        "confirmed": False,
        "explanation": "",
    },
    "s20": {
        "symbol": "W31",
        "description": "HPT coolant bleed (station 31)",
        "module": "hpt",
        "units": "lbm/s",
        "active": True,
        "signal_direction": "rising",
        "confirmed": False,
        "explanation": (
            "Mass flow rate of cooling air bled to the high-pressure turbine "
            "stage. Changes as the turbine blade coating degrades and requires "
            "more aggressive cooling to prevent thermal failure. "
            "Informational: signal is noisy and dataset-specific; "
            "more useful in multi-fault datasets (FD003, FD004) where HPT "
            "degradation is explicitly modelled."
        ),
        "layman_text": "Cooling air sent to the high-pressure turbine. Shifts as turbine blades age and need more cooling.",
    },
    "s21": {
        "symbol": "W32",
        "description": "LPT coolant bleed (station 32)",
        "module": "lpt",
        "units": "lbm/s",
        "active": True,
        "signal_direction": "rising",
        "confirmed": False,
        "explanation": (
            "Mass flow rate of cooling air delivered to the low-pressure "
            "turbine stage. Reflects the thermal management demand as LPT "
            "efficiency falls. Informational in FD001: individual RUL "
            "correlation is weak, but W32 complements T50 when diagnosing "
            "LPT-specific degradation modes."
        ),
        "layman_text": "Cooling air sent to the low-pressure turbine. Tracks the turbine's heat load as it ages.",
    },
}


MODULE_DISPLAY_NAMES: dict[str, str] = {
    "fan": "Fan",
    "lpc": "LPC",
    "hpc": "HPC",
    "hpt": "HPT",
    "lpt": "LPT",
    "bypass": "Bypass Duct",
    "core": "Core Shaft",
    "burner": "Burner",
    "epr": "EPR / Overall",
}


# ---------------------------------------------------------------------------
# SENSOR_CATALOG  — derived from SENSOR_METADATA; keyed by preprocessed
# column name (sensor_N format) as produced by data/preprocess.py.
# Consumed by app/components/sensor_panel.py.
# Includes active sensors with a confirmed signal direction only.
# s6 (P15, bypass-duct pressure) is excluded: active but direction=None.
# ---------------------------------------------------------------------------

SENSOR_CATALOG: dict[str, dict] = {
    f"sensor_{k[1:]}": {
        **{field: v[field] for field in ("symbol", "description", "units",
                                         "signal_direction", "confirmed",
                                         "explanation")},
        # Use module_display override where present (e.g. phi → Combustor),
        # otherwise resolve through MODULE_DISPLAY_NAMES.
        "module": v.get("module_display") or MODULE_DISPLAY_NAMES.get(v["module"], v["module"]),
    }
    for k, v in SENSOR_METADATA.items()
    if v["active"] and v.get("signal_direction") is not None
}


# ---------------------------------------------------------------------------
# SYMBOL_TO_META — keyed by sensor SYMBOL (T24, T30, …) for the /sensors API.
# The /sensors cache is keyed by symbol; this lets the route attach
# human-readable metadata to each symbol-keyed value array at request time.
# Active sensors only. Module resolved to display name (phi → Combustor).
# ---------------------------------------------------------------------------

SYMBOL_TO_META: dict[str, dict] = {
    v["symbol"]: {
        "descriptive_name": v["description"],
        "layman_text": v.get("layman_text", ""),
        "explanation": v.get("explanation", ""),
        "units": v["units"],
        "signal_direction": v.get("signal_direction"),
        "confirmed": v.get("confirmed", False),
        "module": v.get("module_display") or MODULE_DISPLAY_NAMES.get(v["module"], v["module"]),
    }
    for v in SENSOR_METADATA.values()
    if v["active"]
}
