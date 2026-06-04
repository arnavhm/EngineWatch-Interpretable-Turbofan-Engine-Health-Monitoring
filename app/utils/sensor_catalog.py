"""
app/utils/sensor_catalog.py

Purpose:    Full sensor metadata for the 14 active CMAPSS sensors selected in
            config/config.yaml, sourced from Saxena et al. (2008) Table 2
            "Damage Propagation Modeling for Aircraft Engine Run-to-Failure
            Simulation."

            Keys in each entry:
              symbol          — physical symbol from the paper (e.g. T30)
              description     — full description from Table 2
              units           — measurement units; empty string when dimensionless
              module          — engine module (HPC, LPC, Fan, LPT, HPT, Combustor)
              signal_direction — "rising"  → increases with degradation (degradation indicator)
                                 "falling" → decreases with degradation (health indicator)
              confirmed       — True when monotonic RUL correlation confirmed in
                                Phase 1 analysis; False when informational only
              explanation     — plain English summary for maintenance personnel

Assumptions:
    Keys match the sensor column names produced by data/preprocess.py
    (sensor_2 … sensor_21, a strict subset of config.selected_sensors).
Failure:
    KeyError if a downstream component accesses a key not in this dict.
    Intentionally not a class — plain dict accessed read-only at runtime.
"""

from typing import TypedDict


class SensorMeta(TypedDict):
    symbol: str
    description: str
    units: str
    module: str
    signal_direction: str
    confirmed: bool
    explanation: str


SENSOR_CATALOG: dict[str, SensorMeta] = {
    # ── Fan-axis health indicators (falling = unhealthy) ──────────────────
    "sensor_2": {
        "symbol": "T24",
        "description": "Total temperature at LPC outlet",
        "units": "°R",
        "module": "LPC",
        "signal_direction": "falling",
        "confirmed": False,
        "explanation": (
            "Outlet temperature of the low-pressure compressor. "
            "Falls slightly as LPC compression efficiency degrades — the "
            "compressor does less work on the incoming air. "
            "Informational: individual correlation is weaker than T30 or T50, "
            "but it contributes to the fan-axis health index."
        ),
    },
    "sensor_9": {
        "symbol": "Nc",
        "description": "Physical core speed",
        "units": "rpm",
        "module": "Core",
        "signal_direction": "falling",
        "confirmed": False,
        "explanation": (
            "Actual rotational speed of the high-pressure spool (core). "
            "The engine control system largely holds this steady under "
            "constant thrust demand, so changes are subtle. "
            "Informational: most degradation signal for the core is better "
            "captured by the corrected speed (NRc) and HPC temperatures."
        ),
    },
    "sensor_12": {
        "symbol": "phi",
        "description": "Ratio of fuel flow to Ps30",
        "units": "pps/psi",
        "module": "Combustor",
        "signal_direction": "falling",
        "confirmed": True,
        "explanation": (
            "Fuel flow normalised by HPC outlet static pressure. "
            "Confirmed monotonic trend with RUL: as HPC static pressure (Ps30) "
            "rises with compressor degradation faster than fuel flow changes, "
            "this ratio falls. A steady decrease signals increasing HPC workload "
            "and is one of the cleaner individual indicators in FD001."
        ),
    },
    "sensor_15": {
        "symbol": "BPR",
        "description": "Bypass ratio",
        "units": "",
        "module": "Fan",
        "signal_direction": "falling",
        "confirmed": True,
        "explanation": (
            "Fraction of total airflow that bypasses the core and goes "
            "directly to the nozzle. Falls as fan efficiency degrades and "
            "less air is accelerated through the bypass duct. "
            "Confirmed: strong monotonic correlation with RUL, especially "
            "in single-condition datasets (FD001, FD003)."
        ),
    },
    # ── HPC-axis degradation indicators (rising = unhealthy) ──────────────
    "sensor_3": {
        "symbol": "T30",
        "description": "Total temperature at HPC outlet",
        "units": "°R",
        "module": "HPC",
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
    },
    "sensor_4": {
        "symbol": "T50",
        "description": "Total temperature at LPT outlet",
        "units": "°R",
        "module": "LPT",
        "signal_direction": "rising",
        "confirmed": True,
        "explanation": (
            "Exhaust gas temperature at the low-pressure turbine exit. "
            "Rises as turbine efficiency falls — degraded turbine blades "
            "extract less energy from the hot gas, leaving more thermal energy "
            "in the exhaust. Confirmed strong correlation; elevated T50 also "
            "accelerates thermal fatigue in downstream components."
        ),
    },
    "sensor_7": {
        "symbol": "P30",
        "description": "Total pressure at HPC outlet",
        "units": "psia",
        "module": "HPC",
        "signal_direction": "rising",
        "confirmed": True,
        "explanation": (
            "Total pressure delivered by the high-pressure compressor. "
            "Rises with degradation as the control system commands higher "
            "compressor work to compensate for efficiency loss and maintain "
            "thrust. Confirmed monotonic trend; a rising P30 at constant "
            "thrust set-point is a direct measure of HPC margin erosion."
        ),
    },
    "sensor_8": {
        "symbol": "Nf",
        "description": "Physical fan speed",
        "units": "rpm",
        "module": "Fan",
        "signal_direction": "rising",
        "confirmed": True,
        "explanation": (
            "Actual fan shaft rotational speed. Increases to compensate for "
            "aerodynamic efficiency losses as fan blades erode or accumulate "
            "foreign object damage. A higher physical fan speed at the same "
            "thrust level means the fan is working harder to move the same "
            "mass of air — a confirmed sign of fan degradation."
        ),
    },
    "sensor_11": {
        "symbol": "Ps30",
        "description": "Static pressure at HPC outlet",
        "units": "psia",
        "module": "HPC",
        "signal_direction": "rising",
        "confirmed": True,
        "explanation": (
            "Static (wall) pressure at the HPC exit stage. "
            "Rises alongside total pressure as the compressor compensates "
            "for efficiency loss. Confirmed strong individual correlation; "
            "deviations from the expected Ps30–P30 relationship can indicate "
            "specific stage stall or compressor tip-clearance degradation."
        ),
    },
    "sensor_13": {
        "symbol": "NRf",
        "description": "Corrected fan speed",
        "units": "rpm",
        "module": "Fan",
        "signal_direction": "rising",
        "confirmed": True,
        "explanation": (
            "Fan speed normalised to standard inlet temperature and pressure. "
            "Correcting for operating conditions isolates the true aerodynamic "
            "state of the fan. A rising NRf at constant corrected thrust means "
            "the fan is spinning faster to compensate for efficiency loss — "
            "confirmed as one of the top degradation indicators in FD001."
        ),
    },
    "sensor_14": {
        "symbol": "NRc",
        "description": "Corrected core speed",
        "units": "rpm",
        "module": "Core",
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
    },
    "sensor_17": {
        "symbol": "htBleed",
        "description": "Bleed enthalpy",
        "units": "",
        "module": "HPC",
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
    },
    "sensor_20": {
        "symbol": "W31",
        "description": "HPT coolant bleed (station 31)",
        "units": "lbm/s",
        "module": "HPT",
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
    },
    "sensor_21": {
        "symbol": "W32",
        "description": "LPT coolant bleed (station 32)",
        "units": "lbm/s",
        "module": "LPT",
        "signal_direction": "rising",
        "confirmed": False,
        "explanation": (
            "Mass flow rate of cooling air delivered to the low-pressure "
            "turbine stage. Reflects the thermal management demand as LPT "
            "efficiency falls. Informational in FD001: individual RUL "
            "correlation is weak, but W32 complements T50 when diagnosing "
            "LPT-specific degradation modes."
        ),
    },
}
