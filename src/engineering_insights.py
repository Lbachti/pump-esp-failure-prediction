"""Petroleum-engineering knowledge base and dataset-derived interpretation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import FEATURE_LABELS, FEATURE_UNITS
from .eda import univariate_signal_table

PARAMETER_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "intake_pressure": {
        "label": "Intake pressure",
        "unit": "psi",
        "summary": "Pressure at the pump suction; a direct indicator of inflow conditions and drawdown.",
        "established": [
            "Suction pressure reflects the balance between reservoir inflow and pump withdrawal.",
            "Low suction pressure can indicate excessive drawdown, pump-off, or gas interference.",
            "High suction pressure can indicate restricted inflow, high backpressure, or a pump operating far from its design point.",
        ],
        "mechanisms": [
            "Gas interference and slugging reduce pump volumetric efficiency and can cause gas lock.",
            "Insufficient submergence can lead to overheating and poor lubrication of pump stages.",
            "Rapid drawdown changes can destabilise the pump operating point.",
        ],
        "diagnostics": [
            "Compare with bubble-point pressure and pump intake submergence.",
            "Trend with production and frequency to separate inflow changes from speed changes.",
        ],
        "caveat": "A single pressure value cannot distinguish gas interference from inflow decline or a restricted completion.",
    },
    "discharge_pressure": {
        "label": "Discharge pressure",
        "unit": "psi",
        "summary": "Pressure at the pump outlet; reflects the total dynamic head required by the system.",
        "established": [
            "Discharge pressure is set by tubing hydraulics, flowline pressure, fluid gradient, and pump head.",
            "A rising discharge pressure at constant rate can indicate a restriction or increasing head requirement.",
            "A falling discharge pressure at constant speed can indicate pump wear, gas, or loss of stage performance.",
        ],
        "mechanisms": [
            "Scale, wax, or sand deposition can restrict tubing and raise discharge pressure.",
            "Pump wear or stage damage reduces generated head and discharge pressure.",
            "Gas interference reduces the liquid column and changes the pressure profile.",
        ],
        "diagnostics": [
            "Evaluate pressure differential (discharge − intake) together with rate and frequency.",
            "Compare against the manufacturer's head-flow curve at the operating frequency.",
        ],
        "caveat": "Discharge pressure alone does not identify whether the cause is the pump, tubing, or surface system.",
    },
    "motor_temperature": {
        "label": "Motor temperature",
        "unit": "°F",
        "summary": "ESP motor winding temperature; a key indicator of electrical and cooling performance.",
        "established": [
            "Motor insulation life is strongly temperature-dependent; sustained overheating accelerates degradation.",
            "The motor is cooled by produced fluid flowing past the motor and by conduction to the formation.",
            "Low flow, gas locking, high load, or poor motor cooling can raise winding temperature.",
        ],
        "mechanisms": [
            "Insulation ageing and winding failure under sustained thermal stress.",
            "Overload or voltage imbalance increases losses and heating.",
            "Inadequate submergence or low flow reduces cooling.",
        ],
        "diagnostics": [
            "Compare with the motor manufacturer's insulation class and operating limits.",
            "Review current, voltage balance, load factor, and flow together with temperature.",
        ],
        "caveat": "Temperature is affected by sensor placement, fluid properties, and operating history; a universal alarm limit is not appropriate.",
    },
    "intake_temperature": {
        "label": "Intake temperature",
        "unit": "°F",
        "summary": "Well-fluid temperature at the pump intake; influences viscosity, gas behaviour, and cooling.",
        "established": [
            "Fluid temperature affects viscosity, gas solubility, and multiphase flow behaviour.",
            "A changing intake temperature can reflect changes in inflow, geothermal gradient, or produced-fluid mix.",
        ],
        "mechanisms": [
            "Higher temperature can reduce viscosity but also affects gas breakout and seal elastomers.",
            "Temperature changes may accompany water breakthrough or changes in produced-fluid composition.",
        ],
        "diagnostics": [
            "Compare with surface and reservoir temperatures and produced-fluid composition.",
            "Do not use intake temperature alone as a failure indicator.",
        ],
        "caveat": "In this dataset the intake-temperature range is narrow; weak associations should not be over-interpreted.",
    },
    "motor_amps": {
        "label": "Motor current",
        "unit": "A",
        "summary": "Motor current reflects electrical loading and mechanical power demand.",
        "established": [
            "Current is related to motor load, voltage, power factor, and fluid viscosity.",
            "High current can indicate overload, high viscosity, mechanical drag, or voltage imbalance.",
            "Low current can indicate gas interference, pump-off, or loss of hydraulic load.",
        ],
        "mechanisms": [
            "Overcurrent accelerates winding heating and insulation degradation.",
            "Current imbalance can indicate supply or cable problems.",
            "Rapid current fluctuations can accompany gas slugging.",
        ],
        "diagnostics": [
            "Compare current with nameplate current, voltage, load factor, and production.",
            "Trend current against frequency and rate to separate load changes from supply changes.",
        ],
        "caveat": "Current is not a direct failure measurement; it is an electrical response to the whole lift system.",
    },
    "motor_volt": {
        "label": "Motor voltage",
        "unit": "V",
        "summary": "Voltage supplied to the motor, affected by the VSD, cable, and supply quality.",
        "established": [
            "Low voltage increases current for a given power demand and can cause overheating.",
            "Voltage imbalance produces negative-sequence currents and additional motor heating.",
            "Cable voltage drop increases with current, cable length, and temperature.",
        ],
        "mechanisms": [
            "Undervoltage or imbalance can lead to overload, tripping, and insulation stress.",
            "VSD harmonics and cable effects can distort the voltage waveform.",
        ],
        "diagnostics": [
            "Measure at the switchboard and at the motor terminals where possible.",
            "Review VSD settings, cable sizing, and supply quality.",
        ],
        "caveat": "A single voltage reading cannot establish power quality; waveform and phase data are needed.",
    },
    "vibration": {
        "label": "Vibration",
        "unit": "g",
        "summary": "Mechanical vibration; a direct indicator of rotating-equipment condition.",
        "established": [
            "Vibration can indicate imbalance, misalignment, bearing wear, looseness, or flow-induced excitation.",
            "Changing vibration signatures often precede mechanical failures.",
            "Sensor location and mounting strongly affect measured amplitude.",
        ],
        "mechanisms": [
            "Bearing and shaft wear under sustained vibration.",
            "Mechanical rubs, stage damage, or sand production.",
            "Gas slugging and unstable flow can excite vibration.",
        ],
        "diagnostics": [
            "Use frequency-domain analysis and manufacturer vibration criteria.",
            "Compare with baseline signatures at similar speed and flow.",
        ],
        "caveat": "Vibration is highly installation-specific; a universal threshold is not appropriate.",
    },
    "frequency": {
        "label": "Frequency",
        "unit": "Hz",
        "summary": "VSD output frequency, which sets pump speed and affects flow, head, and load.",
        "established": [
            "Pump speed changes flow approximately linearly and head with the square of speed.",
            "Operating above the design frequency can overload the motor and pump.",
            "Operating below the design range can reduce cooling and lift efficiency.",
        ],
        "mechanisms": [
            "Speed excursions move the pump away from its best-efficiency point.",
            "Frequent speed changes create transient loads and thermal cycling.",
        ],
        "diagnostics": [
            "Compare with the pump curve, motor rating, and intended production target.",
            "Review VSD tuning and operating envelope.",
        ],
        "caveat": "Frequency is a control variable, not a failure symptom by itself.",
    },
    "production": {
        "label": "Production rate",
        "unit": "bbl/d",
        "summary": "Measured or allocated production; integrates pump performance and inflow behaviour.",
        "established": [
            "Declining production at constant speed and pressure can indicate pump wear, gas, or inflow decline.",
            "Production is affected by allocation uncertainty, downtime, and operational changes.",
            "Rate should be interpreted with head, power, and frequency.",
        ],
        "mechanisms": [
            "Loss of pump efficiency reduces rate and increases power per barrel.",
            "Gas interference and slugging cause rate instability.",
            "Reservoir pressure decline reduces inflow and production.",
        ],
        "diagnostics": [
            "Compare with expected rate from the pump curve and inflow model.",
            "Use rate-per-power or rate-per-head indicators where data quality allows.",
        ],
        "caveat": "Production allocation can be uncertain; do not treat a single daily value as a precise pump-performance measurement.",
    },
}

FAILURE_MECHANISMS = [
    {
        "title": "Motor overheating / insulation degradation",
        "description": "Sustained thermal stress accelerates insulation ageing and can lead to winding failure.",
        "indicators": ["Motor temperature", "Motor current", "Voltage balance", "Flow / cooling"],
        "note": "Temperature limits are manufacturer- and insulation-class-specific.",
    },
    {
        "title": "Mechanical wear and vibration",
        "description": "Bearing wear, imbalance, misalignment, and stage damage produce vibration and progressive performance loss.",
        "indicators": ["Vibration", "Current", "Discharge pressure", "Production"],
        "note": "Vibration signatures require frequency-domain and baseline analysis.",
    },
    {
        "title": "Gas interference / gas lock",
        "description": "Free gas at the intake reduces pump efficiency and can cause unstable or zero liquid throughput.",
        "indicators": ["Intake pressure", "Discharge pressure", "Current", "Production"],
        "note": "Compare intake pressure with bubble-point pressure and submergence.",
    },
    {
        "title": "Scale, wax, sand, and deposition",
        "description": "Deposition restricts flow paths, raises head requirements, and can damage stages.",
        "indicators": ["Discharge pressure", "Production", "Current", "Vibration"],
        "note": "Requires fluid chemistry and completion information.",
    },
    {
        "title": "Electrical supply and cable problems",
        "description": "Undervoltage, imbalance, harmonics, and cable degradation increase losses and can cause trips.",
        "indicators": ["Motor voltage", "Motor current", "Temperature"],
        "note": "Power-quality measurements are needed for diagnosis.",
    },
    {
        "title": "Pump hydraulic wear / stage damage",
        "description": "Worn stages reduce head and efficiency, changing the pressure-rate relationship.",
        "indicators": ["Discharge pressure", "Production", "Current", "Frequency"],
        "note": "Compare with the manufacturer's head-flow curve.",
    },
    {
        "title": "Inflow decline / pump-off",
        "description": "Insufficient inflow reduces submergence, cooling, and production, and may cause pump-off.",
        "indicators": ["Intake pressure", "Production", "Current", "Temperature"],
        "note": "Requires reservoir and inflow analysis.",
    },
    {
        "title": "Operational excursions",
        "description": "Frequent speed changes, restart cycling, and operation outside the design envelope create transient loads.",
        "indicators": ["Frequency", "Current", "Vibration", "Temperature"],
        "note": "Requires operating history and control-system data.",
    },
]

EVIDENCE_LAYERS = [
    {
        "title": "Established ESP engineering principle",
        "text": "General physical or manufacturer-based relationship documented in artificial-lift practice. It is not proof about this specific dataset.",
    },
    {
        "title": "Dataset-derived statistical relationship",
        "text": "A measured association in the loaded data (group difference, correlation, univariate AUC). It may reflect confounding, data generation, or sampling.",
    },
    {
        "title": "Model association",
        "text": "Feature importance, coefficient magnitude, or SHAP attribution from a fitted model. It explains the model, not the physical system.",
    },
    {
        "title": "Hypothesis requiring field validation",
        "text": "A plausible mechanism that must be tested with equipment specifications, high-frequency data, maintenance records, and controlled field studies.",
    },
]


def parameter_stat(df: pd.DataFrame, feature: str) -> dict[str, Any] | None:
    table = univariate_signal_table(df, [feature])
    if table.empty:
        return None
    return table.iloc[0].to_dict()


def dataset_parameter_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return univariate_signal_table(df, features)


def engineering_disclaimer() -> str:
    return (
        "ESP diagnosis requires the actual pump and motor specifications, manufacturer "
        "limits, completion and cable data, fluid properties, reservoir conditions, "
        "high-frequency sensor data, and field maintenance history. The relationships "
        "shown here are educational and statistical; they are not universal operating "
        "limits and must not be used as the sole basis for a field decision."
    )


def classification_vs_prediction() -> str:
    return (
        "This application performs current-state failure classification: it estimates "
        "whether the operational snapshot belongs to the 'failed' class. It does not "
        "predict that a pump will fail in the next N days, because the dataset has no "
        "future-failure label or lead time. A true predictive-maintenance target would "
        "require run-to-failure histories and a label such as 'failure within the next "
        "30 days', followed by temporal validation that respects the forecast horizon."
    )


DEPLOYMENT_REQUIREMENTS = [
    "Run-to-failure histories with consistent failure timestamps and maintenance work orders.",
    "High-frequency motor, pressure, temperature, vibration, and VSD data rather than daily snapshots.",
    "Equipment specifications, pump curves, motor ratings, cable data, and completion details.",
    "Fluid properties, reservoir pressure, gas-oil-ratio, water cut, and sand production.",
    "A prospective validation protocol with a defined lead time, alarm threshold, and operational response.",
    "Probability calibration, drift monitoring, and periodic retraining on field data.",
    "Alarm-management design that accounts for the cost of false positives and missed failures.",
]


def esp_schematic_html() -> str:
    """A compact, theme-matched ESP completion schematic."""
    return """
<div class="schematic-card">
  <div class="schematic-title">ESP completion — simplified system view</div>
  <svg viewBox="0 0 980 350" role="img" aria-label="Simplified electric submersible pump completion schematic" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="metal" x1="0" x2="1">
        <stop offset="0" stop-color="#17324f"/><stop offset="1" stop-color="#0e2138"/>
      </linearGradient>
      <linearGradient id="fluid" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#22d3ee" stop-opacity=".55"/><stop offset="1" stop-color="#2d7dd2" stop-opacity=".1"/>
      </linearGradient>
    </defs>
    <line x1="40" y1="42" x2="760" y2="42" stroke="#8aa4bf" stroke-width="1" stroke-dasharray="3 4" opacity=".45"/>
    <text x="40" y="30" fill="#8aa4bf" font-size="10" font-family="Inter, Segoe UI, sans-serif">SURFACE</text>
    <rect x="620" y="12" width="130" height="48" rx="5" fill="#0e2138" stroke="#22d3ee" stroke-width="1.2"/>
    <text x="685" y="32" fill="#e6eef8" font-size="11" text-anchor="middle" font-family="Inter, Segoe UI, sans-serif">VSD / control</text>
    <text x="685" y="48" fill="#8aa4bf" font-size="9" text-anchor="middle" font-family="Inter, Segoe UI, sans-serif">speed · voltage · protection</text>
    <rect x="175" y="34" width="145" height="18" rx="3" fill="#17324f" stroke="#2d7dd2" stroke-width="1"/>
    <text x="247" y="47" fill="#8aa4bf" font-size="9" text-anchor="middle" font-family="Inter, Segoe UI, sans-serif">WELLHEAD</text>
    <line x1="180" y1="52" x2="180" y2="322" stroke="#2d7dd2" stroke-width="2.5" opacity=".9"/>
    <line x1="315" y1="52" x2="315" y2="322" stroke="#2d7dd2" stroke-width="2.5" opacity=".9"/>
    <line x1="222" y1="52" x2="222" y2="190" stroke="#8aa4bf" stroke-width="1.4"/>
    <line x1="273" y1="52" x2="273" y2="190" stroke="#8aa4bf" stroke-width="1.4"/>
    <rect x="213" y="184" width="69" height="48" rx="4" fill="url(#metal)" stroke="#22d3ee" stroke-width="1.3"/>
    <line x1="213" y1="196" x2="282" y2="196" stroke="#22d3ee" opacity=".45"/>
    <line x1="213" y1="208" x2="282" y2="208" stroke="#22d3ee" opacity=".45"/>
    <line x1="213" y1="220" x2="282" y2="220" stroke="#22d3ee" opacity=".45"/>
    <rect x="220" y="232" width="55" height="20" rx="3" fill="#0e2138" stroke="#22c55e" stroke-width="1.2"/>
    <text x="247" y="245" fill="#22c55e" font-size="8" text-anchor="middle" font-family="Inter, Segoe UI, sans-serif">INTAKE</text>
    <rect x="220" y="252" width="55" height="18" rx="3" fill="#0e2138" stroke="#f59e0b" stroke-width="1.1"/>
    <text x="247" y="264" fill="#f59e0b" font-size="8" text-anchor="middle" font-family="Inter, Segoe UI, sans-serif">SEAL</text>
    <rect x="213" y="270" width="69" height="42" rx="4" fill="#0e2138" stroke="#ef4444" stroke-width="1.3"/>
    <text x="247" y="288" fill="#ef4444" font-size="9" text-anchor="middle" font-family="Inter, Segoe UI, sans-serif">MOTOR</text>
    <text x="247" y="302" fill="#8aa4bf" font-size="8" text-anchor="middle" font-family="Inter, Segoe UI, sans-serif">windings · bearings</text>
    <path d="M620 36 H330 V278 H285" fill="none" stroke="#f59e0b" stroke-width="1.6" stroke-dasharray="4 3"/>
    <text x="475" y="28" fill="#f59e0b" font-size="9" text-anchor="middle" font-family="Inter, Segoe UI, sans-serif">power cable</text>
    <path d="M247 184 V60" stroke="url(#fluid)" stroke-width="7" opacity=".6"/>
    <path d="M247 120 L241 130 L253 130 Z" fill="#22d3ee" opacity=".7"/>
    <text x="345" y="150" fill="#8aa4bf" font-size="10" font-family="Inter, Segoe UI, sans-serif">Produced fluid</text>
    <text x="345" y="164" fill="#22d3ee" font-size="10" font-family="Inter, Segoe UI, sans-serif">to surface</text>
    <line x1="315" y1="198" x2="345" y2="198" stroke="#8aa4bf" stroke-width=".8"/>
    <text x="352" y="196" fill="#e6eef8" font-size="10" font-family="Inter, Segoe UI, sans-serif">Pump stages</text>
    <text x="352" y="209" fill="#8aa4bf" font-size="9" font-family="Inter, Segoe UI, sans-serif">head generation · wear</text>
    <line x1="275" y1="242" x2="345" y2="242" stroke="#8aa4bf" stroke-width=".8"/>
    <text x="352" y="240" fill="#22c55e" font-size="10" font-family="Inter, Segoe UI, sans-serif">Intake / gas separator</text>
    <text x="352" y="253" fill="#8aa4bf" font-size="9" font-family="Inter, Segoe UI, sans-serif">suction pressure · submergence</text>
    <line x1="275" y1="291" x2="345" y2="291" stroke="#8aa4bf" stroke-width=".8"/>
    <text x="352" y="289" fill="#ef4444" font-size="10" font-family="Inter, Segoe UI, sans-serif">Motor + seal</text>
    <text x="352" y="302" fill="#8aa4bf" font-size="9" font-family="Inter, Segoe UI, sans-serif">load · temperature · insulation</text>
    <line x1="180" y1="320" x2="315" y2="320" stroke="#8aa4bf" stroke-width="1" stroke-dasharray="2 4"/>
    <text x="500" y="330" fill="#8aa4bf" font-size="9" font-family="Inter, Segoe UI, sans-serif">Casing / annulus</text>
    <text x="700" y="330" fill="#8aa4bf" font-size="9" font-family="Inter, Segoe UI, sans-serif">Reservoir inflow → pump intake → tubing → surface</text>
  </svg>
  <div class="schematic-legend">Simplified educational schematic — not a completion design drawing.</div>
</div>
"""
