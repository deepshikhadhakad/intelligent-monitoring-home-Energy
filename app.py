"""
HomeEnergy - Modern Energy Analytics Dashboard (single-file edition)

Run it with:   streamlit run app.py

This one file contains everything: the calculation logic, CSV data handling,
Plotly charts and the redesigned dark dashboard. The calculation formulas, the
validation rules and the CSV file formats are unchanged from the earlier
multi-file version, so your saved data in data/*.csv keeps working.

Every number shown is an ESTIMATE worked out from rated power.
"""

import html
import io
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="HomeEnergy | Energy Analytics", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# 1. THEME CONSTANTS
# =============================================================================

BG = "#0B1220"
SIDEBAR_BG = "#111827"
CARD = "#182235"
CARD_2 = "#1E293B"
BLUE = "#3B82F6"
ELECTRIC = "#60A5FA"
GREEN = "#10B981"
ORANGE = "#F59E0B"
RED = "#EF4444"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
BORDER = "#263449"

LEVEL_COLORS = {"ok": GREEN, "warning": ORANGE, "exceeded": RED, "none": MUTED}

THEME_TOML = """[theme]
base = "dark"
primaryColor = "#3B82F6"
backgroundColor = "#0B1220"
secondaryBackgroundColor = "#182235"
textColor = "#F8FAFC"
"""

# =============================================================================
# 2. CALCULATIONS (same formulas as before, nothing rounded here)
# =============================================================================

MONTHS_PER_YEAR = 12
HIGH_SHARE_PCT = 20.0  # an appliance at or above this share is flagged as "high"

STATUS_NORMAL = "Normal"
STATUS_WARNING = "Budget warning"
STATUS_EXCEEDED = "Budget exceeded"
STATUS_NOT_SET = "Budget not set"
STATUS_TO_LEVEL = {STATUS_NORMAL: "ok", STATUS_WARNING: "warning",
                   STATUS_EXCEEDED: "exceeded", STATUS_NOT_SET: "none"}
LEVEL_LABELS = {"ok": "Normal", "warning": "Budget warning",
                "exceeded": "Budget exceeded", "none": "Budget not set"}

NUMBER_COLUMNS = ["Power_W", "Quantity", "Hours_Per_Day", "Days_Per_Month"]
IDEA_COLUMNS = ["Appliance", "Current_Hours", "Reduce_By_Hours",
                "kWh_Saved", "Monthly_Saving", "Annual_Saving"]

# Simple, hedged tips. They only talk about running time, which is what this app can model.
CATEGORY_TIPS = {
    "Air Conditioning": "Air conditioners usually have a high rated power, so even a small cut in daily running hours can show up clearly in the estimate.",
    "Water Heating": "Heat water only when you need it and switch the heater off afterwards. Fewer running hours means a lower estimate.",
    "Lighting": "Switching lights off in empty rooms reduces their operating hours.",
    "Fan": "Turn fans off in rooms nobody is using.",
    "Entertainment": "Switch the TV off, rather than leaving it running, when nobody is watching.",
    "Refrigeration": "A fridge cycles on and off, so entering 24 hours at full rated power may overestimate it. Consider using its real running hours.",
    "Water Pump": "Run the pump only as long as the tank actually needs to fill.",
    "Laundry": "Running full loads where practical reduces the number of wash days per month.",
    "Computing": "Put computers to sleep or switch them off when you're away from them.",
}


def daily_energy_kwh(power_w, quantity, hours_per_day):
    """Daily energy (kWh) = (W x quantity x hours per day) / 1000."""
    return (power_w * quantity * hours_per_day) / 1000


def monthly_energy_kwh(power_w, quantity, hours_per_day, days_per_month):
    """Monthly energy (kWh) = (W x quantity x hours per day x days per month) / 1000."""
    return (power_w * quantity * hours_per_day * days_per_month) / 1000


def energy_cost(monthly_kwh, tariff_per_kwh):
    """Energy cost = monthly kWh x tariff per kWh."""
    return monthly_kwh * tariff_per_kwh


def total_bill(energy_cost_value, fixed_charge=0.0):
    """Estimated total bill = energy cost + fixed monthly charges."""
    return energy_cost_value + fixed_charge


def annual_value(monthly_value):
    """Assumes every month looks like this one."""
    return monthly_value * MONTHS_PER_YEAR


def add_consumption_columns(appliances, tariff_per_kwh):
    """Takes the appliance table and returns a copy with all the calculated columns added."""
    df = appliances.copy()
    for col in NUMBER_COLUMNS:  # a bad cell can't crash anything
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["Daily_kWh"] = daily_energy_kwh(df["Power_W"], df["Quantity"], df["Hours_Per_Day"])
    df["Monthly_kWh"] = monthly_energy_kwh(
        df["Power_W"], df["Quantity"], df["Hours_Per_Day"], df["Days_Per_Month"])
    df["Annual_kWh"] = annual_value(df["Monthly_kWh"])

    total = df["Monthly_kWh"].sum()
    df["Share_Pct"] = (df["Monthly_kWh"] / total * 100) if total > 0 else 0.0  # no divide by zero
    df["Monthly_Cost"] = energy_cost(df["Monthly_kWh"], tariff_per_kwh)
    return df


def rank_appliances(analysis):
    """Highest monthly consumption first, with a Rank column."""
    ranked = analysis.sort_values("Monthly_kWh", ascending=False, kind="mergesort")
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "Rank", list(range(1, len(ranked) + 1)))
    return ranked


def top_n(analysis, n=5):
    return rank_appliances(analysis).head(n)


def high_consumers(analysis, share_threshold=HIGH_SHARE_PCT):
    """Appliances that make up a big slice of the total."""
    if analysis.empty:
        return analysis
    return analysis[analysis["Share_Pct"] >= share_threshold]


def budget_status(bill, budget, warning_threshold=80.0):
    """
    remaining   = budget - bill
    utilization = (bill / budget) x 100
    A budget of zero (or less) is treated as "not set", so nothing is divided.
    """
    budget = 0.0 if budget is None else float(budget)

    if not budget > 0:  # also catches NaN
        return {"budget": 0.0, "bill": float(bill), "remaining": 0.0,
                "utilization_pct": 0.0, "exceeded_by": 0.0,
                "status": STATUS_NOT_SET, "level": "none", "is_set": False}

    remaining = budget - bill
    utilization = (bill * 100) / budget
    exceeded_by = max(bill - budget, 0.0)

    # tiny tolerance so exactly 80% or 100% isn't missed by float noise
    if utilization >= 100 - 1e-9:
        status, level = STATUS_EXCEEDED, "exceeded"
    elif utilization >= warning_threshold - 1e-9:
        status, level = STATUS_WARNING, "warning"
    else:
        status, level = STATUS_NORMAL, "ok"

    return {"budget": budget, "bill": float(bill), "remaining": remaining,
            "utilization_pct": utilization, "exceeded_by": exceeded_by,
            "status": status, "level": level, "is_set": True}


def energy_to_cut_kwh(exceeded_by, tariff_per_kwh):
    """How many kWh would need to go to remove the overshoot (0 if tariff is 0)."""
    return exceeded_by / tariff_per_kwh if tariff_per_kwh > 0 else 0.0


def summarize(analysis, tariff_per_kwh, fixed_charge, budget, warning_threshold=80.0):
    """One dictionary with every headline number. All pages read from this."""
    monthly_kwh = float(analysis["Monthly_kWh"].sum())
    daily_kwh = float(analysis["Daily_kWh"].sum())
    cost = energy_cost(monthly_kwh, tariff_per_kwh)
    bill = total_bill(cost, fixed_charge)

    top_name, top_kwh = "—", 0.0
    if not analysis.empty and monthly_kwh > 0:
        top_row = analysis.loc[analysis["Monthly_kWh"].idxmax()]
        top_name, top_kwh = str(top_row["Name"]), float(top_row["Monthly_kWh"])

    return {
        "appliance_count": int(len(analysis)),
        "total_daily_kwh": daily_kwh,
        "total_monthly_kwh": monthly_kwh,
        "total_annual_kwh": annual_value(monthly_kwh),
        "energy_cost": cost,
        "fixed_charge": float(fixed_charge),
        "total_bill": bill,
        "annual_cost": annual_value(bill),
        "top_appliance": top_name,
        "top_appliance_kwh": top_kwh,
        "budget": budget_status(bill, budget, warning_threshold),
    }


def savings_for_change(power_w, quantity, old_hours, new_hours, days, tariff_per_kwh):
    """Savings for ONE appliance when its daily hours change."""
    old_kwh = monthly_energy_kwh(power_w, quantity, old_hours, days)
    new_kwh = monthly_energy_kwh(power_w, quantity, new_hours, days)
    saved = old_kwh - new_kwh
    monthly_saving = saved * tariff_per_kwh
    return {"original_kwh": old_kwh, "new_kwh": new_kwh, "saved_kwh": saved,
            "monthly_saving": monthly_saving, "annual_saving": annual_value(monthly_saving)}


def simulate_scenario(analysis, appliance_id, new_hours, tariff_per_kwh,
                      fixed_charge, budget, warning_threshold=80.0):
    """What-if for one appliance. It only reads the table and never changes it."""
    match = analysis[analysis["ID"] == appliance_id]
    if match.empty:
        raise ValueError("Appliance not found for the simulation.")
    row = match.iloc[0]

    change = savings_for_change(row["Power_W"], row["Quantity"], row["Hours_Per_Day"],
                                new_hours, row["Days_Per_Month"], tariff_per_kwh)

    original_total_kwh = float(analysis["Monthly_kWh"].sum())
    new_total_kwh = max(original_total_kwh - change["saved_kwh"], 0.0)

    # same bill formula as everywhere else, so alerts and savings stay consistent
    original_bill = total_bill(energy_cost(original_total_kwh, tariff_per_kwh), fixed_charge)
    new_bill = total_bill(energy_cost(new_total_kwh, tariff_per_kwh), fixed_charge)

    result = dict(change)
    result.update({
        "original_hours": float(row["Hours_Per_Day"]),
        "new_hours": float(new_hours),
        "original_total_kwh": original_total_kwh,
        "new_total_kwh": new_total_kwh,
        "original_bill": original_bill,
        "new_bill": new_bill,
        "original_budget": budget_status(original_bill, budget, warning_threshold),
        "new_budget": budget_status(new_bill, budget, warning_threshold),
    })
    return result


def saving_ideas(analysis, tariff_per_kwh, how_many=3):
    """For the top consumers, cut daily use by 1 hour (or 25% of current hours if smaller)."""
    rows = []
    for _, r in rank_appliances(analysis).head(how_many).iterrows():
        if r["Monthly_kWh"] <= 0:
            continue
        cut = min(1.0, r["Hours_Per_Day"] * 0.25)
        change = savings_for_change(r["Power_W"], r["Quantity"], r["Hours_Per_Day"],
                                    r["Hours_Per_Day"] - cut, r["Days_Per_Month"], tariff_per_kwh)
        rows.append({
            "Appliance": r["Name"],
            "Current_Hours": r["Hours_Per_Day"],
            "Reduce_By_Hours": cut,
            "kWh_Saved": change["saved_kwh"],
            "Monthly_Saving": change["monthly_saving"],
            "Annual_Saving": change["annual_saving"],
        })
    return pd.DataFrame(rows, columns=IDEA_COLUMNS)


def general_tips(analysis):
    """Plain rule-based tips. No claims about how efficient any appliance is."""
    if analysis.empty or analysis["Monthly_kWh"].sum() <= 0:
        return []

    tips = []
    for r in high_consumers(analysis).itertuples():
        tips.append(f"**{r.Name}** makes up about {r.Share_Pct:.1f}% of your estimated "
                    f"consumption, so changes here matter the most.")

    used_categories = analysis.loc[analysis["Monthly_kWh"] > 0, "Category"].unique()
    for category in used_categories:
        if category in CATEGORY_TIPS:
            tips.append(CATEGORY_TIPS[category])

    tips.append("For better accuracy, compare the watts entered here with the appliance "
                "label, or measure real usage with a plug-in energy meter.")
    return tips


def format_month(year_month):
    """'2026-09' -> 'Sep 2026'. Falls back to the original text if it can't be parsed."""
    try:
        return datetime.strptime(str(year_month), "%Y-%m").strftime("%b %Y")
    except ValueError:
        return str(year_month)


def build_history_record(month, summary, notes=""):
    """Turns the current summary into one row for the history CSV."""
    b = summary["budget"]
    return {
        "Month": month,
        "Total_kWh": summary["total_monthly_kwh"],
        "Energy_Cost": summary["energy_cost"],
        "Fixed_Charges": summary["fixed_charge"],
        "Total_Bill": summary["total_bill"],
        "Budget": b["budget"],
        "Utilization_Pct": b["utilization_pct"],
        "Alert_Status": b["status"],
        "Notes": notes,
    }


def add_month_over_month(history):
    """Adds month-over-month % change for consumption and bill, plus a readable month label."""
    df = history.copy().sort_values("Month").reset_index(drop=True)
    for source, target in (("Total_kWh", "kWh_Change_Pct"), ("Total_Bill", "Bill_Change_Pct")):
        previous = df[source].shift(1)
        safe_previous = previous.where(previous > 0)  # NaN where we can't divide
        df[target] = (df[source] - previous) / safe_previous * 100
    df["Month_Label"] = df["Month"].map(format_month)
    return df


def history_stats(history):
    if history is None or history.empty:
        return {"has_data": False, "months_saved": 0, "highest_month": None,
                "highest_kwh": 0.0, "average_kwh": 0.0}
    peak = history["Total_kWh"].idxmax()
    return {"has_data": True,
            "months_saved": int(len(history)),
            "highest_month": history.loc[peak, "Month"],
            "highest_kwh": float(history.loc[peak, "Total_kWh"]),
            "average_kwh": float(history["Total_kWh"].mean())}


# =============================================================================
# 3. DATA MANAGER (CSV loading, saving, validation, import and export)
#    Files live in data/ next to this script:
#    appliances.csv, history.csv, settings.csv (created automatically)
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
APPLIANCE_FILE = DATA_DIR / "appliances.csv"
HISTORY_FILE = DATA_DIR / "history.csv"
SETTINGS_FILE = DATA_DIR / "settings.csv"

CATEGORIES = ["Fan", "Lighting", "Refrigeration", "Entertainment", "Air Conditioning",
              "Laundry", "Water Heating", "Water Pump", "Computing", "Kitchen", "Other"]

APPLIANCE_COLUMNS = ["ID", "Name", "Category", "Power_W", "Quantity",
                     "Hours_Per_Day", "Days_Per_Month"]
IMPORT_COLUMNS = APPLIANCE_COLUMNS[1:]  # a CSV doesn't need to carry IDs
NUMERIC_APPLIANCE_COLUMNS = ["Power_W", "Quantity", "Hours_Per_Day", "Days_Per_Month"]

HISTORY_COLUMNS = ["Month", "Total_kWh", "Energy_Cost", "Fixed_Charges", "Total_Bill",
                   "Budget", "Utilization_Pct", "Alert_Status", "Notes"]
TEXT_HISTORY_COLUMNS = ["Month", "Alert_Status", "Notes"]
MONTH_PATTERN = r"^\d{4}-(?:0[1-9]|1[0-2])$"

MAX_NAME_LENGTH = 60
MAX_POWER_W = 20000
MAX_QUANTITY = 1000
MAX_IMPORT_ROWS = 500

DEFAULT_SETTINGS = {
    "tariff_per_kwh": 6.0,        # placeholder only; users should enter their own rate
    "fixed_charge": 50.0,
    "monthly_budget": 2500.0,
    "warning_threshold_pct": 80.0,
}
SETTING_LIMITS = {
    "tariff_per_kwh": (0, 1000, "Tariff"),
    "fixed_charge": (0, 100000, "Fixed charge"),
    "monthly_budget": (0, 10000000, "Monthly budget"),
    "warning_threshold_pct": (1, 99, "Warning threshold"),
}

# Typical rated values just to save typing. Always check your own appliance label.
COMMON_APPLIANCES = {
    "Ceiling Fan": {"category": "Fan", "power": 75, "quantity": 1, "hours": 8, "days": 30},
    "LED Bulb": {"category": "Lighting", "power": 9, "quantity": 1, "hours": 5, "days": 30},
    "LED Tube Light": {"category": "Lighting", "power": 18, "quantity": 1, "hours": 6, "days": 30},
    "Refrigerator": {"category": "Refrigeration", "power": 150, "quantity": 1, "hours": 8, "days": 30},
    "LED Television": {"category": "Entertainment", "power": 100, "quantity": 1, "hours": 4, "days": 30},
    "Air Conditioner (1.5 ton)": {"category": "Air Conditioning", "power": 1500, "quantity": 1, "hours": 6, "days": 30},
    "Washing Machine": {"category": "Laundry", "power": 500, "quantity": 1, "hours": 1, "days": 12},
    "Laptop": {"category": "Computing", "power": 65, "quantity": 1, "hours": 6, "days": 30},
    "Desktop Computer": {"category": "Computing", "power": 200, "quantity": 1, "hours": 4, "days": 30},
    "Water Heater (Geyser)": {"category": "Water Heating", "power": 2000, "quantity": 1, "hours": 0.5, "days": 30},
    "Water Pump": {"category": "Water Pump", "power": 750, "quantity": 1, "hours": 0.5, "days": 30},
    "Microwave Oven": {"category": "Kitchen", "power": 1200, "quantity": 1, "hours": 0.25, "days": 30},
    "Mixer Grinder": {"category": "Kitchen", "power": 500, "quantity": 1, "hours": 0.25, "days": 30},
}
CUSTOM_DEFAULTS = {"category": "Other", "power": 100, "quantity": 1, "hours": 1, "days": 30}

# Illustrative sample data, built into this file. NOT real measurements.
SAMPLE_APPLIANCES_CSV = """Name,Category,Power_W,Quantity,Hours_Per_Day,Days_Per_Month
Ceiling Fan,Fan,75,3,10,30
LED Tube Light,Lighting,18,4,6,30
LED Bulb,Lighting,9,5,5,30
Refrigerator,Refrigeration,150,1,8,30
LED Television,Entertainment,100,1,4,30
Air Conditioner (1.5 ton),Air Conditioning,1500,1,6,20
Washing Machine,Laundry,500,1,1,12
Laptop,Computing,65,1,6,30
Water Heater (Geyser),Water Heating,2000,1,0.5,30
Water Pump,Water Pump,750,1,0.5,30
"""

SAMPLE_HISTORY_CSV = """Month,Total_kWh,Energy_Cost,Fixed_Charges,Total_Bill,Budget,Utilization_Pct,Alert_Status,Notes
2026-04,392.0,2352.00,50.0,2402.00,2500.0,96.08,Budget warning,Illustrative sample data
2026-05,421.5,2529.00,50.0,2579.00,2500.0,103.16,Budget exceeded,Illustrative sample data
2026-06,398.2,2389.20,50.0,2439.20,2500.0,97.568,Budget warning,Illustrative sample data
2026-07,340.8,2044.80,50.0,2094.80,2500.0,83.792,Budget warning,Illustrative sample data
2026-08,318.4,1910.40,50.0,1960.40,2500.0,78.416,Normal,Illustrative sample data
2026-09,352.6,2115.60,50.0,2165.60,2500.0,86.624,Budget warning,Illustrative sample data
"""


def _to_number(value):
    """Returns a finite float, or None if the value isn't a usable number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _safe_text(text):
    """If a cell starts with = + - or @, Excel may treat it as a formula, so quote it."""
    text = str(text).strip()
    if text and text[0] in "=+-@":
        text = "'" + text
    return text


def to_csv_bytes(df, float_format="%.4f"):
    """DataFrame -> CSV bytes for download buttons."""
    return df.to_csv(index=False, float_format=float_format).encode("utf-8")


def ensure_data_files():
    """Creates the data folder and any missing working files. Never overwrites existing ones."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not APPLIANCE_FILE.exists():
        pd.DataFrame(columns=APPLIANCE_COLUMNS).to_csv(APPLIANCE_FILE, index=False)
    if not HISTORY_FILE.exists():
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(HISTORY_FILE, index=False)
    if not SETTINGS_FILE.exists():
        save_settings(DEFAULT_SETTINGS)


def read_sample_bytes():
    """The sample appliance CSV as bytes (used for the template download)."""
    return SAMPLE_APPLIANCES_CSV.encode("utf-8")


def _tidy_appliances(df):
    """Cleans whatever came out of the CSV so the rest of the app can trust it."""
    df = df.copy()
    for col in APPLIANCE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[APPLIANCE_COLUMNS]

    for col in NUMERIC_APPLIANCE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Name"] + NUMERIC_APPLIANCE_COLUMNS)  # malformed rows
    if df.empty:
        return pd.DataFrame(columns=APPLIANCE_COLUMNS)

    df["Name"] = df["Name"].astype(str).str.strip()
    df = df[df["Name"] != ""].reset_index(drop=True)

    lookup = {c.lower(): c for c in CATEGORIES}
    df["Category"] = df["Category"].astype(str).str.strip().str.lower().map(lookup).fillna("Other")

    for col in NUMERIC_APPLIANCE_COLUMNS:
        df[col] = df[col].clip(lower=0)
    df["Quantity"] = df["Quantity"].round().astype(int)

    ids = pd.to_numeric(df["ID"], errors="coerce")
    if ids.isna().any() or ids.duplicated().any():
        df["ID"] = list(range(1, len(df) + 1))  # IDs were broken, so hand out fresh ones
    else:
        df["ID"] = ids.astype(int)
    return df


def load_appliances():
    try:
        raw = pd.read_csv(APPLIANCE_FILE)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=APPLIANCE_COLUMNS)
    return _tidy_appliances(raw)


def save_appliances(df):
    df[APPLIANCE_COLUMNS].to_csv(APPLIANCE_FILE, index=False)


def next_id(df):
    return int(df["ID"].max()) + 1 if not df.empty else 1


def validate_appliance(name, category, power, quantity, hours, days,
                       existing=None, ignore_id=None):
    """Returns a list of error messages. An empty list means everything is fine."""
    errors = []

    name = "" if name is None or (isinstance(name, float) and np.isnan(name)) else str(name).strip()
    if not name:
        errors.append("Appliance name cannot be empty.")
    elif len(name) > MAX_NAME_LENGTH:
        errors.append(f"Appliance name must be {MAX_NAME_LENGTH} characters or fewer.")
    elif existing is not None and not existing.empty:
        others = existing if ignore_id is None else existing[existing["ID"] != ignore_id]
        if name.lower() in others["Name"].str.lower().tolist():
            errors.append(f'An appliance called "{name}" already exists. '
                          "Pick a different name or edit the existing one.")

    if category not in CATEGORIES:
        errors.append("Please choose a category from the list.")

    p = _to_number(power)
    if p is None or p <= 0 or p > MAX_POWER_W:
        errors.append(f"Rated power must be a number above 0 and up to {MAX_POWER_W:,} W.")

    q = _to_number(quantity)
    if q is None or q < 1 or q > MAX_QUANTITY or q != int(q):
        errors.append(f"Quantity must be a whole number from 1 to {MAX_QUANTITY:,}.")

    h = _to_number(hours)
    if h is None or h < 0 or h > 24:
        errors.append("Operating hours per day must be between 0 and 24.")

    d = _to_number(days)
    if d is None or d < 1 or d > 31 or d != int(d):
        errors.append("Usage days per month must be a whole number from 1 to 31.")

    return errors


def add_appliance(name, category, power, quantity, hours, days):
    """Returns a list of errors (empty = saved)."""
    df = load_appliances()
    errors = validate_appliance(name, category, power, quantity, hours, days, existing=df)
    if errors:
        return errors

    new_row = pd.DataFrame([{
        "ID": next_id(df), "Name": _safe_text(name), "Category": category,
        "Power_W": float(power), "Quantity": int(quantity),
        "Hours_Per_Day": float(hours), "Days_Per_Month": int(days),
    }])
    df = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)
    save_appliances(df)
    return []


def update_appliance(appliance_id, name, category, power, quantity, hours, days):
    df = load_appliances()
    if appliance_id not in df["ID"].tolist():
        return ["That appliance no longer exists. Please refresh and try again."]

    errors = validate_appliance(name, category, power, quantity, hours, days,
                                existing=df, ignore_id=appliance_id)
    if errors:
        return errors

    mask = df["ID"] == appliance_id
    new_values = {"Name": _safe_text(name), "Category": category, "Power_W": float(power),
                  "Quantity": int(quantity), "Hours_Per_Day": float(hours),
                  "Days_Per_Month": int(days)}
    for column, value in new_values.items():
        df.loc[mask, column] = value
    save_appliances(df)
    return []


def delete_appliance(appliance_id):
    df = load_appliances()
    if appliance_id not in df["ID"].tolist():
        return False
    save_appliances(df[df["ID"] != appliance_id])
    return True


def parse_appliance_csv(file_or_path):
    """
    Reads and checks an appliance CSV. Returns (clean_df, errors, warnings).
    If there are any errors, clean_df is None, so nothing half-valid gets imported.
    """
    errors, warnings = [], []

    try:
        raw = pd.read_csv(file_or_path)
    except pd.errors.EmptyDataError:
        return None, ["The file is empty."], warnings
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
        return None, ["This file couldn't be read as a CSV. Please check the format."], warnings

    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in IMPORT_COLUMNS if c not in raw.columns]
    if missing:
        return None, [f"Missing column(s): {', '.join(missing)}. "
                      f"Expected columns: {', '.join(IMPORT_COLUMNS)}."], warnings
    if raw.empty:
        return None, ["The file has headers but no appliance rows."], warnings
    if len(raw) > MAX_IMPORT_ROWS:
        return None, [f"Too many rows. The limit is {MAX_IMPORT_ROWS} appliances per file."], warnings

    category_lookup = {c.lower(): c for c in CATEGORIES}
    rows, seen_names = [], set()

    for position, row in raw.iterrows():
        line = position + 2  # +1 for the header row, +1 because people count from 1
        name = "" if pd.isna(row["Name"]) else str(row["Name"]).strip()
        raw_category = "" if pd.isna(row["Category"]) else str(row["Category"]).strip()

        category = category_lookup.get(raw_category.lower())
        if category is None:
            warnings.append(f'Row {line}: category "{raw_category}" isn\'t recognised, so "Other" was used.')
            category = "Other"

        row_errors = validate_appliance(name, category, row["Power_W"], row["Quantity"],
                                        row["Hours_Per_Day"], row["Days_Per_Month"])
        if row_errors:
            errors.extend(f"Row {line}: {message}" for message in row_errors)
            continue

        if name.lower() in seen_names:
            warnings.append(f'Row {line}: "{name}" appears more than once in the file, so the repeat was skipped.')
            continue
        seen_names.add(name.lower())

        rows.append({"Name": _safe_text(name), "Category": category,
                     "Power_W": float(row["Power_W"]), "Quantity": int(float(row["Quantity"])),
                     "Hours_Per_Day": float(row["Hours_Per_Day"]),
                     "Days_Per_Month": int(float(row["Days_Per_Month"]))})

    if errors:
        return None, errors, warnings
    if not rows:
        return None, ["No usable appliance rows were found."], warnings
    return pd.DataFrame(rows, columns=IMPORT_COLUMNS), [], warnings


def import_appliances(new_rows, replace_all=False):
    """Saves validated rows. By default they're added; existing names are skipped. Returns (added, skipped)."""
    if replace_all:
        base = pd.DataFrame(columns=APPLIANCE_COLUMNS)
        to_add, skipped = new_rows.copy(), 0
    else:
        base = load_appliances()
        existing_names = set(base["Name"].str.lower())
        keep = ~new_rows["Name"].str.lower().isin(existing_names)
        to_add, skipped = new_rows[keep].copy(), int((~keep).sum())

    start = next_id(base)
    to_add.insert(0, "ID", list(range(start, start + len(to_add))))
    final = to_add if base.empty else pd.concat([base, to_add], ignore_index=True)
    save_appliances(final)
    return len(to_add), skipped


def add_sample_appliances():
    """Adds the illustrative sample list. Returns (added, skipped, error_message_or_None)."""
    clean, errors, _ = parse_appliance_csv(io.StringIO(SAMPLE_APPLIANCES_CSV))
    if clean is None:
        return 0, 0, "; ".join(errors)
    added, skipped = import_appliances(clean, replace_all=False)
    return added, skipped, None


def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    try:
        df = pd.read_csv(SETTINGS_FILE)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return settings

    for _, row in df.iterrows():
        key = str(row.get("Setting", "")).strip()
        value = _to_number(row.get("Value"))
        if key in SETTING_LIMITS and value is not None:
            low, high, _ = SETTING_LIMITS[key]
            if low <= value <= high:
                settings[key] = value
    return settings


def validate_settings(settings):
    errors = []
    for key, (low, high, label) in SETTING_LIMITS.items():
        value = _to_number(settings.get(key))
        if value is None or value < low or value > high:
            errors.append(f"{label} must be a number between {low:,} and {high:,}.")
    return errors


def save_settings(settings):
    pd.DataFrame({"Setting": list(settings.keys()),
                  "Value": list(settings.values())}).to_csv(SETTINGS_FILE, index=False)


def update_settings(changes):
    """Changes some settings, keeps the rest. Returns a list of errors (empty = saved)."""
    settings = load_settings()
    settings.update(changes)
    errors = validate_settings(settings)
    if errors:
        return errors
    save_settings(settings)
    return []


def _tidy_history(df):
    df = df.copy()
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in TEXT_HISTORY_COLUMNS else 0.0
    df = df[HISTORY_COLUMNS]

    df["Month"] = df["Month"].astype(str).str.strip()
    df = df[df["Month"].str.match(MONTH_PATTERN)].copy()  # drops rows with a bad month

    for col in HISTORY_COLUMNS:
        if col in TEXT_HISTORY_COLUMNS:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.drop_duplicates(subset="Month", keep="last")
    return df.sort_values("Month").reset_index(drop=True)


def load_history():
    try:
        raw = pd.read_csv(HISTORY_FILE, dtype={"Month": str})
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return _tidy_history(raw)


def save_history_record(record, replace=False):
    """Returns 'added', 'replaced', or 'exists' (nothing saved because replace wasn't confirmed)."""
    record = dict(record)
    record["Notes"] = _safe_text(record.get("Notes", ""))[:200]

    history = load_history()
    exists = record["Month"] in history["Month"].tolist()
    if exists and not replace:
        return "exists"
    if exists:
        history = history[history["Month"] != record["Month"]]

    new_row = pd.DataFrame([record])[HISTORY_COLUMNS]
    history = new_row if history.empty else pd.concat([history, new_row], ignore_index=True)
    _tidy_history(history).to_csv(HISTORY_FILE, index=False)
    return "replaced" if exists else "added"


def delete_history_record(month):
    history = load_history()
    if month not in history["Month"].tolist():
        return False
    history[history["Month"] != month].to_csv(HISTORY_FILE, index=False)
    return True


def add_sample_history():
    """Adds sample months that aren't already saved. Returns (added, skipped, error_or_None)."""
    try:
        sample = _tidy_history(pd.read_csv(io.StringIO(SAMPLE_HISTORY_CSV), dtype={"Month": str}))
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return 0, 0, "The sample history couldn't be read."

    history = load_history()
    keep = ~sample["Month"].isin(history["Month"])
    to_add = sample[keep]
    if not to_add.empty:
        combined = to_add if history.empty else pd.concat([history, to_add], ignore_index=True)
        _tidy_history(combined).to_csv(HISTORY_FILE, index=False)
    return int(keep.sum()), int((~keep).sum()), None


def last_updated_text():
    """Newest modification time of the three data files."""
    times = [p.stat().st_mtime for p in (APPLIANCE_FILE, HISTORY_FILE, SETTINGS_FILE) if p.exists()]
    if not times:
        return "No data saved yet"
    return "Updated " + datetime.fromtimestamp(max(times)).strftime("%d %b %Y, %H:%M")


def ensure_dark_theme_file():
    """
    Streamlit's own widgets (dropdown menus, sliders) follow .streamlit/config.toml.
    If it is missing, or still holds the old light theme, write the dark one.
    Returns True when the file was just written (Streamlit must be restarted to load it).
    """
    config_file = BASE_DIR / ".streamlit" / "config.toml"
    try:
        if config_file.exists() and 'base = "light"' not in config_file.read_text(encoding="utf-8"):
            return False
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(THEME_TOML, encoding="utf-8")
        return True
    except OSError:
        return False


# =============================================================================
# 4. STYLING (dark theme CSS) AND SMALL HTML BUILDING BLOCKS
# =============================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, button, input, textarea, select { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.stApp { background: #0B1220; color: #F8FAFC; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, .stDeployButton { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1320px; }
h1, h2, h3, h4, h5 { color: #F8FAFC; letter-spacing: -0.01em; }
p, li, span { color: inherit; }
hr { border-color: #263449; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #94A3B8; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #263449; }
@media (min-width: 992px) {
  [data-testid="stSidebar"][aria-expanded="true"] { min-width: 272px; max-width: 272px; }
}
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
.he-brand { display: flex; align-items: center; gap: 12px; padding: 4px 6px 18px; border-bottom: 1px solid #263449; margin-bottom: 14px; }
.he-brand-icon { width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, #3B82F6, #60A5FA); color: #0B1220; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(59,130,246,.35); }
.he-brand-name { font-size: 1.15rem; font-weight: 700; color: #F8FAFC; line-height: 1.1; }
.he-brand-sub { font-size: .75rem; color: #94A3B8; margin-top: 3px; }
.he-side-note { font-size: .74rem; color: #94A3B8; line-height: 1.5; padding: 14px 8px 0; border-top: 1px solid #263449; margin-top: 18px; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] [role="radiogroup"] > label { padding: 10px 14px; border-radius: 10px; border-left: 3px solid transparent; width: 100%; margin: 0; cursor: pointer; transition: background .15s, border-color .15s; }
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: #1E293B; }
[data-testid="stSidebar"] [role="radiogroup"] > label p { color: #94A3B8; font-size: .92rem; font-weight: 500; }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) { background: rgba(59,130,246,.16); border-left-color: #3B82F6; }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p { color: #60A5FA; font-weight: 600; }

/* ---------- page header and pills ---------- */
.he-title { font-size: 1.8rem; font-weight: 700; color: #F8FAFC; line-height: 1.2; }
.he-subtitle { color: #94A3B8; font-size: .95rem; margin-top: 4px; }
.he-pills { display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap; padding-top: 6px; }
.he-pill { display: inline-flex; align-items: center; gap: 8px; background: #182235; border: 1px solid #263449; border-radius: 999px; padding: 7px 14px; color: #94A3B8; font-size: .8rem; white-space: nowrap; }
.he-section { margin: 30px 0 12px; }
.he-section-title { font-size: 1.1rem; font-weight: 700; color: #F8FAFC; }
.he-section-sub { font-size: .84rem; color: #94A3B8; margin-top: 2px; }
.he-card-title { font-size: 1rem; font-weight: 600; color: #F8FAFC; }
.he-card-sub { font-size: .8rem; color: #94A3B8; margin: 2px 0 10px; }
.he-group-label { font-size: .85rem; font-weight: 600; color: #60A5FA; margin: 14px 0 4px; }

/* ---------- cards ---------- */
.he-card, [class*="st-key-card_"] { background: #182235; border: 1px solid #263449; border-radius: 16px; padding: 20px 22px; box-shadow: 0 4px 18px rgba(0,0,0,.28); transition: border-color .2s; }
.he-card:hover, [class*="st-key-card_"]:hover { border-color: #35496a; }
[class*="st-key-row_"] { background: #182235; border: 1px solid #263449; border-radius: 14px; padding: 12px 18px; transition: border-color .2s; }
[class*="st-key-row_"]:hover { border-color: #3B82F6; }
[data-testid="stForm"] { border: none; padding: 0; background: transparent; }
.he-metric { background: #182235; border: 1px solid #263449; border-radius: 16px; padding: 18px 20px; box-shadow: 0 4px 18px rgba(0,0,0,.28); min-height: 160px; transition: transform .2s, border-color .2s; }
.he-metric:hover { transform: translateY(-2px); border-color: #3B82F6; }
.he-metric-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.he-metric-label { color: #94A3B8; font-size: .85rem; font-weight: 500; }
.he-icon { width: 36px; height: 36px; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; }
.he-icon-blue { background: rgba(59,130,246,.16); color: #60A5FA; }
.he-icon-green { background: rgba(16,185,129,.16); color: #10B981; }
.he-icon-orange { background: rgba(245,158,11,.16); color: #F59E0B; }
.he-icon-red { background: rgba(239,68,68,.16); color: #EF4444; }
.he-metric-value { font-size: 1.85rem; font-weight: 700; line-height: 1.15; color: #F8FAFC; }
.he-metric-value small { font-size: .9rem; font-weight: 500; color: #94A3B8; margin-left: 4px; }
.he-metric-sub { color: #94A3B8; font-size: .8rem; margin-top: 8px; line-height: 1.4; }
.he-chip-delta { display: inline-flex; align-items: center; gap: 4px; font-size: .75rem; font-weight: 600; margin-top: 8px; padding: 2px 8px; border-radius: 999px; }
.he-delta-up { background: rgba(245,158,11,.14); color: #F59E0B; }
.he-delta-down { background: rgba(16,185,129,.14); color: #10B981; }
.he-delta-flat { background: rgba(148,163,184,.14); color: #94A3B8; }

/* ---------- progress, badges, notices ---------- */
.he-progress { height: 10px; background: #263449; border-radius: 999px; overflow: hidden; margin: 10px 0 6px; }
.he-progress > span { display: block; height: 100%; border-radius: 999px; transition: width .4s ease; }
.he-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px; font-size: .7rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; border: 1px solid transparent; white-space: nowrap; }
.he-badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.he-badge-ok { background: rgba(16,185,129,.12); color: #10B981; border-color: rgba(16,185,129,.35); }
.he-badge-warning { background: rgba(245,158,11,.12); color: #F59E0B; border-color: rgba(245,158,11,.35); }
.he-badge-exceeded { background: rgba(239,68,68,.12); color: #EF4444; border-color: rgba(239,68,68,.35); }
.he-badge-none { background: rgba(148,163,184,.12); color: #94A3B8; border-color: rgba(148,163,184,.3); }
.he-notice { display: flex; gap: 12px; align-items: flex-start; padding: 14px 16px; border-radius: 12px; border: 1px solid; margin: 10px 0; font-size: .9rem; line-height: 1.5; }
.he-notice-title { font-weight: 700; margin-bottom: 2px; }
.he-notice-body { color: #CBD5E1; }
.he-n-success { background: rgba(16,185,129,.1); border-color: rgba(16,185,129,.35); color: #10B981; }
.he-n-warning { background: rgba(245,158,11,.1); border-color: rgba(245,158,11,.35); color: #F59E0B; }
.he-n-error { background: rgba(239,68,68,.1); border-color: rgba(239,68,68,.35); color: #EF4444; }
.he-n-info { background: rgba(59,130,246,.1); border-color: rgba(59,130,246,.35); color: #60A5FA; }

/* ---------- budget card, receipt rows, insights ---------- */
.he-card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; gap: 10px; }
.he-big { font-size: 2.3rem; font-weight: 700; line-height: 1.1; color: #F8FAFC; }
.he-big-sub { color: #94A3B8; font-size: .82rem; }
.he-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #263449; font-size: .92rem; gap: 12px; }
.he-row:last-child { border-bottom: none; }
.he-row > span:first-child { color: #94A3B8; }
.he-row > span:last-child { font-weight: 600; color: #F8FAFC; text-align: right; }
.he-row-total { font-size: 1.05rem; }
.he-row-total > span:last-child { font-size: 1.35rem; color: #60A5FA; }
.he-red { color: #EF4444 !important; }
.he-green { color: #10B981 !important; }
.he-orange { color: #F59E0B !important; }
.he-insight { background: #182235; border: 1px solid #263449; border-radius: 16px; padding: 18px 20px; height: 100%; min-height: 150px; transition: border-color .2s; }
.he-insight:hover { border-color: #3B82F6; }
.he-insight-label { color: #94A3B8; font-size: .82rem; font-weight: 500; display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.he-insight-main { font-size: 1.2rem; font-weight: 700; color: #F8FAFC; line-height: 1.3; }
.he-insight-text { color: #94A3B8; font-size: .84rem; margin-top: 6px; line-height: 1.5; }
.he-tip { display: flex; gap: 12px; align-items: flex-start; background: #182235; border: 1px solid #263449; border-radius: 12px; padding: 12px 16px; margin-bottom: 8px; color: #CBD5E1; font-size: .88rem; line-height: 1.5; }
.he-tip strong { color: #F8FAFC; }
.he-tip .he-tip-icon { color: #F59E0B; margin-top: 2px; flex-shrink: 0; }
.he-empty { text-align: center; padding: 40px 24px; border: 1px dashed #334766; border-radius: 16px; background: #182235; }
.he-empty-icon { width: 52px; height: 52px; border-radius: 14px; background: rgba(59,130,246,.14); color: #60A5FA; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 14px; }
.he-empty-title { font-size: 1.05rem; font-weight: 600; color: #F8FAFC; }
.he-empty-text { color: #94A3B8; font-size: .88rem; max-width: 460px; margin: 6px auto 0; line-height: 1.55; }
.he-preview { display: flex; align-items: center; gap: 10px; background: rgba(59,130,246,.1); border: 1px solid rgba(59,130,246,.3); color: #CBD5E1; border-radius: 12px; padding: 12px 16px; font-size: .88rem; margin: 12px 0 6px; flex-wrap: wrap; }
.he-preview b { color: #60A5FA; }
.he-versus-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.he-versus-name { font-size: 1rem; font-weight: 600; color: #F8FAFC; }
.he-save-card { border-radius: 16px; padding: 18px 20px; border: 1px solid #263449; background: #182235; min-height: 120px; }
.he-save-card.positive { border-color: rgba(16,185,129,.45); background: linear-gradient(180deg, rgba(16,185,129,.10), #182235); }
.he-save-card.negative { border-color: rgba(239,68,68,.45); background: linear-gradient(180deg, rgba(239,68,68,.10), #182235); }
.he-ap-name { font-weight: 600; color: #F8FAFC; font-size: .98rem; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.he-ap-specs { color: #94A3B8; font-size: .82rem; margin-top: 4px; }
.he-ap-kwh { font-weight: 700; color: #F8FAFC; text-align: right; }
.he-ap-sub { color: #94A3B8; font-size: .78rem; text-align: right; }

/* ---------- tables ---------- */
.he-table-wrap { overflow-x: auto; border: 1px solid #263449; border-radius: 12px; background: #182235; margin-top: 8px; }
.he-table { width: 100%; border-collapse: collapse; font-size: .86rem; color: #F8FAFC; }
.he-table th { text-align: left; padding: 12px 14px; color: #94A3B8; font-weight: 600; font-size: .78rem; background: #1E293B; border-bottom: 1px solid #263449; white-space: nowrap; }
.he-table td { padding: 12px 14px; border-bottom: 1px solid #263449; white-space: nowrap; }
.he-table tbody tr:last-child td { border-bottom: none; }
.he-table tbody tr:hover td { background: rgba(59,130,246,.07); }
.he-table .he-right { text-align: right; }
.he-chip { display: inline-block; background: #1E293B; border: 1px solid #263449; color: #94A3B8; border-radius: 6px; padding: 2px 8px; font-size: .72rem; font-weight: 500; }
.he-rank { display: inline-flex; width: 26px; height: 26px; border-radius: 8px; background: #1E293B; color: #94A3B8; align-items: center; justify-content: center; font-weight: 700; font-size: .78rem; }
.he-rank-1 { background: rgba(59,130,246,.2); color: #60A5FA; }
.he-share { display: flex; align-items: center; gap: 10px; min-width: 140px; }
.he-share > span { width: 48px; text-align: right; }
.he-share-bar { flex: 1; height: 6px; background: #263449; border-radius: 99px; overflow: hidden; }
.he-share-bar i { display: block; height: 100%; background: #3B82F6; border-radius: 99px; }

/* ---------- Streamlit widgets in the dark theme ---------- */
[data-testid="stWidgetLabel"] p, label p { color: #94A3B8; font-size: .84rem; font-weight: 500; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"] > div { background-color: #1E293B !important; border-color: #263449 !important; border-radius: 10px !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within { border-color: #3B82F6 !important; }
input, textarea { color: #F8FAFC !important; -webkit-text-fill-color: #F8FAFC !important; background: transparent !important; }
input::placeholder { color: #64748B !important; -webkit-text-fill-color: #64748B !important; }
[data-baseweb="select"] div, [data-baseweb="select"] span { color: #F8FAFC; }
[data-baseweb="popover"] > div, [data-baseweb="popover"] ul, [data-baseweb="menu"] { background: #1E293B !important; border: 1px solid #263449; }
[data-baseweb="popover"] li, [data-baseweb="popover"] li * { color: #F8FAFC !important; }
[data-baseweb="popover"] li:hover { background: rgba(59,130,246,.18) !important; }
.stNumberInput button { background: #263449 !important; color: #F8FAFC !important; border-color: #263449 !important; }
.stButton > button, .stDownloadButton > button, [data-testid^="stBaseButton-secondary"] { background: #1E293B; color: #F8FAFC; border: 1px solid #263449; border-radius: 10px; font-weight: 600; transition: border-color .15s, color .15s, background .15s; }
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid^="stBaseButton-secondary"]:hover { border-color: #3B82F6; color: #60A5FA; background: #1E293B; }
[data-testid^="stBaseButton-primary"], button[kind="primary"] { background: #3B82F6; color: #FFFFFF; border: 1px solid #3B82F6; border-radius: 10px; font-weight: 600; }
[data-testid^="stBaseButton-primary"]:hover, button[kind="primary"]:hover { background: #2563EB; border-color: #2563EB; color: #FFFFFF; }
.stButton button p, .stDownloadButton button p { color: inherit; }
button[data-baseweb="tab"] { color: #94A3B8; background: transparent; }
button[data-baseweb="tab"][aria-selected="true"] { color: #60A5FA; }
[data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #263449; }
[data-baseweb="tab-highlight"] { background: #3B82F6 !important; }
[data-testid="stFileUploaderDropzone"] { background: #1E293B; border: 1px dashed #334766; border-radius: 12px; }
[data-testid="stFileUploaderDropzone"] * { color: #94A3B8; }
[data-testid="stExpander"] { background: #182235; border: 1px solid #263449; border-radius: 12px; }
[data-testid="stCheckbox"] p, [data-testid="stRadio"] p, [data-testid="stToggle"] p { color: #CBD5E1; }
[data-testid="stCode"] pre, code { background: #1E293B !important; color: #E2E8F0; }
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #94A3B8; }
</style>
"""

ICONS = {
    "bolt": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "wallet": '<path d="M20 12V8H6a2 2 0 0 1-2-2c0-1.1.9-2 2-2h12v4"/><path d="M4 6v12c0 1.1.9 2 2 2h14v-4"/><path d="M18 12a2 2 0 0 0-2 2c0 1.1.9 2 2 2h4v-4h-4z"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "plug": '<path d="M12 22v-5"/><path d="M9 8V2"/><path d="M15 8V2"/><path d="M18 8v5a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V8Z"/>',
    "trend_up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "trend_down": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "bulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "up": '<line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/>',
    "down": '<line x1="7" y1="7" x2="17" y2="17"/><polyline points="17 7 17 17 7 17"/>',
    "award": '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
}


def icon(name, size=18):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round">{ICONS[name]}</svg>')


def compact(markup):
    """Streamlit's markdown treats indented lines as code, so strip every line first."""
    return "".join(line.strip() for line in markup.strip().splitlines())


def render(markup):
    st.markdown(compact(markup), unsafe_allow_html=True)


def esc(value):
    return html.escape(str(value))


def money(value):
    return f"-₹{abs(value):,.2f}" if value < 0 else f"₹{value:,.2f}"


def signed_money(value):
    return f"{'-' if value < 0 else '+'}₹{abs(value):,.2f}"


def clamp(value, low, high):
    return max(low, min(high, value))


def card(key):
    """A styled card container. The key must be unique on the page."""
    return st.container(key=f"card_{key}")


def page_header(title, subtitle):
    render(f'<div class="he-title">{esc(title)}</div><div class="he-subtitle">{esc(subtitle)}</div>')


def section_header(title, subtitle=""):
    sub = f'<div class="he-section-sub">{esc(subtitle)}</div>' if subtitle else ""
    render(f'<div class="he-section"><div class="he-section-title">{esc(title)}</div>{sub}</div>')


def card_title(title, subtitle=""):
    sub = f'<div class="he-card-sub">{esc(subtitle)}</div>' if subtitle else '<div style="height:10px"></div>'
    render(f'<div class="he-card-title">{esc(title)}</div>{sub}')


def group_label(text):
    render(f'<div class="he-group-label">{esc(text)}</div>')


def pill(icon_name, text):
    return f'<span class="he-pill">{icon(icon_name, 14)}{esc(text)}</span>'


def badge(level, text=None):
    return f'<span class="he-badge he-badge-{level}">{esc(text or LEVEL_LABELS[level])}</span>'


def progress_html(pct, level):
    width = clamp(pct, 0, 100)  # the bar never goes past 100%, the number beside it stays honest
    return (f'<div class="he-progress"><span style="width:{width:.1f}%;'
            f'background:{LEVEL_COLORS[level]}"></span></div>')


def metric_card(icon_name, label, value, unit="", sub="", accent="blue", extra=""):
    unit_html = f"<small>{esc(unit)}</small>" if unit else ""
    sub_html = f'<div class="he-metric-sub">{sub}</div>' if sub else ""
    return (f'<div class="he-metric"><div class="he-metric-top"><span class="he-metric-label">{esc(label)}</span>'
            f'<span class="he-icon he-icon-{accent}">{icon(icon_name, 18)}</span></div>'
            f'<div class="he-metric-value">{esc(value)}{unit_html}</div>{extra}{sub_html}</div>')


def delta_chip(change_pct, reference):
    """Consumption going up is shown in orange, going down in green."""
    if change_pct is None or pd.isna(change_pct):
        return ""
    if abs(change_pct) < 0.05:
        cls, arrow = "flat", ""
    elif change_pct > 0:
        cls, arrow = "up", icon("up", 12)
    else:
        cls, arrow = "down", icon("down", 12)
    return f'<div><span class="he-chip-delta he-delta-{cls}">{arrow}{change_pct:+.1f}% vs {esc(reference)}</span></div>'


def insight_card(icon_name, accent, label, main, text=""):
    text_html = f'<div class="he-insight-text">{text}</div>' if text else ""
    return (f'<div class="he-insight"><div class="he-insight-label"><span class="he-icon he-icon-{accent}">'
            f'{icon(icon_name, 18)}</span>{esc(label)}</div><div class="he-insight-main">{main}</div>{text_html}</div>')


def empty_html(icon_name, title, text):
    return (f'<div class="he-empty"><div class="he-empty-icon">{icon(icon_name, 26)}</div>'
            f'<div class="he-empty-title">{esc(title)}</div><div class="he-empty-text">{esc(text)}</div></div>')


NOTICE_ICONS = {"success": "check", "warning": "alert", "error": "alert", "info": "info"}


def notice(kind, message, title=None):
    """kind: success, warning, error or info. The message is plain text (it gets escaped)."""
    title_html = f'<div class="he-notice-title">{esc(title)}</div>' if title else ""
    render(f'<div class="he-notice he-n-{kind}"><span>{icon(NOTICE_ICONS[kind], 20)}</span>'
           f'<div>{title_html}<div class="he-notice-body">{esc(message)}</div></div></div>')


def flash(kind, message):
    """Keeps a message alive across st.rerun(), so it shows on the refreshed page."""
    st.session_state["flash"] = (kind, message)


def show_flash():
    item = st.session_state.pop("flash", None)
    if item:
        notice(*item)


def show_errors(errors, limit=15):
    for message in errors[:limit]:
        notice("error", message)
    if len(errors) > limit:
        st.caption(f"…and {len(errors) - limit} more problem(s).")


def goto(page_name):
    """Switch pages from a button. Applied at the start of the next run."""
    st.session_state["_goto"] = page_name
    st.rerun()


def alert_banner(b):
    """One message per alert level, each with its own colour and icon."""
    if b["level"] == "exceeded":
        notice("error", f"The estimated bill of {money(b['bill'])} is {money(b['exceeded_by'])} "
                        f"over your {money(b['budget'])} budget.", "Budget exceeded")
    elif b["level"] == "warning":
        notice("warning", f"About {b['utilization_pct']:.1f}% of your budget is used. "
                          f"Only {money(b['remaining'])} is left.", "Budget warning")
    elif b["level"] == "ok":
        notice("success", f"{b['utilization_pct']:.1f}% of your budget is used, "
                          f"with {money(b['remaining'])} remaining.", "Within budget")
    else:
        notice("info", "Set a monthly budget on the Budget & Alerts page to switch on alerts.", "No budget set")


def budget_card_html(b, settings, title="Budget Monitoring"):
    level = b["level"]
    if not b["is_set"]:
        return (f'<div class="he-card"><div class="he-card-head"><div class="he-card-title">{esc(title)}</div>'
                f'{badge(level)}</div><div class="he-big-sub">No monthly budget is set, so alerts are switched off.</div>'
                f'{progress_html(0, level)}'
                f'<div class="he-row"><span>Estimated bill</span><span>{money(b["bill"])}</span></div>'
                f'<div class="he-row"><span>Monthly budget</span><span>Not set</span></div></div>')

    remaining_cls = "he-red" if b["remaining"] < 0 else ""
    exceeded_row = ""
    if b["exceeded_by"] > 0:
        exceeded_row = f'<div class="he-row"><span>Exceeded by</span><span class="he-red">{money(b["exceeded_by"])}</span></div>'
    cap_note = " (bar capped at 100%)" if b["utilization_pct"] > 100 else ""
    return (f'<div class="he-card"><div class="he-card-head"><div class="he-card-title">{esc(title)}</div>{badge(level)}</div>'
            f'<div class="he-big" style="color:{LEVEL_COLORS[level]}">{b["utilization_pct"]:.1f}%</div>'
            f'<div class="he-big-sub">of monthly budget used{cap_note}</div>'
            f'{progress_html(b["utilization_pct"], level)}'
            f'<div class="he-row"><span>Monthly budget</span><span>{money(b["budget"])}</span></div>'
            f'<div class="he-row"><span>Estimated bill</span><span>{money(b["bill"])}</span></div>'
            f'<div class="he-row"><span>Remaining</span><span class="{remaining_cls}">{money(b["remaining"])}</span></div>'
            f'{exceeded_row}'
            f'<div class="he-big-sub" style="margin-top:8px">Warning starts at {settings["warning_threshold_pct"]:.0f}% of the budget.</div></div>')


# ---------- table helpers ----------
# A column spec is (header, sort_column_or_None, formatter(row) -> safe html, "left" or "right")

def f_text(col):
    return lambda r: esc(r[col])


def f_chip(col):
    return lambda r: f'<span class="he-chip">{esc(r[col])}</span>'


def f_num(col, pattern="{:,.1f}", suffix=""):
    return lambda r: (pattern.format(r[col]) + suffix) if pd.notna(r[col]) else "—"


def f_money(col):
    return lambda r: money(r[col]) if pd.notna(r[col]) else "—"


def f_rank(col="Rank"):
    return lambda r: f'<span class="he-rank he-rank-{int(r[col]) if int(r[col]) == 1 else 0}">{int(r[col])}</span>'


def f_share(col="Share_Pct"):
    def build(r):
        pct = float(r[col])
        return (f'<div class="he-share"><span>{pct:.1f}%</span><div class="he-share-bar">'
                f'<i style="width:{clamp(pct, 0, 100):.1f}%"></i></div></div>')
    return build


def f_change(col):
    def build(r):
        value = r[col]
        if pd.isna(value):
            return "—"
        cls = "he-orange" if value > 0.05 else ("he-green" if value < -0.05 else "")
        return f'<span class="{cls}">{value:+.1f}%</span>'
    return build


def f_status(col="Alert_Status"):
    return lambda r: badge(STATUS_TO_LEVEL.get(str(r[col]), "none"), str(r[col]))


APPLIANCE_COLS = [
    ("Appliance", "Name", f_text("Name"), "left"),
    ("Category", "Category", f_chip("Category"), "left"),
    ("Power (W)", "Power_W", f_num("Power_W", "{:,.0f}"), "right"),
    ("Qty", "Quantity", f_num("Quantity", "{:,.0f}"), "right"),
    ("Hours/day", "Hours_Per_Day", f_num("Hours_Per_Day", "{:g}"), "right"),
    ("Days/month", "Days_Per_Month", f_num("Days_Per_Month", "{:,.0f}"), "right"),
]
RANKING_COLS = [
    ("Rank", "Rank", f_rank(), "left"),
    ("Appliance", "Name", f_text("Name"), "left"),
    ("Category", "Category", f_chip("Category"), "left"),
    ("Power", "Power_W", lambda r: f"{r['Power_W']:,.0f} W × {int(r['Quantity'])}", "right"),
    ("Monthly consumption", "Monthly_kWh", f_num("Monthly_kWh", "{:,.1f}", " kWh"), "right"),
    ("Est. monthly cost", "Monthly_Cost", f_money("Monthly_Cost"), "right"),
    ("Consumption share", "Share_Pct", f_share(), "left"),
]
IDEA_TABLE_COLS = [
    ("Appliance", "Appliance", f_text("Appliance"), "left"),
    ("Current hours/day", "Current_Hours", f_num("Current_Hours", "{:g}"), "right"),
    ("Reduce by (h/day)", "Reduce_By_Hours", f_num("Reduce_By_Hours", "{:g}"), "right"),
    ("Saves (kWh/month)", "kWh_Saved", f_num("kWh_Saved", "{:,.2f}"), "right"),
    ("Saves (₹/month)", "Monthly_Saving", f_money("Monthly_Saving"), "right"),
    ("Saves (₹/year)", "Annual_Saving", f_money("Annual_Saving"), "right"),
]


def data_table(df, columns, key, controls=True, max_rows=None, default_sort=None,
               empty_text="Nothing to show yet."):
    """A dark HTML table with optional search, category filter and sorting."""
    work = df.copy()

    if controls and not work.empty:
        sortable = [c[0] for c in columns if c[1]]
        c1, c2, c3, c4 = st.columns([2.2, 1.4, 1.5, 0.9])
        query = c1.text_input("Search", key=f"{key}_q", placeholder="Search by name or category",
                              label_visibility="collapsed")
        category = "All categories"
        if "Category" in work.columns:
            category = c2.selectbox("Category", ["All categories"] + sorted(work["Category"].unique()),
                                    key=f"{key}_cat", label_visibility="collapsed")
        default_index = sortable.index(default_sort) if default_sort in sortable else 0
        sort_header = c3.selectbox("Sort by", sortable, index=default_index, key=f"{key}_sort",
                                   label_visibility="collapsed")
        descending = c4.toggle("Descending", value=True, key=f"{key}_desc")

        if query.strip():
            needle = query.strip()
            mask = pd.Series(False, index=work.index)
            for col in ("Name", "Category", "Appliance"):
                if col in work.columns:
                    mask |= work[col].astype(str).str.contains(needle, case=False, regex=False, na=False)
            work = work[mask]
        if category != "All categories":
            work = work[work["Category"] == category]
        sort_col = next(c[1] for c in columns if c[0] == sort_header)
        work = work.sort_values(sort_col, ascending=not descending, kind="mergesort")

    if work.empty:
        render(empty_html("info", "No matching records", empty_text))
        return

    total = len(work)
    if max_rows:
        work = work.head(max_rows)

    head = "".join(f'<th class="{"he-right" if align == "right" else ""}">{esc(header)}</th>'
                   for header, _, _, align in columns)
    body = ""
    for _, row in work.iterrows():
        cells = "".join(f'<td class="{"he-right" if align == "right" else ""}">{fmt(row)}</td>'
                        for _, _, fmt, align in columns)
        body += f"<tr>{cells}</tr>"
    render(f'<div class="he-table-wrap"><table class="he-table"><thead><tr>{head}</tr></thead>'
           f'<tbody>{body}</tbody></table></div>')
    if max_rows and total > max_rows:
        st.caption(f"Showing {max_rows} of {total} appliances.")


# =============================================================================
# 5. PLOTLY CHARTS (transparent backgrounds so they sit on the dark cards)
# =============================================================================

DONUT_COLORS = [BLUE, ELECTRIC, GREEN, ORANGE, "#8B5CF6", "#14B8A6", "#64748B"]


def _style(fig, height=340, legend=True, x_title=None, y_title=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
        font=dict(family="Inter, Segoe UI, sans-serif", color=MUTED, size=12),
        margin=dict(l=8, r=8, t=24, b=8), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=CARD_2, bordercolor=BORDER, font=dict(color=TEXT)))
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER, title_text=x_title,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER, title_text=y_title,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED), automargin=True)
    return fig


def _has_values(df, column):
    return df is not None and not df.empty and df[column].fillna(0).sum() > 0


def consumption_trend_chart(history):
    if history is None or history.empty:
        return None
    labels = [format_month(m) for m in history["Month"]]
    fig = go.Figure(go.Scatter(
        x=labels, y=history["Total_kWh"], mode="lines+markers", name="Estimated consumption (kWh)",
        line=dict(color=BLUE, width=3), marker=dict(size=8, color=ELECTRIC, line=dict(color=BG, width=2)),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.14)",
        hovertemplate="%{x}<br>%{y:,.1f} kWh<extra></extra>"))
    _style(fig, 330, legend=True, x_title="Month", y_title="Energy (kWh)")
    fig.update_xaxes(type="category", showgrid=False)
    fig.update_yaxes(rangemode="tozero")
    return fig


def appliance_bar_chart(analysis, column="Monthly_kWh"):
    """Horizontal bars, biggest on top. The highest consumer is drawn in the lighter blue."""
    if not _has_values(analysis, column):
        return None
    data = analysis[analysis[column] > 0].sort_values(column, ascending=True)
    top_value = data[column].max()
    colors = [ELECTRIC if v == top_value else "rgba(59,130,246,0.6)" for v in data[column]]
    is_energy = column == "Monthly_kWh"
    label = "Monthly energy (kWh)" if is_energy else "Monthly energy cost (₹)"
    fmt = "{:,.1f} kWh" if is_energy else "₹{:,.0f}"
    fig = go.Figure(go.Bar(
        x=data[column], y=data["Name"], orientation="h", marker=dict(color=colors),
        name=label, text=[fmt.format(v) for v in data[column]], textposition="outside",
        textfont=dict(color=TEXT, size=11), cliponaxis=False,
        hovertemplate="%{y}<br>" + ("%{x:,.1f} kWh" if is_energy else "₹%{x:,.2f}") + "<extra></extra>"))
    _style(fig, max(300, 36 * len(data) + 90), legend=False,
           x_title="Energy (kWh per month)" if is_energy else "Energy cost (₹ per month)", y_title=None)
    fig.update_xaxes(range=[0, top_value * 1.25])
    fig.update_yaxes(showgrid=False)
    return fig


def contribution_donut(analysis):
    if not _has_values(analysis, "Monthly_kWh"):
        return None
    data = analysis[analysis["Monthly_kWh"] > 0].sort_values("Monthly_kWh", ascending=False)
    labels, values = list(data["Name"].head(6)), list(data["Monthly_kWh"].head(6))
    rest = data["Monthly_kWh"].iloc[6:].sum()
    if rest > 0:  # lump the small ones together so the chart stays readable
        labels.append("Others")
        values.append(rest)
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.64, sort=False, direction="clockwise",
        marker=dict(colors=DONUT_COLORS[:len(values)], line=dict(color=CARD, width=2)),
        textinfo="percent", textfont=dict(color=TEXT, size=11),
        hovertemplate="%{label}<br>%{value:,.1f} kWh (%{percent})<extra></extra>"))
    _style(fig, 340, legend=True)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5,
                                  font=dict(color=MUTED, size=11)),
                      annotations=[dict(text=f"<b>{sum(values):,.0f}</b><br>kWh / month", x=0.5, y=0.5,
                                        showarrow=False, font=dict(color=TEXT, size=15))])
    return fig


def monthly_cost_chart(history):
    if history is None or history.empty:
        return None
    labels = [format_month(m) for m in history["Month"]]
    fig = go.Figure()
    fig.add_bar(x=labels, y=history["Total_Bill"], name="Estimated total bill (₹)",
                marker=dict(color="rgba(59,130,246,0.65)"),
                hovertemplate="%{x}<br>Total bill ₹%{y:,.2f}<extra></extra>")
    fig.add_scatter(x=labels, y=history["Energy_Cost"], name="Energy cost (₹)", mode="lines+markers",
                    line=dict(color=GREEN, width=3), marker=dict(size=7),
                    hovertemplate="%{x}<br>Energy cost ₹%{y:,.2f}<extra></extra>")
    _style(fig, 330, legend=True, x_title="Month", y_title="Amount (₹)")
    fig.update_xaxes(type="category", showgrid=False)
    return fig


def budget_vs_bill_history_chart(history):
    if history is None or history.empty:
        return None
    labels = [format_month(m) for m in history["Month"]]
    over = (history["Budget"] > 0) & (history["Total_Bill"] > history["Budget"])
    bill_colors = [RED if flag else BLUE for flag in over]
    fig = go.Figure()
    fig.add_bar(x=labels, y=history["Budget"], name="Monthly budget (₹)", marker=dict(color="#475569"),
                hovertemplate="%{x}<br>Budget ₹%{y:,.2f}<extra></extra>")
    fig.add_bar(x=labels, y=history["Total_Bill"], name="Estimated bill (₹)", marker=dict(color=bill_colors),
                hovertemplate="%{x}<br>Estimated bill ₹%{y:,.2f}<extra></extra>")
    _style(fig, 330, legend=True, x_title="Month", y_title="Amount (₹)")
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category", showgrid=False)
    fig.add_annotation(text="Red bars: estimated bill over budget", xref="paper", yref="paper", x=1, y=1.12,
                       showarrow=False, xanchor="right", font=dict(color=RED, size=11))
    return fig


def bill_vs_budget_chart(budget, bills):
    """bills = {"label": amount}. Bars turn red above the budget; a dashed line marks the budget."""
    labels, values = list(bills.keys()), [float(v) for v in bills.values()]
    if not values:
        return None
    colors = [RED if (budget > 0 and v > budget) else BLUE for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker=dict(color=colors), width=0.45,
                           name="Estimated bill (₹)", text=[f"₹{v:,.0f}" for v in values], textposition="outside",
                           textfont=dict(color=TEXT), cliponaxis=False,
                           hovertemplate="%{x}<br>₹%{y:,.2f}<extra></extra>"))
    if budget > 0:
        fig.add_hline(y=budget, line=dict(color=ORANGE, width=2, dash="dash"),
                      annotation_text=f"Budget ₹{budget:,.0f}", annotation_position="top left",
                      annotation_font=dict(color=ORANGE))
    _style(fig, 330, legend=False, x_title=None, y_title="Amount (₹ per month)")
    fig.update_yaxes(range=[0, max(max(values), budget, 1.0) * 1.25])  # works even when all values are zero
    fig.update_xaxes(showgrid=False)
    return fig


def before_after_chart(result):
    labels = ["This appliance", "Household total"]
    fig = go.Figure()
    fig.add_bar(x=labels, y=[result["original_kwh"], result["original_total_kwh"]], name="Before (kWh)",
                marker=dict(color="#475569"), text=[f"{result['original_kwh']:,.1f}", f"{result['original_total_kwh']:,.1f}"],
                textposition="outside", textfont=dict(color=TEXT), cliponaxis=False)
    fig.add_bar(x=labels, y=[result["new_kwh"], result["new_total_kwh"]], name="After (kWh)",
                marker=dict(color=GREEN), text=[f"{result['new_kwh']:,.1f}", f"{result['new_total_kwh']:,.1f}"],
                textposition="outside", textfont=dict(color=TEXT), cliponaxis=False)
    _style(fig, 330, legend=True, x_title=None, y_title="Energy (kWh per month)")
    fig.update_layout(barmode="group")
    top = max(result["original_total_kwh"], result["new_total_kwh"], 1.0)
    fig.update_yaxes(range=[0, top * 1.25])
    fig.update_xaxes(showgrid=False)
    return fig


STREAMLIT_VERSION = tuple(int(p) for p in re.findall(r"\d+", st.__version__)[:2])


def show_plot(fig, key):
    """Draws a Plotly figure on the card, using whichever width option this Streamlit version has."""
    config = {"displayModeBar": False}
    if STREAMLIT_VERSION >= (1, 50):
        st.plotly_chart(fig, width="stretch", theme=None, config=config, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key=key)


# =============================================================================
# 6. SHARED PAGE PIECES
# =============================================================================

ESTIMATE_NOTE = ("These figures are estimates based on rated power. Real consumption can differ, "
                 "especially for appliances that cycle on and off (fridges, air conditioners, geysers) "
                 "or draw less than their rated power.")


def empty_appliance_state(key):
    """Shown on any page that needs appliances when there are none yet."""
    render(empty_html("plug", "No appliances yet",
                      "Add the appliances in your home to see consumption, bill and budget estimates. "
                      "You can also load the illustrative sample list to explore the dashboard."))
    st.write("")
    c1, c2, _ = st.columns([1.3, 1.3, 3])
    if c1.button("Load sample appliances", key=f"sample_{key}", icon=":material/science:"):
        added, _, error = add_sample_appliances()
        if error:
            flash("error", error)
        else:
            flash("success", f"Added {added} sample appliances. They are illustrative, not real measurements.")
        st.rerun()
    if c2.button("Add my own", key=f"own_{key}", type="primary", icon=":material/add:"):
        st.session_state["app_mode"] = "add"
        goto("Appliances")


def render_insights(analysis, settings, s):
    section_header("Energy Saving Insights",
                   "Calculated from your appliance data. These are estimates, not guaranteed savings.")
    if analysis.empty or s["total_monthly_kwh"] <= 0:
        render(empty_html("bulb", "No insights yet", "Add appliances with some daily usage to see saving ideas."))
        return

    top = rank_appliances(analysis).iloc[0]
    ideas = saving_ideas(analysis, settings["tariff_per_kwh"])

    c1, c2, c3 = st.columns(3)
    with c1:
        render(insight_card("award", "blue", "Highest-consuming appliance", esc(top["Name"]),
                            f"{top['Monthly_kWh']:,.1f} kWh per month, about {top['Share_Pct']:.1f}% of your estimated total."))
    if not ideas.empty:
        first = ideas.iloc[0]
        with c2:
            render(insight_card("trend_down", "green", "Largest saving opportunity",
                                f"Use {esc(first['Appliance'])} {first['Reduce_By_Hours']:g} h/day less",
                                f"Saves about {first['kWh_Saved']:,.1f} kWh and {money(first['Monthly_Saving'])} per month "
                                f"({money(first['Annual_Saving'])} a year)."))
        with c3:
            render(insight_card("wallet", "orange", "Combined potential savings",
                                f"{money(ideas['Monthly_Saving'].sum())} / month",
                                f"All {len(ideas)} suggested cuts together save about {ideas['kWh_Saved'].sum():,.1f} kWh "
                                f"and {money(ideas['Annual_Saving'].sum())} a year."))

    high = high_consumers(analysis)
    if not high.empty:
        notice("warning", ", ".join(high["Name"]) + f" each make up {HIGH_SHARE_PCT:.0f}% or more of your estimated consumption.",
               "High estimated consumption")

    if not ideas.empty:
        st.write("")
        card_title("Suggested running-time cuts", "A one-hour cut (or 25% of current hours, if smaller) on your top consumers.")
        data_table(ideas, IDEA_TABLE_COLS, "ideas", controls=False)

    tips = general_tips(analysis)
    if tips:
        st.write("")
        card_title("Tips based on your appliances")
        for tip in tips:
            safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc(tip))
            render(f'<div class="he-tip"><span class="he-tip-icon">{icon("bulb", 18)}</span><div>{safe}</div></div>')


# =============================================================================
# 7. PAGES
# =============================================================================

def page_dashboard(ctx):
    s, analysis, settings, history = ctx["summary"], ctx["analysis"], ctx["settings"], ctx["history"]
    b = s["budget"]

    c_title, c_pills, c_button = st.columns([3.1, 3, 1.1])
    with c_title:
        page_header("Energy Overview", "Monitor your household electricity consumption and spending.")
    with c_pills:
        render(f'<div class="he-pills">{pill("calendar", date.today().strftime("%B %Y"))}'
               f'{pill("clock", last_updated_text())}</div>')
    with c_button:
        if st.button("Refresh", icon=":material/refresh:", key="dash_refresh"):
            st.rerun()

    st.write("")
    if analysis.empty:
        empty_appliance_state("dashboard")
        return

    # ---- four metric cards (all values come from the calculation functions) ----
    delta = ""
    if not history.empty and history.iloc[-1]["Total_kWh"] > 0:
        last = history.iloc[-1]
        delta = delta_chip((s["total_monthly_kwh"] - last["Total_kWh"]) / last["Total_kWh"] * 100,
                           format_month(last["Month"]))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render(metric_card("bolt", "Monthly consumption", f"{s['total_monthly_kwh']:,.1f}", "kWh",
                           "Estimated electricity usage", "blue", delta))
    with m2:
        render(metric_card("wallet", "Estimated electricity bill", money(s["total_bill"]), "",
                           f"Energy {money(s['energy_cost'])} + fixed {money(s['fixed_charge'])}", "green"))
    with m3:
        if b["is_set"]:
            bar = progress_html(b["utilization_pct"], b["level"])
            rem_cls = "he-red" if b["remaining"] < 0 else ""
            render(metric_card("target", "Budget utilization", f"{b['utilization_pct']:.1f}%", "",
                               f'Remaining <span class="{rem_cls}">{money(b["remaining"])}</span>',
                               {"ok": "green", "warning": "orange", "exceeded": "red"}[b["level"]], bar))
        else:
            render(metric_card("target", "Budget utilization", "Not set", "", "Set a monthly budget to see utilization", "orange"))
    with m4:
        render(metric_card("plug", "Total appliances", str(s["appliance_count"]), "",
                           f"Highest: {esc(s['top_appliance'])}", "blue"))

    # ---- trend + budget ----
    section_header("Consumption Analytics")
    left, right = st.columns([2, 1])
    with left:
        with card("trend"):
            card_title("Electricity Consumption Trend", "Monthly estimated consumption from your saved history")
            fig = consumption_trend_chart(history)
            if fig is None:
                render(empty_html("calendar", "No monthly history yet",
                                  "Save this month's summary on the Monthly History page and your trend line will appear here."))
            else:
                show_plot(fig, "dash_trend")
    with right:
        render(budget_card_html(b, settings))

    # ---- appliance analysis ----
    section_header("Appliance Analysis", "Where your estimated electricity goes")
    left, right = st.columns([3, 2])
    with left:
        with card("dash_bar"):
            card_title("Monthly Consumption by Appliance", "Highest consumer shown in the lighter blue")
            fig = appliance_bar_chart(analysis)
            if fig is None:
                render(empty_html("info", "Nothing to chart", "Appliances need some daily usage to appear here."))
            else:
                show_plot(fig, "dash_bar")
    with right:
        with card("dash_donut"):
            card_title("Consumption Share", "Share of estimated monthly kWh")
            fig = contribution_donut(analysis)
            if fig is None:
                render(empty_html("info", "Nothing to chart", "Appliances need some daily usage to appear here."))
            else:
                show_plot(fig, "dash_donut")

    st.write("")
    with card("dash_table"):
        card_title("Top Appliances", "Ranked by estimated monthly consumption. Search, filter or change the sorting.")
        data_table(rank_appliances(analysis), RANKING_COLS, "dash_top", max_rows=10,
                   default_sort="Monthly consumption", empty_text="No appliance matches your search.")

    render_insights(analysis, settings, s)
    st.caption(ESTIMATE_NOTE)


def appliance_form(appliances, settings, mode):
    """Add form (mode == 'add') or edit form (mode == appliance ID). No st.form, so the preview is live."""
    editing = mode != "add"
    if editing:
        match = appliances[appliances["ID"] == mode]
        if match.empty:
            st.session_state["app_mode"] = None
            st.rerun()
        row = match.iloc[0]
        d = {"name": row["Name"], "category": row["Category"],
             "power": clamp(float(row["Power_W"]), 1.0, float(MAX_POWER_W)),
             "quantity": int(clamp(row["Quantity"], 1, MAX_QUANTITY)),
             "hours": clamp(float(row["Hours_Per_Day"]), 0.0, 24.0),
             "days": int(clamp(row["Days_Per_Month"], 1, 31))}
        k = f"edit_{mode}"
    with card("appliance_form"):
        card_title("Edit appliance" if editing else "Add appliance",
                   "Check the rated power on the appliance label for the most accurate estimate.")
        if not editing:
            choice = st.selectbox("Start from", ["Custom appliance"] + list(COMMON_APPLIANCES), key="add_preset")
            p = COMMON_APPLIANCES.get(choice, CUSTOM_DEFAULTS)
            d = {"name": "" if choice == "Custom appliance" else choice, "category": p["category"],
                 "power": float(p["power"]), "quantity": int(p["quantity"]),
                 "hours": float(p["hours"]), "days": int(p["days"])}
            k = f"add_{choice}_{st.session_state.get('add_version', 0)}"

        group_label("Appliance information")
        c1, c2 = st.columns(2)
        name = c1.text_input("Appliance name", value=d["name"], max_chars=MAX_NAME_LENGTH, key=f"f_name_{k}")
        category = c2.selectbox("Category", CATEGORIES, index=CATEGORIES.index(d["category"]), key=f"f_cat_{k}")

        group_label("Electrical specifications")
        c1, c2 = st.columns(2)
        power = c1.number_input("Rated power (W)", 1.0, float(MAX_POWER_W), float(d["power"]), 1.0, key=f"f_power_{k}")
        quantity = c2.number_input("Quantity", 1, MAX_QUANTITY, int(d["quantity"]), 1, key=f"f_qty_{k}")

        group_label("Usage information")
        c1, c2 = st.columns(2)
        hours = c1.number_input("Hours per day", 0.0, 24.0, float(d["hours"]), 0.25, key=f"f_hours_{k}")
        days = c2.number_input("Days per month", 1, 31, int(d["days"]), 1, key=f"f_days_{k}")

        daily = daily_energy_kwh(power, quantity, hours)
        monthly = monthly_energy_kwh(power, quantity, hours, days)
        render(f'<div class="he-preview">{icon("bolt", 16)}Estimated use: <b>{daily:,.2f} kWh/day</b> · '
               f'<b>{monthly:,.2f} kWh/month</b> · energy cost <b>{money(energy_cost(monthly, settings["tariff_per_kwh"]))}</b>/month</div>')

        b1, b2, _ = st.columns([1.2, 1, 3])
        if b1.button("Save changes" if editing else "Add appliance", type="primary", key=f"f_save_{k}",
                     icon=":material/check:"):
            if editing:
                errors = update_appliance(mode, name, category, power, quantity, hours, days)
            else:
                errors = add_appliance(name, category, power, quantity, hours, days)
            if errors:
                show_errors(errors)
            else:
                if not editing:
                    st.session_state["add_version"] = st.session_state.get("add_version", 0) + 1
                st.session_state["app_mode"] = None
                flash("success", f'"{name.strip()}" was {"updated" if editing else "added"}.')
                st.rerun()
        if b2.button("Cancel", key=f"f_cancel_{k}"):
            st.session_state["app_mode"] = None
            st.rerun()


def page_appliances(ctx):
    appliances, analysis, settings = ctx["appliances"], ctx["analysis"], ctx["settings"]

    c1, c2 = st.columns([4, 1.3])
    with c1:
        page_header("Appliances", "Add, edit and remove the appliances used in your estimates.")
    with c2:
        if st.button("Add appliance", icon=":material/add:", type="primary", key="open_add"):
            st.session_state["app_mode"] = "add"
            st.session_state["pending_delete"] = None
            st.rerun()

    mode = st.session_state.get("app_mode")
    if mode is not None:
        st.write("")
        appliance_form(appliances, settings, mode)

    st.write("")
    if appliances.empty:
        if mode is None:
            empty_appliance_state("manage")
        return

    section_header("Your appliances", f"{len(appliances)} appliance(s) in your list")
    f1, f2, _ = st.columns([2.2, 1.4, 2])
    query = f1.text_input("Search", key="ap_search", placeholder="Search by name or category",
                          label_visibility="collapsed")
    category = f2.selectbox("Category", ["All categories"] + sorted(appliances["Category"].unique()),
                            key="ap_cat", label_visibility="collapsed")

    shown = analysis.sort_values("Name", key=lambda col: col.str.lower())
    if query.strip():
        shown = shown[shown["Name"].str.contains(query.strip(), case=False, regex=False)
                      | shown["Category"].str.contains(query.strip(), case=False, regex=False)]
    if category != "All categories":
        shown = shown[shown["Category"] == category]
    if shown.empty:
        render(empty_html("info", "No matching appliances", "Try a different search or category."))
        return

    pending = st.session_state.get("pending_delete")
    for r in shown.itertuples():
        appliance_id = int(r.ID)
        with st.container(key=f"row_{appliance_id}"):
            c_info, c_use, c_edit, c_del = st.columns([4.4, 2, 1.1, 1.3], vertical_alignment="center")
            c_info.markdown(compact(
                f'<div class="he-ap-name">{esc(r.Name)}<span class="he-chip">{esc(r.Category)}</span></div>'
                f'<div class="he-ap-specs">{r.Power_W:,.0f} W × {int(r.Quantity)} · {r.Hours_Per_Day:g} h/day · '
                f'{int(r.Days_Per_Month)} days/month</div>'), unsafe_allow_html=True)
            c_use.markdown(compact(
                f'<div class="he-ap-kwh">{r.Monthly_kWh:,.1f} kWh</div>'
                f'<div class="he-ap-sub">{money(r.Monthly_Cost)} / month</div>'), unsafe_allow_html=True)
            if c_edit.button("Edit", key=f"edit_{appliance_id}", icon=":material/edit:"):
                st.session_state["app_mode"] = appliance_id
                st.session_state["pending_delete"] = None
                st.rerun()
            if c_del.button("Delete", key=f"del_{appliance_id}", icon=":material/delete:"):
                st.session_state["pending_delete"] = appliance_id
                st.rerun()

            if pending == appliance_id:
                notice("warning", f'Delete "{r.Name}" permanently? This cannot be undone.', "Confirm delete")
                d1, d2, _ = st.columns([1.3, 1, 3])
                if d1.button("Yes, delete", key=f"del_yes_{appliance_id}", type="primary"):
                    if delete_appliance(appliance_id):
                        flash("success", f'"{r.Name}" was deleted.')
                    else:
                        flash("error", "That appliance was already gone.")
                    st.session_state["pending_delete"] = None
                    st.rerun()
                if d2.button("Keep it", key=f"del_no_{appliance_id}"):
                    st.session_state["pending_delete"] = None
                    st.rerun()


def page_analytics(ctx):
    s, analysis, settings = ctx["summary"], ctx["analysis"], ctx["settings"]
    page_header("Consumption Analytics", "See how much each appliance uses per day, month and year.")
    notice("info", ESTIMATE_NOTE)

    if analysis.empty:
        empty_appliance_state("analytics")
        return

    m1, m2, m3 = st.columns(3)
    with m1:
        render(metric_card("bolt", "Estimated daily consumption", f"{s['total_daily_kwh']:,.2f}", "kWh",
                           "Sum of all appliances", "blue"))
    with m2:
        render(metric_card("calendar", "Estimated monthly consumption", f"{s['total_monthly_kwh']:,.1f}", "kWh",
                           "Uses each appliance's days per month", "green"))
    with m3:
        render(metric_card("trend_up", "Estimated annual consumption", f"{s['total_annual_kwh']:,.0f}", "kWh",
                           "Monthly estimate × 12, assuming similar months", "blue"))

    section_header("Appliance-wise consumption")
    with card("an_table"):
        data_table(analysis, [
            ("Appliance", "Name", f_text("Name"), "left"),
            ("Category", "Category", f_chip("Category"), "left"),
            ("Daily (kWh)", "Daily_kWh", f_num("Daily_kWh", "{:,.2f}"), "right"),
            ("Monthly (kWh)", "Monthly_kWh", f_num("Monthly_kWh", "{:,.1f}"), "right"),
            ("Annual (kWh)", "Annual_kWh", f_num("Annual_kWh", "{:,.0f}"), "right"),
            ("Share of total", "Share_Pct", f_share(), "left"),
        ], "an_all", default_sort="Monthly (kWh)")

    left, right = st.columns([3, 2])
    with left:
        with card("an_bar"):
            card_title("Monthly Energy by Appliance", "kWh per month")
            fig = appliance_bar_chart(analysis)
            if fig is not None:
                show_plot(fig, "an_bar")
    with right:
        with card("an_donut"):
            card_title("Contribution Share", "Share of estimated monthly kWh")
            fig = contribution_donut(analysis)
            if fig is not None:
                show_plot(fig, "an_donut")

    st.write("")
    with card("an_cost"):
        card_title("Estimated Energy Cost by Appliance", f"At ₹{settings['tariff_per_kwh']:.2f} per kWh, before fixed charges")
        fig = appliance_bar_chart(analysis, "Monthly_Cost")
        if fig is not None:
            show_plot(fig, "an_cost")

    section_header("Top five energy consumers")
    with card("an_top5"):
        data_table(top_n(analysis, 5), RANKING_COLS, "an_top", controls=False)
    render_insights(analysis, settings, s)


def page_bill(ctx):
    s, analysis, settings = ctx["summary"], ctx["analysis"], ctx["settings"]
    page_header("Bill Estimator", "Estimate your monthly and yearly electricity bill from your tariff.")
    notice("info", "Everything here is an estimate, not your actual bill. Real bills depend on your provider's slabs, "
                   "taxes, surcharges and meter readings. The tariff is only a placeholder, so enter the rate from your own bill.")

    left, right = st.columns([1, 1.15])
    with left:
        with card("bill_inputs"):
            card_title("Tariff and charges", "Saved settings are used on every page.")
            with st.form("tariff_form"):
                tariff = st.number_input("Electricity tariff (₹ per kWh)", 0.0, 1000.0,
                                         float(settings["tariff_per_kwh"]), 0.1, format="%.2f")
                fixed = st.number_input("Fixed monthly charges (₹, optional)", 0.0, 100000.0,
                                        float(settings["fixed_charge"]), 10.0, format="%.2f")
                saved = st.form_submit_button("Save tariff settings", type="primary")
            if saved:
                errors = update_settings({"tariff_per_kwh": tariff, "fixed_charge": fixed})
                if errors:
                    show_errors(errors)
                else:
                    flash("success", "Tariff settings saved.")
                    st.rerun()
    with right:
        render(f'<div class="he-card"><div class="he-card-head"><div class="he-card-title">Bill breakdown</div>'
               f'<span class="he-chip">Estimate</span></div>'
               f'<div class="he-row"><span>Total monthly consumption</span><span>{s["total_monthly_kwh"]:,.1f} kWh</span></div>'
               f'<div class="he-row"><span>Tariff per kWh</span><span>₹{settings["tariff_per_kwh"]:,.2f}</span></div>'
               f'<div class="he-row"><span>Energy cost</span><span>{money(s["energy_cost"])}</span></div>'
               f'<div class="he-row"><span>Fixed charges</span><span>{money(s["fixed_charge"])}</span></div>'
               f'<div class="he-row he-row-total"><span>Estimated total bill</span><span>{money(s["total_bill"])}</span></div>'
               f'<div class="he-row"><span>Annual estimate</span><span>{money(s["annual_cost"])}</span></div></div>')

    st.caption(f"{s['total_monthly_kwh']:,.1f} kWh × ₹{settings['tariff_per_kwh']:.2f} per kWh + "
               f"₹{settings['fixed_charge']:.2f} fixed charges. The annual estimate is the monthly total × 12.")

    if analysis.empty:
        st.write("")
        empty_appliance_state("bill")
        return

    section_header("Appliance-wise energy cost", "Fixed charges are not split between appliances.")
    left, right = st.columns([3, 2])
    with left:
        with card("bill_chart"):
            fig = appliance_bar_chart(analysis, "Monthly_Cost")
            if fig is not None:
                show_plot(fig, "bill_cost")
    with right:
        with card("bill_table"):
            data_table(rank_appliances(analysis).head(8), [
                ("Appliance", "Name", f_text("Name"), "left"),
                ("Monthly (kWh)", "Monthly_kWh", f_num("Monthly_kWh", "{:,.1f}"), "right"),
                ("Energy cost", "Monthly_Cost", f_money("Monthly_Cost"), "right"),
            ], "bill_t", controls=False)


def page_budget(ctx):
    s, analysis, settings = ctx["summary"], ctx["analysis"], ctx["settings"]
    b = s["budget"]
    page_header("Budget & Alerts", "Set a monthly budget and get warned before the estimate goes over it.")

    left, right = st.columns([1, 1.1])
    with left:
        with card("budget_form"):
            card_title("Budget settings", "Set the budget to 0 to switch alerts off.")
            with st.form("budget_form"):
                budget = st.number_input("Monthly budget (₹)", 0.0, 10000000.0, float(settings["monthly_budget"]),
                                         100.0, format="%.2f")
                threshold = st.slider("Warning threshold (% of budget)", 10, 99,
                                      int(round(clamp(settings["warning_threshold_pct"], 10, 99))))
                saved = st.form_submit_button("Save budget settings", type="primary")
            if saved:
                errors = update_settings({"monthly_budget": budget, "warning_threshold_pct": float(threshold)})
                if errors:
                    show_errors(errors)
                else:
                    flash("success", "Budget settings saved.")
                    st.rerun()
            st.caption(f"Below {settings['warning_threshold_pct']:.0f}% is normal, "
                       f"{settings['warning_threshold_pct']:.0f}% up to just under 100% is a warning, and 100% or more "
                       "is exceeded. Comparisons use the estimated bill, not your real bill.")
    with right:
        render(budget_card_html(b, settings, "Budget status"))

    alert_banner(b)

    section_header("Estimated bill vs budget")
    with card("budget_chart"):
        fig = bill_vs_budget_chart(b["budget"], {"Estimated bill": b["bill"]})
        if fig is not None:
            show_plot(fig, "budget_chart")

    if analysis.empty:
        st.write("")
        empty_appliance_state("budget")
        return

    section_header("Where you could save", "Recommendations use the same bill calculation as the alerts.")
    if b["level"] == "exceeded" and settings["tariff_per_kwh"] > 0:
        cut = energy_to_cut_kwh(b["exceeded_by"], settings["tariff_per_kwh"])
        notice("warning", f"To get back to the budget, the estimate needs to drop by about {money(b['exceeded_by'])} "
                          f"(roughly {cut:,.1f} kWh per month).", "Over budget")
    ideas = saving_ideas(analysis, settings["tariff_per_kwh"])
    if ideas.empty:
        notice("info", "No appliance has any estimated consumption to reduce.")
    else:
        data_table(ideas, IDEA_TABLE_COLS, "budget_ideas", controls=False)
        notice("success", f"Making all of these cuts together could save about {money(ideas['Monthly_Saving'].sum())} "
                          "per month (estimate). Try your own changes on the Savings Simulator page.", "Potential savings")


def versus_card(title, tag_html, rows):
    body = "".join(f'<div class="he-row"><span>{label}</span><span>{value}</span></div>' for label, value in rows)
    return (f'<div class="he-card"><div class="he-versus-head"><span class="he-versus-name">{title}</span>'
            f'{tag_html}</div>{body}</div>')


def page_simulator(ctx):
    analysis, settings = ctx["analysis"], ctx["settings"]
    page_header("Savings Simulator", "Test what-if changes. Your saved appliance settings are never modified.")

    if analysis.empty:
        st.write("")
        empty_appliance_state("simulator")
        return

    labels = {int(r.ID): f"{r.Name} ({r.Category})" for r in analysis.itertuples()}
    with card("sim_controls"):
        card_title("Simulation controls", "Pick an appliance and move the slider to change its daily operating hours.")
        c1, c2 = st.columns([1, 1.4])
        chosen = c1.selectbox("Appliance", list(labels), format_func=lambda i: labels[i], key="sim_choice")
        original_hours = float(analysis[analysis["ID"] == chosen].iloc[0]["Hours_Per_Day"])
        new_hours = c2.slider("Daily operating hours", 0.0, 24.0, original_hours, 0.25,
                              key=f"sim_hours_{chosen}_{original_hours}")

    r = simulate_scenario(analysis, chosen, new_hours, settings["tariff_per_kwh"], settings["fixed_charge"],
                          settings["monthly_budget"], settings["warning_threshold_pct"])
    ob, nb = r["original_budget"], r["new_budget"]
    saved_kwh = r["saved_kwh"]
    bill_change = r["new_bill"] - r["original_bill"]
    better = "he-green" if bill_change < -1e-9 else ("he-red" if bill_change > 1e-9 else "")

    st.write("")
    before, after = st.columns(2)
    with before:
        render(versus_card("Before", badge("none", "Current"), [
            ("Operating hours per day", f"{r['original_hours']:g} h"),
            ("Appliance monthly consumption", f"{r['original_kwh']:,.2f} kWh"),
            ("Household monthly consumption", f"{r['original_total_kwh']:,.1f} kWh"),
            ("Estimated monthly bill", money(r["original_bill"])),
            ("Budget used", f"{ob['utilization_pct']:.1f}% {badge(ob['level'], ob['status'])}" if ob["is_set"] else "No budget set"),
        ]))
    with after:
        if not nb["is_set"]:
            within = badge("none", "No budget set")
        elif nb["level"] == "exceeded":
            within = badge("exceeded", "Over budget")
        else:
            within = badge("ok", "Within budget")
        render(versus_card("After", within, [
            ("Operating hours per day", f"{r['new_hours']:g} h"),
            ("Appliance monthly consumption", f"{r['new_kwh']:,.2f} kWh"),
            ("Household monthly consumption", f"{r['new_total_kwh']:,.1f} kWh"),
            ("Estimated monthly bill", f'<span class="{better}">{money(r["new_bill"])}</span>'),
            ("Budget used", f"{nb['utilization_pct']:.1f}% {badge(nb['level'], nb['status'])}" if nb["is_set"] else "No budget set"),
        ]))

    st.write("")
    s1, s2, s3 = st.columns(3)
    cls = "positive" if saved_kwh > 1e-9 else ("negative" if saved_kwh < -1e-9 else "")
    tone = "he-green" if cls == "positive" else ("he-red" if cls == "negative" else "")
    note = "Savings" if saved_kwh >= -1e-9 else "Extra cost"
    for col, label, value, sub in (
            (s1, "Energy saved", f"{saved_kwh:,.2f} kWh", "per month. Negative means extra use."),
            (s2, f"Estimated monthly {note.lower()}", money(r["monthly_saving"]), "energy cost only"),
            (s3, f"Estimated annual {note.lower()}", money(r["annual_saving"]), "monthly × 12")):
        with col:
            render(f'<div class="he-save-card {cls}"><div class="he-metric-label">{esc(label)}</div>'
                   f'<div class="he-big {tone}" style="font-size:1.7rem;margin-top:8px">{value}</div>'
                   f'<div class="he-metric-sub">{esc(sub)}</div></div>')

    st.write("")
    alert_banner(nb)

    left, right = st.columns(2)
    with left:
        with card("sim_compare"):
            card_title("Consumption comparison", "Monthly kWh before and after the change")
            show_plot(before_after_chart(r), "sim_compare")
    with right:
        with card("sim_budget"):
            card_title("Budget comparison", "Estimated monthly bill against your budget")
            show_plot(bill_vs_budget_chart(nb["budget"], {"Current estimate": r["original_bill"],
                                                         "Simulated scenario": r["new_bill"]}), "sim_budget")
    st.caption("Savings only cover this appliance's energy cost. Fixed monthly charges stay the same.")


def page_history(ctx):
    history, analysis, summary = ctx["history"], ctx["analysis"], ctx["summary"]
    page_header("Monthly History", "Save each month's estimate to track consumption, cost and budget over time.")

    today = date.today()
    st.write("")
    with card("hist_form"):
        card_title("Save this month's summary", "Saves your current estimate (appliances, tariff and budget) as one row.")
        with st.form("history_form"):
            c1, c2, c3 = st.columns([1.2, 1, 2])
            month_name = c1.selectbox("Month", MONTH_NAMES, index=today.month - 1)
            year = c2.number_input("Year", 2000, 2100, today.year, 1)
            notes = c3.text_input("Notes (optional)", max_chars=200)
            replace = st.checkbox("Replace the record if this month is already saved")
            submitted = st.form_submit_button("Save monthly summary", type="primary")

    if submitted:
        if analysis.empty:
            notice("error", "Add at least one appliance before saving a monthly summary.")
        else:
            month = f"{int(year)}-{MONTH_NAMES.index(month_name) + 1:02d}"
            outcome = save_history_record(build_history_record(month, summary, notes), replace=replace)
            if outcome == "exists":
                notice("warning", f"{month_name} {int(year)} is already saved. Tick the replace box and save again to overwrite it.")
            else:
                flash("success", f"{month_name} {int(year)} was {'updated' if outcome == 'replaced' else 'saved'}.")
                st.rerun()

    if history.empty:
        st.write("")
        render(empty_html("calendar", "No history saved yet",
                          "Add your appliances, set your tariff and budget, then save this month's summary above. "
                          "Each saved month adds a point to your trend charts."))
        if st.button("Load sample history", key="hist_sample", icon=":material/science:"):
            added, _, error = add_sample_history()
            flash("error", error) if error else flash("success", f"Added {added} illustrative sample month(s).")
            st.rerun()
        return

    detailed = add_month_over_month(history)
    stats = history_stats(history)
    latest_change = detailed.iloc[-1]["kWh_Change_Pct"]

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render(metric_card("calendar", "Months saved", str(stats["months_saved"]), "", "In your history file", "blue"))
    with m2:
        render(metric_card("award", "Highest-consumption month", format_month(stats["highest_month"]), "",
                           f"{stats['highest_kwh']:,.1f} kWh", "orange"))
    with m3:
        render(metric_card("bolt", "Average consumption", f"{stats['average_kwh']:,.1f}", "kWh", "Per saved month", "green"))
    with m4:
        change_text = "—" if pd.isna(latest_change) else f"{latest_change:+.1f}%"
        sub = "Needs at least two months" if pd.isna(latest_change) else f"{detailed.iloc[-1]['Month_Label']} vs previous month"
        render(metric_card("trend_up", "Latest month-over-month", change_text, "", sub, "blue"))

    st.write("")
    left, right = st.columns(2)
    with left:
        with card("hist_cons"):
            card_title("Monthly consumption", "Estimated kWh per saved month")
            show_plot(consumption_trend_chart(history), "hist_cons")
    with right:
        with card("hist_cost"):
            card_title("Monthly bill", "Estimated total bill and energy cost")
            show_plot(monthly_cost_chart(history), "hist_cost")

    st.write("")
    with card("hist_budget"):
        card_title("Budget vs estimated bill", "Bars turn red when the estimated bill was above the budget")
        show_plot(budget_vs_bill_history_chart(history), "hist_budget")

    section_header("History table", "The first month has no previous month to compare against, so its change is blank.")
    with card("hist_table"):
        data_table(detailed, [
            ("Month", None, f_text("Month_Label"), "left"),
            ("Energy (kWh)", None, f_num("Total_kWh", "{:,.1f}"), "right"),
            ("kWh change", None, f_change("kWh_Change_Pct"), "right"),
            ("Energy cost", None, f_money("Energy_Cost"), "right"),
            ("Fixed charges", None, f_money("Fixed_Charges"), "right"),
            ("Est. total bill", None, f_money("Total_Bill"), "right"),
            ("Bill change", None, f_change("Bill_Change_Pct"), "right"),
            ("Budget", None, f_money("Budget"), "right"),
            ("Budget used", None, f_num("Utilization_Pct", "{:.1f}", "%"), "right"),
            ("Alert status", None, f_status(), "left"),
            ("Notes", None, f_text("Notes"), "left"),
        ], "hist_t", controls=False)

    st.write("")
    with st.expander("Delete a saved month"):
        month_choice = st.selectbox("Month", history["Month"].tolist(), format_func=format_month,
                                    key="history_delete_choice")
        confirm = st.checkbox("Yes, delete this record", key=f"history_del_{month_choice}")
        if st.button("Delete record", disabled=not confirm, key="hist_delete_btn"):
            delete_history_record(month_choice)
            flash("success", f"{format_month(month_choice)} was deleted.")
            st.rerun()


def page_import_export(ctx):
    analysis, appliances, history = ctx["analysis"], ctx["appliances"], ctx["history"]
    page_header("Import / Export", "Move appliance and history data in and out with CSV files.")
    st.write("")
    tab_down, tab_up, tab_sample = st.tabs(["Download", "Import appliances", "Sample data"])

    with tab_down:
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            with card("ex_template"):
                card_title("Template", "A ready-made CSV showing the format to import.")
                st.download_button("Download sample appliance CSV", read_sample_bytes(), "sample_appliances.csv",
                                   "text/csv", icon=":material/download:")
        with c2:
            with card("ex_mine"):
                card_title("Your appliances", "Export what you have saved.")
                if appliances.empty:
                    st.caption("Add appliances first, then you can export them here.")
                else:
                    st.download_button("Appliance list (re-importable)", to_csv_bytes(appliances[IMPORT_COLUMNS]),
                                       "my_appliances.csv", "text/csv", icon=":material/download:")
                    export_cols = ["Name", "Category", "Power_W", "Quantity", "Hours_Per_Day", "Days_Per_Month",
                                   "Daily_kWh", "Monthly_kWh", "Annual_kWh", "Share_Pct", "Monthly_Cost"]
                    st.download_button("Appliance analysis (estimates)", to_csv_bytes(analysis[export_cols]),
                                       "appliance_analysis.csv", "text/csv", icon=":material/download:")
        st.write("")
        with card("ex_hist"):
            card_title("Monthly history", "Export every saved month.")
            if history.empty:
                st.caption("No monthly history to export yet.")
            else:
                st.download_button("Monthly history CSV", to_csv_bytes(history), "monthly_history.csv",
                                   "text/csv", icon=":material/download:")

    with tab_up:
        st.write("")
        st.caption(f"Upload a CSV with these columns: {', '.join(IMPORT_COLUMNS)}")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="appliance_upload")
        if uploaded is not None:
            clean, errors, warnings = parse_appliance_csv(io.BytesIO(uploaded.getvalue()))
            show_errors(errors)
            for message in warnings[:15]:
                notice("warning", message)

            if clean is not None:
                notice("success", f"{len(clean)} valid appliance(s) found. Nothing is saved until you click Import.")
                data_table(clean, APPLIANCE_COLS, "import_preview", controls=False)
                st.write("")
                mode = st.radio("How should they be imported?",
                                ["Add to my existing appliances (skip names I already have)",
                                 "Replace all my current appliances"])
                replace_all = mode.startswith("Replace")
                confirmed = True
                if replace_all:
                    st.caption(f"You currently have {len(appliances)} appliance(s). Tip: download them first.")
                    confirmed = st.checkbox("I understand this will replace everything I have now")
                if st.button("Import appliances", disabled=not confirmed, type="primary", key="do_import"):
                    added, skipped = import_appliances(clean, replace_all=replace_all)
                    extra = f" {skipped} already in your list were skipped." if skipped else ""
                    flash("success", f"Imported {added} appliance(s).{extra}")
                    st.rerun()

    with tab_sample:
        st.write("")
        notice("warning", "The sample data is illustrative only. It is not a real measurement of any household.")
        c1, c2, _ = st.columns([1.3, 1.5, 2])
        if c1.button("Add sample appliances", icon=":material/science:"):
            added, skipped, error = add_sample_appliances()
            if error:
                flash("error", error)
            else:
                flash("success", f"Added {added} sample appliance(s); {skipped} skipped because the name already existed.")
            st.rerun()
        if c2.button("Add sample monthly history", icon=":material/science:"):
            added, skipped, error = add_sample_history()
            if error:
                flash("error", error)
            else:
                flash("success", f"Added {added} sample month(s); {skipped} skipped because you already had them.")
            st.rerun()


def page_settings(ctx):
    settings = ctx["settings"]
    page_header("Settings", "Tariff, fixed charge, monthly budget and the budget warning threshold.")
    st.write("")
    with card("settings_form"):
        card_title("Billing and budget", "These values are saved to data/settings.csv and used on every page.")
        with st.form("settings_form"):
            c1, c2 = st.columns(2)
            tariff = c1.number_input("Electricity tariff (₹ per kWh)", 0.0, 1000.0, float(settings["tariff_per_kwh"]),
                                     0.1, format="%.2f")
            fixed = c2.number_input("Fixed monthly charges (₹)", 0.0, 100000.0, float(settings["fixed_charge"]),
                                    10.0, format="%.2f")
            budget = c1.number_input("Monthly budget (₹, 0 turns alerts off)", 0.0, 10000000.0,
                                     float(settings["monthly_budget"]), 100.0, format="%.2f")
            threshold = c2.slider("Budget warning threshold (%)", 10, 99,
                                  int(round(clamp(settings["warning_threshold_pct"], 10, 99))))
            saved = st.form_submit_button("Save settings", type="primary")
        if saved:
            errors = update_settings({"tariff_per_kwh": tariff, "fixed_charge": fixed, "monthly_budget": budget,
                                      "warning_threshold_pct": float(threshold)})
            if errors:
                show_errors(errors)
            else:
                flash("success", "Settings saved.")
                st.rerun()

    section_header("Where your data is saved")
    rows = "".join(
        f'<div class="he-row"><span>{esc(p.name)}</span><span>{"Found" if p.exists() else "Not created yet"}</span></div>'
        for p in (APPLIANCE_FILE, HISTORY_FILE, SETTINGS_FILE))
    render(f'<div class="he-card">{rows}<div class="he-big-sub" style="margin-top:8px">Folder: {esc(DATA_DIR)}. '
           f'These files are created automatically. Sample data is built into the app and is illustrative only.</div></div>')

    section_header("About HomeEnergy")
    with card("about"):
        st.markdown("**HomeEnergy** estimates a household's electricity use and bill from appliance ratings, tracks it "
                    "against a monthly budget and lets you test what-if savings. It is a Data Science mini-project built "
                    "with Python, Pandas, NumPy, Plotly and Streamlit, using plain CSV files for storage "
                    "(no machine learning and no database).")
        st.code("""Daily energy (kWh)   = (Power W x Quantity x Hours per day) / 1000
Monthly energy (kWh) = (Power W x Quantity x Hours per day x Days per month) / 1000
Energy cost          = Monthly kWh x Tariff per kWh
Estimated total bill = Energy cost + Fixed monthly charges
Remaining budget     = Budget - Estimated total bill
Budget used (%)      = (Estimated total bill / Budget) x 100
Annual values        = Monthly values x 12""", language="text")
        st.markdown("**Alert levels:** Normal below the warning threshold (default 80%), Budget warning from the "
                    "threshold up to just under 100%, Budget exceeded at 100% or more, and Budget not set when the "
                    "budget is zero.")
        st.markdown("**Limitations:** results are estimates from rated power, not measurements. Appliances with duty "
                    "cycles (fridge, AC, geyser) need realistic running hours. A single flat tariff is used, while real "
                    "bills may have slabs, taxes and surcharges. Annual figures assume every month is similar.")


# =============================================================================
# 8. MAIN: load data, build the sidebar, show the chosen page
# =============================================================================

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]

PAGES = {
    "Dashboard": (":material/dashboard:", page_dashboard),
    "Appliances": (":material/power:", page_appliances),
    "Consumption Analytics": (":material/monitoring:", page_analytics),
    "Bill Estimator": (":material/receipt_long:", page_bill),
    "Budget & Alerts": (":material/notifications:", page_budget),
    "Savings Simulator": (":material/savings:", page_simulator),
    "Monthly History": (":material/calendar_month:", page_history),
    "Import / Export": (":material/swap_vert:", page_import_export),
    "Settings": (":material/settings:", page_settings),
}


def main():
    ensure_data_files()
    if ensure_dark_theme_file():
        st.session_state["theme_notice"] = True
    st.markdown(CSS, unsafe_allow_html=True)

    # load everything fresh on each run; the files are tiny
    appliances = load_appliances()
    settings = load_settings()
    history = load_history()
    analysis = add_consumption_columns(appliances, settings["tariff_per_kwh"])
    summary = summarize(analysis, settings["tariff_per_kwh"], settings["fixed_charge"],
                        settings["monthly_budget"], settings["warning_threshold_pct"])
    ctx = {"appliances": appliances, "settings": settings, "history": history,
           "analysis": analysis, "summary": summary}

    if "_goto" in st.session_state:  # a button asked to switch pages
        st.session_state["nav"] = st.session_state.pop("_goto")

    with st.sidebar:
        render(f'<div class="he-brand"><div class="he-brand-icon">{icon("bolt", 22)}</div><div>'
               f'<div class="he-brand-name">HomeEnergy</div><div class="he-brand-sub">Smart Energy Management</div></div></div>')
        page = st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed",
                        format_func=lambda name: f"{PAGES[name][0]} {name}")
        render('<div class="he-side-note">All results are estimates from rated power, not your actual electricity bill.</div>')
        if st.session_state.get("theme_notice"):
            st.caption("Dark theme file saved. Restart Streamlit once so dropdowns and sliders match.")

    show_flash()
    PAGES[page][1](ctx)


main()
