#!/usr/bin/env python3
"""Regenerate all slide visuals for the soil-moisture EDA deck."""
from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.signal import savgol_filter

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 18,
    }
)

BASE_DIR = Path("presentation/eda_story/images")
BASE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path("Archive/processed_data.db")


# -- helper plotters -----------------------------------------------------

def equation_card(text: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(
        0.5,
        0.6,
        text,
        ha="center",
        va="center",
        color="#f5f8fc",
        fontsize=36,
        bbox=dict(facecolor="#12222f", edgecolor="none", pad=20, alpha=0.95),
    )
    fig.patch.set_facecolor("#08141c")
    fig.savefig(BASE_DIR / filename, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def sensor_pipeline_placeholder() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    labels = [
        "Sensor array\n(Insert photo)",
        "LoRaWAN gateway\n(Insert photo)",
        "Processing scripts\n(Insert graphic)",
        "SQLite database\n(Insert screenshot)",
    ]
    xpos = [0.05, 0.32, 0.59, 0.86]
    for x, color, label in zip(xpos, colors, labels):
        patch = FancyBboxPatch((x, 0.45), 0.18, 0.25, boxstyle="round,pad=0.02", linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(patch)
        ax.text(x + 0.09, 0.575, label, ha="center", va="center", fontsize=16, color=color)
    for left, right in zip(xpos, xpos[1:]):
        ax.annotate("", xy=(right, 0.575), xytext=(left + 0.18, 0.575), arrowprops=dict(arrowstyle="->", linewidth=2, color="#999"))
    ax.text(0.5, 0.82, "Replace with actual photos & diagram", ha="center", color="#555", fontsize=18)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "sensor_pipeline_placeholder.png", dpi=200)
    plt.close(fig)


def data_audit_card() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT TIMESTAMP, VWC_06, VWC_18, VWC_30, irrigation "
        "FROM data_table WHERE plot_number=2003 ORDER BY TIMESTAMP LIMIT 5",
        conn,
    )
    conn.close()
    df_raw = df.copy()
    df_clean = (
        df.set_index("TIMESTAMP")
        .resample("D")
        .mean()
        .dropna()
        .head(5)
        .assign(day_sin=lambda d: np.sin(np.linspace(0, 2 * np.pi, len(d))))
        .assign(precip_irrig=lambda d: 0.0)
        .reset_index()
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for ax in (ax1, ax2):
        ax.axis("off")
    ax1.set_title("Raw ingest sample", color="#1f77b4", pad=12)
    table1 = ax1.table(cellText=df_raw.values, colLabels=df_raw.columns, loc="center")
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    table1.scale(1.1, 1.5)
    ax2.set_title("Clean extract sample", color="#2ca02c", pad=12)
    table2 = ax2.table(cellText=df_clean.values, colLabels=df_clean.columns, loc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(1.1, 1.5)
    fig.suptitle("Replace with actual screenshots when available", color="#555")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "data_audit_card.png", dpi=200)
    plt.close(fig)


def vwc_cleaning_plot(plot_number: int = 2003) -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT TIMESTAMP, VWC_06 FROM data_table WHERE plot_number=?",
        conn,
        params=(plot_number,),
    )
    conn.close()
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.set_index("TIMESTAMP").sort_index()
    raw = df["VWC_06"]
    daily = raw.resample("D").mean()
    sg = savgol_filter(daily.interpolate(method="pchip"), window_length=21, polyorder=3)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(raw.index, raw.values, color="lightgray", linewidth=1.5, label="Hourly VWC_06")
    ax.plot(daily.index, daily.values, color="#1f77b4", marker="o", linewidth=2.5, label="Daily mean")
    ax.plot(daily.index, sg, color="#d62728", linewidth=3, label="Savitzky-Golay (21,3)")
    ax.set_title(f"Plot {plot_number} smoothing example (6\" depth)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volumetric Water Content (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "vwc_cleaning_highlight.png", dpi=200)
    plt.close(fig)


def irrigation_feature_stack() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT TIMESTAMP, precip_irrig FROM data_table WHERE plot_number=2003",
        conn,
    )
    conn.close()
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.set_index("TIMESTAMP").sort_index()
    daily = df["precip_irrig"].resample("D").sum().loc["2023-07-20":"2023-09-03"]
    log_precip = np.log1p(daily)
    rolling = daily.rolling(7, min_periods=1).sum()
    flag = (daily > 0).astype(int)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].bar(daily.index, daily.values, color="#1f77b4")
    axes[0].set_ylabel("precip")
    axes[0].grid(alpha=0.2)
    axes[0].set_title("Irrigation + rainfall signals")
    axes[1].plot(log_precip.index, log_precip.values, color="#ff7f0e", linewidth=3)
    axes[1].set_ylabel("log(precip+1)")
    axes[1].grid(alpha=0.2)
    axes[2].plot(rolling.index, rolling.values, color="#2ca02c", linewidth=3)
    axes[2].set_ylabel("7-day sum")
    axes[2].grid(alpha=0.2)
    axes[3].step(daily.index, flag.values, where="mid", color="#9467bd", linewidth=3)
    axes[3].set_ylabel("flag")
    axes[3].set_xlabel("Date")
    axes[3].set_yticks([0, 1])
    axes[3].grid(alpha=0.2)
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "irrigation_feature_stack.png", dpi=200)
    plt.close(fig)


def feature_story() -> None:
    kept = {
        "Soil moisture layers": ["VWC_06", "VWC_18", "VWC_30", "ΔVWC derivatives"],
        "Temporal encoding": ["day/hour sine & cosine", "day-of-week sine & cosine"],
        "Weather load": [
            "Ta avg/max/min",
            "RH avg/max/min",
            "Solar_2m_Avg",
            "WndAveSpd_3m",
            "canopy_temp",
            "HeatIndex",
        ],
        "Irrigation context": [
            "precip_irrig",
            "log(precip+1)",
            "7-day cumulative",
            "time-since thresholds",
            "irrigation flag",
        ],
    }
    removed = {
        "Unreliable depth": ["VWC_42 flatlines"],
        "Sparse agronomy": ["crop label", "growth_stage"],
        "Low coverage stress": ["daily_et", "CWSI", "SWSI"],
        "Redundant": ["Elevation", "Unnamed index columns"],
    }
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_title("Feature story: kept vs documented cuts", fontsize=24, color="#1f4e5f", pad=20)
    x0, y0 = 0.05, 0.8
    box_w, box_h = 0.42, 0.15
    for idx, (category, items) in enumerate(kept.items()):
        y = y0 - idx * (box_h + 0.05)
        patch = FancyBboxPatch((x0, y), box_w, box_h, boxstyle="round,pad=0.02", linewidth=2, edgecolor="#1f4e5f", facecolor="#e8f5ff")
        ax.add_patch(patch)
        ax.text(x0 + box_w / 2, y + box_h * 0.7, category, ha="center", fontsize=18, color="#1f4e5f")
        ax.text(x0 + box_w / 2, y + box_h * 0.35, " \n".join(items), ha="center", fontsize=13, color="#1f4e5f")
    x1, y1 = 0.55, 0.8
    for idx, (category, items) in enumerate(removed.items()):
        y = y1 - idx * (0.13 + 0.05)
        patch = FancyBboxPatch((x1, y), 0.38, 0.13, boxstyle="round,pad=0.02", linewidth=2, edgecolor="#d62728", facecolor="#ffeaea")
        ax.add_patch(patch)
        ax.text(x1 + 0.19, y + 0.09, category, ha="center", fontsize=17, color="#b22222")
        ax.text(x1 + 0.19, y + 0.045, " \n".join(items), ha="center", fontsize=13, color="#b22222")
    fig.savefig(BASE_DIR / "feature_story.png", dpi=200)
    plt.close(fig)


def sliding_window_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    start_x = 0.08
    width = 0.05
    spacing = 0.01
    for i in range(7):
        x = start_x + i * (width + spacing)
        rect = Rectangle((x, 0.6), width, 0.12, facecolor="#1f77b4", alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + width / 2, 0.66, f"Day {i+1}", ha="center", va="center", fontsize=14, color="white")
    ax.text(start_x + 3 * (width + spacing), 0.8, "168 hourly inputs (7 days)", ha="center", color="#1f77b4", fontsize=20)
    ax.annotate("", xy=(0.45, 0.66), xytext=(0.42, 0.66), arrowprops=dict(arrowstyle="->", linewidth=2, color="#555"))
    ax.text(0.435, 0.72, "sliding window", fontsize=16, color="#555")
    ax.add_patch(Rectangle((0.45, 0.6), 0.12, 0.12, facecolor="#ff7f0e", alpha=0.85))
    ax.text(0.51, 0.66, "LSTM stack\n(512→64)", ha="center", va="center", fontsize=15, color="white")
    ax.annotate("", xy=(0.62, 0.66), xytext=(0.57, 0.66), arrowprops=dict(arrowstyle="->", linewidth=2, color="#555"))
    ax.text(0.6, 0.72, "forecast", fontsize=16, color="#555")
    for i in range(4):
        x = 0.63 + i * (width + spacing)
        rect = Rectangle((x, 0.6), width, 0.12, facecolor="#2ca02c", alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + width / 2, 0.66, f"Day +{i+1}", ha="center", va="center", fontsize=14, color="white")
    ax.text(0.63 + 1.5 * (width + spacing), 0.8, "96 hour forecast horizon", ha="center", color="#2ca02c", fontsize=20)
    ax.text(0.5, 0.4, "window slides forward to build each training pair", ha="center", fontsize=18, color="#555")
    fig.savefig(BASE_DIR / "sliding_window_diagram.png", dpi=200)
    plt.close(fig)


def regenerate_all() -> None:
    sensor_pipeline_placeholder()
    data_audit_card()
    vwc_cleaning_plot()
    irrigation_feature_stack()
    feature_story()
    sliding_window_diagram()
    equation_card(r"$\\tilde{v}_d(t) = \\mathrm{SG}_{21,3}(v_d(t))$", "eq_savgol.png")
    equation_card(r"$\\Delta v_d(t) = v_d(t) - v_d(t-1)$", "eq_derivative.png")
    equation_card(
        r"$p_{\\log}(t) = \\log(\\text{precip\\_irrig}(t) + 1)$" "\\n"
        r"$P_{7\\,\\text{day}}(t) = \\sum_{k=0}^{6} \\text{precip\\_irrig}(t-k)$" "\\n"
        r"$\\tau_{\\theta}(t) = \\min\\{k \\ge 0 \\mid \\text{precip\\_irrig}(t-k) > \\theta\\}$",
        "eq_irrigation.png",
    )
    print("Images regenerated in", BASE_DIR.resolve())


if __name__ == "__main__":
    regenerate_all()
