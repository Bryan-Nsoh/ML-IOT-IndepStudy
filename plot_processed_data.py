import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

# Current date and end date
current_date = datetime.now().strftime("%Y-%m-%d")
end_date = "2023-07-20"


# New HTML directory
html_dir = "./html_plots"
if not os.path.exists(html_dir):
    os.makedirs(html_dir)

# Custom color palette to make the color scheme fit better with Nebraska colors.
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# Function to check and clean data for a specific date range and plot number
def clean_data(
    plot_num, start_date="2023-07-20", end_date=datetime.now().strftime("%Y-%m-%d")
):
    # SQLite connection
    conn = sqlite3.connect("./kalman_interpolated_data.db")

    # SQL query
    query = f"""
    SELECT TIMESTAMP, VWC_06, VWC_18, VWC_30, VWC_42, canopy_temp, Rain_1m_Tot, "CWSI", "daily_et", "HeatIndex_2m_Avg", "SWSI"
    FROM kalman_interpolated_data 
    WHERE TIMESTAMP BETWEEN '{start_date}' AND '{end_date}' AND plot_number = '{plot_num}'
    ORDER BY TIMESTAMP
    """

    # Fetch data and close connection
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert TIMESTAMP to datetime and 'None' in Rain_1m_Tot to NaN
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df["Rain_1m_Tot"] = df["Rain_1m_Tot"].replace("None", None).astype(float)

    # Replace negative values with NaN
    df[df.select_dtypes(include=["float64"]).columns] = df.select_dtypes(
        include=["float64"]
    ).apply(lambda x: x.where(x >= 0))

    # Replace VWC values of 0 with NaN
    vwc_columns = ["VWC_06", "VWC_18", "VWC_30", "VWC_42"]
    df[vwc_columns] = df[vwc_columns].replace(0, np.nan)

    # Remove duplicate TIMESTAMPs and keep non-null entries
    df_cleaned = (
        df.dropna(
            subset=[
                "VWC_06",
                "VWC_18",
                "VWC_30",
                "VWC_42",
                "canopy_temp",
                "Rain_1m_Tot",
                "CWSI",
                "daily_et",
                "HeatIndex_2m_Avg",
                "SWSI",
            ],
            how="all",
        )
        .groupby("TIMESTAMP")
        .apply(lambda x: x.dropna().head(1) if not x.dropna().empty else x.head(1))
        .reset_index(drop=True)
    )

    return df_cleaned


def interactive_plotly_plot(df_data, plot_num):
    if df_data["TIMESTAMP"].dt.tz is None:
        df_data["TIMESTAMP"] = df_data["TIMESTAMP"].dt.tz_localize("UTC")
    if df_data["TIMESTAMP"].dt.tz != "US/Central":
        df_data["TIMESTAMP"] = df_data["TIMESTAMP"].dt.tz_convert("US/Central")

    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=False,
    )

    traces = [
        go.Scatter(
            x=df_data["TIMESTAMP"],
            y=df_data[column],
            name=f"{column} (inches)",
            line=dict(width=3, color=colors[i]),
        )
        for i, column in enumerate(["VWC_06", "VWC_18", "VWC_30", "VWC_42"])
    ]
    for trace in traces:
        fig.add_trace(trace, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=df_data["TIMESTAMP"],
            y=df_data["canopy_temp"],
            name="Measured Canopy Temp (°C)",
            line=dict(width=3, color=colors[-2]),
        ),
        row=3,
        col=1,
    )

    # Modified to include Rain in one plot
    fig.add_trace(
        go.Scatter(
            x=df_data["TIMESTAMP"],
            y=df_data["Rain_1m_Tot"],
            name="Rain 1m Total (mm)",
            line=dict(width=3, color=colors[-1]),
        ),
        row=2,
        col=1,
    )

    # Data for the first subplot of column 2 (combining CWSI and SWSI)
    for data in ["CWSI", "SWSI"]:
        fig.add_trace(
            go.Scatter(
                x=df_data["TIMESTAMP"],
                y=df_data[data],
                name=data,
                line=dict(width=3),
            ),
            row=2,
            col=2,
        )

    # Data for the second subplot of column 2 (ET)
    fig.add_trace(
        go.Scatter(
            x=df_data["TIMESTAMP"],
            y=df_data["daily_et"],
            name="daily_et",
            line=dict(width=3),
        ),
        row=3,
        col=2,
    )

    # Data for the third subplot of column 2 (Heat Index)
    fig.add_trace(
        go.Scatter(
            x=df_data["TIMESTAMP"],
            y=df_data["HeatIndex_2m_Avg"],
            name="HeatIndex_2m_Avg",
            line=dict(width=3),
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=f"Data for Plot {plot_num}",
        # Left Side
        yaxis=dict(domain=[0.67, 1], title="% VWC"),  # Top: VWC plots
        yaxis5=dict(
            domain=[0.34, 0.66], title="Canopy Temp (°C)"
        ),  # Middle: Measured canopy temp
        yaxis3=dict(domain=[0, 0.33], title="Rain (mm)"),  # Bottom: Rain
        # Right Side
        yaxis4=dict(domain=[0.67, 1], title="CWSI & SWSI"),  # Top: CWSI & SWSI
        yaxis2=dict(domain=[0.34, 0.66], title="daily ET (mm)"),  # Middle: daily ET
        yaxis6=dict(domain=[0, 0.33], title="Heat Index"),  # Bottom: Heat Index
        margin=dict(l=50, r=20, t=50, b=20),
        font=dict(size=18),
    )

    # Save as HTML
    fig.write_html(html_dir + f"/{plot_num}.html")
    return html_dir + f"/{plot_num}.html"


for plot_num in [
    2001,
    2009,
    5014,
    2014,
    2007,
    5005,
    5010,
    2015,
    2013,
    2004,
    2003,
    2010,
    5009,
    5001,
    5013,
    5015,
    5006,
    5003,
]:
    df_data = clean_data(plot_num, start_date="2023-07-20", end_date="2023-09-07")
    html_path = interactive_plotly_plot(df_data, plot_num)
