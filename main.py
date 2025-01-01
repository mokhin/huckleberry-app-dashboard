import logging
from datetime import datetime

import altair as alt
import polars as pl
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up Altair
alt.data_transformers.enable("vegafusion")
alt.themes.enable("vox")

st.set_page_config(
    page_title="Baby Stat",
    page_icon=":baby:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Constants
CONSTANT_DATE = datetime(1, 1, 1)
RAW_EXAMPLE_CSV_FILE = "data/example.csv"
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 6
DATE_FORMAT = "%Y-%m-%d %H:%M"


# Auxiliary functions
def set_constant_date(dt: datetime) -> datetime:
    return datetime(
        CONSTANT_DATE.year,
        CONSTANT_DATE.month,
        CONSTANT_DATE.day,
        dt.hour,
        dt.minute,
        dt.second,
        dt.microsecond,
    )


# Function to load and process data
@st.cache_data
def load_and_process_data(file_path: str) -> pl.DataFrame:
    try:
        df = (
            pl.read_csv(file_path)
            .with_columns(
                pl.col("Start").str.to_datetime(DATE_FORMAT),
                pl.col("End").str.to_datetime(DATE_FORMAT),
            )
            .with_columns(
                pl.col("Start").dt.date().alias("StartDate"),
                pl.col("End").dt.date().alias("EndDate"),
            )
            .with_columns((pl.col("End") - pl.col("Start")).alias("Duration"))
            .with_columns(
                pl.col("Start").map_elements(
                    set_constant_date, return_dtype=pl.Datetime
                ),
                pl.col("End").map_elements(set_constant_date, return_dtype=pl.Datetime),
            )
            .filter(pl.col("StartDate") > pl.col("StartDate").min())
            .filter(
                (pl.col("EndDate") < pl.col("EndDate").max())
                | pl.col("EndDate").is_null()
            )
        )
        logging.info(f"Successfully read data from {file_path}")
    except Exception as e:
        logging.error(f"Error reading data from {file_path}: {e}")
        raise

    # Add day/night column
    df = df.with_columns(
        pl.when(
            (pl.col("Start").dt.hour() >= NIGHT_START_HOUR)
            | (pl.col("Start").dt.hour() < NIGHT_END_HOUR)
        )
        .then(pl.lit("Night"))
        .otherwise(pl.lit("Day"))
        .alias("DayOrNight"),
    )
    return df


# Dashboard

# Title
st.title("Baby Stat :baby:")

# Create columns for first row
col1, col2 = st.columns(2)

# Select file to analyze
col1.markdown(
    "### Get file from [Huckleberry App](https://huckleberrycare.com/) and put it here:"
)
raw_csv_file = col1.file_uploader("", type=["csv"])

if raw_csv_file is None:
    raw_csv_file = RAW_EXAMPLE_CSV_FILE

df = load_and_process_data(raw_csv_file)

col2.markdown("### Select start date:")
start_date = col2.date_input("", df.select("StartDate").min().item())

st.markdown("# Overview for selected period")

# Create columns for second row
col1, col2, col3, col4 = st.columns(4)

# Filter data
df = df.filter(pl.col("StartDate") >= start_date)
growth_df = (
    df.filter(pl.col("Type") == "Growth")
    .rename(
        {
            "Start Condition": "Weight",
            "Start Location": "Height",
            "End Condition": "HeadCircumference",
        }
    )
    .with_columns(pl.col("Weight").str.replace("kg", "").cast(pl.Float64))
    .with_columns(pl.col("Height").str.replace("cm", "").cast(pl.Float64))
    .with_columns(pl.col("HeadCircumference").str.replace("cm", "").cast(pl.Float64))
    .select(["StartDate", "Weight", "Height", "HeadCircumference"])
)

col1.metric(
    label="Sleep hours per day",
    value=(
        round(
            df.filter(pl.col("Type") == "Sleep")
            .group_by("StartDate")
            .agg(pl.col("Duration").sum().cast(pl.Float64).mul(1 / (60 * 60 * 1e6)))
            .select("Duration")
            .mean()
            .item(),
            1,
        )
    ),
)

col2.metric(
    label="Sleeps per day",
    value=(
        round(
            df.filter(pl.col("Type") == "Sleep")
            .group_by("StartDate")
            .len()
            .select("len")
            .mean()
            .item(),
            1,
        )
    ),
)

col3.metric(
    label="% of night sleep hours",
    value=(
        round(
            df.filter(pl.col("Type") == "Sleep")
            .group_by("StartDate")
            .agg(
                (
                    (
                        pl.col("Duration")
                        * (pl.col("DayOrNight") == "Night").cast(pl.Float64)
                    ).sum()
                    / pl.col("Duration").sum()
                ).alias("ShareOfNightSleeps")
            )
            .select("ShareOfNightSleeps")
            .mean()
            .item()
            * 100,
            1,
        )
    ),
)

col4.metric(
    label="Feeds per day",
    value=(
        round(
            df.filter(pl.col("StartDate") > df.select("StartDate").min().item())
            .filter(pl.col("StartDate") < df.select("StartDate").max().item())
            .filter(pl.col("Type") == "Feed")
            .group_by("StartDate")
            .len()
            .select("len")
            .mean()
            .item(),
            1,
        )
    ),
)


st.markdown("# Sleeps")

# Plot 1
st.altair_chart(
    alt.Chart(
        data=(
            df.filter(pl.col("Type") == "Sleep")
            .group_by("StartDate")
            .agg(pl.col("Duration").sum().cast(pl.Float64).mul(1 / (60 * 60 * 1e6)))
        ),
        title=alt.Title("Sleep Hours"),
    )
    .mark_line()
    .encode(
        x=alt.X("StartDate:T", title=""),
        y=alt.Y("Duration:Q", title="hours/day", scale=alt.Scale(zero=False)),
    )
    .configure_axisY(
        titleAngle=0,
        titleY=-10,
        titleX=-22,
    )
    .interactive(bind_y=False),
    use_container_width=True,
)


# Plot 2
# Average sleep duration for Day and Night
st.altair_chart(
    alt.Chart(
        data=(
            df.filter(pl.col("Type") == "Sleep")
            .group_by(["StartDate", "DayOrNight"])
            .agg(pl.col("Duration").mean().cast(pl.Float64).mul(1 / (60 * 60 * 1e6)))
        ),
        title=alt.Title("Average sleep duration for Day and Night"),
    )
    .mark_line()
    .encode(
        x=alt.X("StartDate", title=""),
        y=alt.Y(
            "Duration",
            title="hours/sleep",
            scale=alt.Scale(zero=False),
        ),
        color=alt.Color("DayOrNight", legend=alt.Legend(orient="top", title=None)),
    )
    .configure_axisY(
        titleAngle=0,
        titleY=-10,
        titleX=-22,
    )
    .interactive(bind_y=False),
    use_container_width=True,
)

# Plot 3
# Share of night sleeps
st.altair_chart(
    alt.Chart(
        data=(
            df.filter(pl.col("Type") == "Sleep")
            .group_by("StartDate")
            .agg(
                (
                    (
                        pl.col("Duration")
                        * (pl.col("DayOrNight") == "Night").cast(pl.Float64)
                    ).sum()
                    / pl.col("Duration").sum()
                ).alias("ShareOfNightSleeps")
            )
        ),
        title=alt.Title("Share of night sleeps"),
    )
    .mark_line()
    .encode(
        x=alt.X("StartDate", title=""),
        y=alt.Y(
            "ShareOfNightSleeps",
            title="share",
            axis=alt.Axis(format="%"),
            scale=alt.Scale(zero=False),
        ),
    )
    .configure_axisY(
        titleAngle=0,
        titleY=-10,
        titleX=-22,
    )
    .interactive(bind_y=False),
    use_container_width=True,
)

st.markdown("# Feedings")

# Plot 4
# Feeds per day
st.altair_chart(
    alt.Chart(
        data=(df.filter(pl.col("Type") == "Feed").group_by("StartDate").len()),
        title=alt.Title("Feeds per day"),
    )
    .mark_line()
    .encode(
        x=alt.X("StartDate", title=""),
        y=alt.Y("len", title="feeds/day", scale=alt.Scale(zero=False)),
    )
    .configure_axisY(
        titleAngle=0,
        titleY=-10,
        titleX=-22,
    )
    .interactive(bind_y=False),
    use_container_width=True,
)

st.markdown("# Growth")

# Plot 5
# Weight
st.altair_chart(
    alt.Chart(
        data=growth_df,
        title=alt.Title("Weight"),
    )
    .mark_line(interpolate="step-after")
    .encode(
        x=alt.X("StartDate", title=""),
        y=alt.Y("Weight", title="kg", scale=alt.Scale(zero=False)),
    )
    .configure_axisY(
        titleAngle=0,
        titleY=-10,
        titleX=-22,
    )
    .interactive(bind_y=False),
    use_container_width=True,
)

# Plot 6
# Height
st.altair_chart(
    alt.Chart(
        data=growth_df,
        title=alt.Title("Height"),
    )
    .mark_line(interpolate="step-after")
    .encode(
        x=alt.X("StartDate", title=""),
        y=alt.Y("Height", title="cm", scale=alt.Scale(zero=False)),
    )
    .configure_axisY(
        titleAngle=0,
        titleY=-10,
        titleX=-22,
    )
    .interactive(bind_y=False),
    use_container_width=True,
)

# Plot 7
# Head Circumference
st.altair_chart(
    alt.Chart(
        data=growth_df,
        title=alt.Title("Head Circumference"),
    )
    .mark_line(interpolate="step-after")
    .encode(
        x=alt.X("StartDate", title=""),
        y=alt.Y("HeadCircumference", title="cm", scale=alt.Scale(zero=False)),
    )
    .configure_axisY(
        titleAngle=0,
        titleY=-10,
        titleX=-22,
    )
    .interactive(bind_y=False),
    use_container_width=True,
)
