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
NIGHT_START_HOUR = 18
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
# @st.cache_data
def load_and_process_data(file_path: str) -> pl.DataFrame:
    try:
        df = (
            pl.read_csv(file_path)
            .with_columns(
                pl.col("Start").str.to_datetime(DATE_FORMAT),
                pl.col("End").str.to_datetime(DATE_FORMAT),
            )
            .with_columns(
                pl.datetime(
                    pl.col("Start").dt.year(),
                    pl.col("Start").dt.month(),
                    pl.col("Start").dt.day(),
                    NIGHT_START_HOUR,
                    0,
                    0,
                ).alias("NightStart"),
            )
            .with_columns(
                pl.col("Start").dt.date().alias("StartDate"),
                pl.col("End").dt.date().alias("EndDate"),
            )
            .with_columns((pl.col("End") - pl.col("Start")).alias("Duration"))
            .with_columns(
                pl.col("Start")
                .map_elements(set_constant_date, return_dtype=pl.Datetime)
                .alias("StartTime"),
                pl.col("End")
                .map_elements(set_constant_date, return_dtype=pl.Datetime)
                .alias("EndTime"),
            )
            # Add NightStart timestamp - take StartDate and add NIGHT_START_HOUR
            .with_columns(
                (pl.col("Start") + (pl.col("End") - pl.col("Start")) / 2).alias(
                    "MiddlePoint"
                )
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
            (pl.col("MiddlePoint").dt.hour() >= NIGHT_START_HOUR)
            | (pl.col("MiddlePoint").dt.hour() < NIGHT_END_HOUR)
        )
        .then(pl.lit("Night"))
        .otherwise(pl.lit("Day"))
        .alias("DayOrNight"),
    )

    # Add NightDay column if StartTime between 18:00 and 00:00 then = StartDate, else = StartDate + 1
    df = df.with_columns(
        pl.when(pl.col("StartTime").dt.hour() >= NIGHT_START_HOUR)
        .then(pl.col("StartDate"))
        .otherwise(pl.col("StartDate") + pl.duration(days=1))
        .alias("NightDay"),
    )

    return df


def create_data_for_gantt(df: pl.DataFrame) -> pl.DataFrame:
    constant_date = datetime(1, 1, 1)

    one_day_df = df.filter(pl.col("StartDate") == pl.col("EndDate")).with_columns(
        pl.col("StartDate").alias("Date")
    )
    two_days_df = df.filter(pl.col("StartDate") != pl.col("EndDate"))

    two_days_part_1_df = two_days_df.with_columns(
        pl.lit(
            datetime(
                constant_date.year,
                constant_date.month,
                constant_date.day,
                23,
                59,
                59,
                0,
            )
        ).alias("EndTime"),
        pl.col("StartDate").alias("Date"),
    )

    two_days_part_2_df = two_days_df.with_columns(
        pl.lit(
            datetime(
                constant_date.year, constant_date.month, constant_date.day, 0, 0, 0, 0
            )
        ).alias("StartTime"),
        pl.col("EndDate").alias("Date"),
    )

    return pl.concat([one_day_df, two_days_part_1_df, two_days_part_2_df]).select(
        ["Type", "StartTime", "EndTime", "Date"]
    )


# Calculate StartNights with the longest average sleep duration
def create_best_days_df(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("Type") == "Sleep")
        .filter(pl.col("DayOrNight") == "Night")
        .group_by("NightDay")
        .agg(pl.col("Duration").mean())
        .sort("Duration", descending=True, nulls_last=True)
        .select(["NightDay", "Duration"])
        .rename({"NightDay": "Date"})
        .head(10)
    )


def create_worst_days_df(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("Type") == "Sleep")
        .filter(pl.col("DayOrNight") == "Night")
        .group_by("NightDay")
        .agg(pl.col("Duration").mean())
        .sort("Duration", nulls_last=True)
        .select(["NightDay", "Duration"])
        .rename({"NightDay": "Date"})
        .head(10)
    )


def create_worst_days_stat_df(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("Type") == "Sleep")
        .filter(pl.col("DayOrNight") == "Day")
        .group_by("StartDate")
        .agg(
            # Col1
            pl.col("Duration")
            .sum()
            .cast(pl.Float64)
            .mul(1 / (60 * 60 * 1e6))
            .round(1)
            .alias("Day Sleep Hours"),
            # Col2
            pl.col("Duration").len().alias("Day Naps"),
            # Col3
            pl.col("Duration")
            .mean()
            .cast(pl.Float64)
            .mul(1 / (60 * 60 * 1e6))
            .round(1)
            .alias("Hours per Nap"),
        )
        .rename({"StartDate": "Date"})
        .join(
            worst_days_df.with_columns(
                pl.col("Duration").cast(pl.Float32).mul(1 / (60 * 60 * 1e6)).round(2)
            ).rename({"Duration": "Next Day Average Sleep Duration"}),
            on="Date",
            how="inner",
        )
        .with_columns(pl.col("Date").cast(pl.String))
        .sort(["Date"], nulls_last=True)
    )


def create_best_days_stat_df(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("Type") == "Sleep")
        .filter(pl.col("DayOrNight") == "Day")
        .group_by("StartDate")
        .agg(
            # Col1
            pl.col("Duration")
            .sum()
            .cast(pl.Float64)
            .mul(1 / (60 * 60 * 1e6))
            .round(1)
            .alias("Day Sleep Hours"),
            # Col2
            pl.col("Duration").len().alias("Day Naps"),
            # Col3
            pl.col("Duration")
            .mean()
            .cast(pl.Float64)
            .mul(1 / (60 * 60 * 1e6))
            .round(1)
            .alias("Hours per Nap"),
        )
        .rename({"StartDate": "Date"})
        .join(
            best_days_df.with_columns(
                pl.col("Duration").cast(pl.Float32).mul(1 / (60 * 60 * 1e6)).round(2)
            ).rename({"Duration": "Next Night Average Sleep Duration"}),
            on="Date",
            how="inner",
        )
        .with_columns(pl.col("Date").cast(pl.String))
        .sort(["Date"], nulls_last=True)
    )


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

gantt_df = create_data_for_gantt(df)

col1.metric(
    label="Sleep hours per day",
    value=(
        int(
            round(
                df.filter(pl.col("Type") == "Sleep")
                .group_by("StartDate")
                .agg(pl.col("Duration").sum().cast(pl.Float64).mul(1 / (60 * 60 * 1e6)))
                .select("Duration")
                .mean()
                .item(),
                0,
            )
        )
    ),
)

col2.metric(
    label="Sleeps per day",
    value=(
        int(
            round(
                df.filter(pl.col("Type") == "Sleep")
                .group_by("StartDate")
                .len()
                .select("len")
                .mean()
                .item(),
                0,
            )
        )
    ),
)

col3.metric(
    label="% of night sleep hours",
    value=(
        int(
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
                0,
            )
        )
    ),
)

col4.metric(
    label="Feeds per day",
    value=(
        int(
            round(
                df.filter(pl.col("Type") == "Feed")
                .group_by("StartDate")
                .len()
                .select("len")
                .mean()
                .item(),
                0,
            )
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
    ),
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
    ),
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
    ),
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
    ),
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
    ),
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
    ),
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
    ),
    use_container_width=True,
)

st.markdown("## Best days")
st.markdown("### Top 10 longest average next night sleep duration")

best_days_df = create_best_days_df(df)
best_days_stat_df = create_best_days_stat_df(df)

# Plot 8
# Gantt chart
st.altair_chart(
    alt.Chart(
        data=gantt_df.filter(pl.col("Type") == "Sleep")
        .join(best_days_df, on="Date", how="inner")
        .with_columns(pl.col("Date").cast(pl.String)),
    )
    .mark_bar()
    .encode(
        x=alt.X(
            "StartTime:T",
            title="",
            axis=alt.Axis(format="%H:%M"),
        ),
        x2="EndTime:T",
        y=alt.Y("Date", title=""),
        color=alt.Color("Type", legend=alt.Legend(orient="top", title=None)),
    )
    .configure_axisY(
        titleAngle=0,
        titleY=-10,
        titleX=-22,
    ),
    use_container_width=True,
)

st.write(best_days_stat_df)

st.markdown("## Worst days")
st.markdown("### Top 10 shortest average next night sleep duration")


worst_days_df = create_worst_days_df(df)
worst_days_stat_df = create_worst_days_stat_df(df)


# Plot 9
# Gantt chart
st.altair_chart(
    alt.Chart(
        data=gantt_df.filter(pl.col("Type") == "Sleep")
        .join(worst_days_df, on="Date", how="inner")
        .with_columns(pl.col("Date").cast(pl.String)),
    )
    .mark_bar()
    .encode(
        x=alt.X(
            "StartTime:T",
            title="",
            axis=alt.Axis(format="%H:%M"),
        ),
        x2="EndTime:T",
        y=alt.Y("Date:O", title="", sort="y"),
        color=alt.Color("Type", legend=alt.Legend(orient="top", title=None)),
    )
    .configure_axisY(
        titleAngle=0,
        titleY=-10,
        titleX=-22,
    ),
    use_container_width=True,
)

st.write(worst_days_stat_df)
