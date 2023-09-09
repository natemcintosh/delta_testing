from datetime import date, timedelta

import polars as pl
import numpy as np


def create_random_data(
    n_rows: int,
    seed: int = 123,
    events_start: date = date(2022, 1, 1),
    events_end: date = date(2022, 12, 1),
    report_delay_mean: int | float = 2,
    report_delay_std: int | float = 0.75,
) -> pl.DataFrame:
    """
    Creates a dataframe with `n_rows`.

    Assumes report delays are lognormal.

    values are from a lognormal distribution

    Columns are:
    - event_date: date
    - report_date: date
    - sex: pl.Categorical
    - age: int
    - value: int
    """
    rng = np.random.default_rng(seed=seed)

    # Create the event dates
    n_event_days: int = (events_end - events_start).days
    event_offsets = rng.integers(low=0, high=n_event_days, size=n_rows).astype(
        np.timedelta64
    )
    event_dates = np.datetime64(events_start) + event_offsets

    # Create the report dates, using lognormal dist of delays from event
    delays = rng.lognormal(
        mean=report_delay_mean,
        sigma=report_delay_std,
        size=n_rows,
    ).astype(int)
    report_dates = event_dates + delays

    # Create final columns
    sex = rng.choice(["M", "F"], size=n_rows)
    age = rng.integers(low=0, high=110, size=n_rows)
    values = rng.lognormal(mean=10, sigma=2, size=n_rows).astype(int)

    return (
        pl.DataFrame(
            dict(
                event_date=event_dates,
                report_date=report_dates,
                sex=sex,
                age=age,
                values=values,
            )
        )
        # Make sex a category
        .with_columns(pl.col.sex.cast(pl.Categorical))
        .sort(["event_date", "report_date"])
    )
