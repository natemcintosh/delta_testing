from pathlib import Path
from datetime import date

import polars as pl


def read_as_of(path: str | Path, asof: date, grouping_cols: list[str]) -> pl.LazyFrame:
    """
    Get the data from the DeltaTable as of a certain date.

    When getting the reports closest to `asof`, will group by `grouping_cols` if any.
    Examples of `grouping_cols` might be:
    - `["state", "event_date"]`
    - `["geo_value", "date"]`

    NOTE: The LazyFrame returned will be sorted by the grouping columns, in the order
    they are passed in.
    """

    return (
        pl.scan_delta(str(path))
        # Calculate time relative to `asof`
        .with_columns(time_to_report=asof - pl.col.report_date)
        # Get rid of anything later than `asof`
        .filter(pl.col.time_to_report >= 0)
        # Get only the closest to the `time_to_report` in each group
        .filter(
            (pl.col.time_to_report == pl.col.time_to_report.min()).over(grouping_cols)
        )
        .drop("time_to_report")
        .sort(grouping_cols)
    )
