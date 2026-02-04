"""Rolling feature engineering utilities using Polars.

Provides configurable feature engineering for time-series and streaming data:
- Rolling statistics (mean, std, min, max, sum, quantiles)
- Lag features
- Rate of change features
- Time-based aggregations

Usage:
    from utils.rolling_features import RollingFeatureConfig, compute_features

    config = RollingFeatureConfig(
        rolling_windows=[5, 10, 20],
        include_lags=True,
        lag_periods=[1, 2, 3],
        include_rate_of_change=True,
    )

    df_with_features = compute_features(df, config)
"""

import logging
from dataclasses import dataclass, field

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class RollingFeatureConfig:
    """Configuration for rolling feature computation.

    Attributes:
        rolling_windows: Window sizes for rolling statistics
        include_stats: Which statistics to compute (mean, std, min, max, sum, median)
        include_lags: Whether to include lag features
        lag_periods: Lag periods to compute
        include_rate_of_change: Whether to compute rate of change features
        include_ewm: Whether to compute exponentially weighted features
        ewm_spans: Spans for EWM (larger = smoother)
        columns: Specific columns to process (None = all numeric)
    """

    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    include_stats: list[str] = field(
        default_factory=lambda: ["mean", "std", "min", "max"]
    )
    include_lags: bool = True
    lag_periods: list[int] = field(default_factory=lambda: [1, 2, 3])
    include_rate_of_change: bool = False
    include_ewm: bool = False
    ewm_spans: list[int] = field(default_factory=lambda: [5, 10, 20])
    columns: list[str] | None = None


def compute_features(
    df: pl.DataFrame,
    config: RollingFeatureConfig | None = None,
) -> pl.DataFrame:
    """Compute rolling features for a DataFrame.

    Args:
        df: Input DataFrame with numeric columns
        config: Feature configuration (uses defaults if None)

    Returns:
        DataFrame with original columns plus computed features
    """
    if config is None:
        config = RollingFeatureConfig()

    if len(df) == 0:
        return df

    # Determine columns to process
    if config.columns:
        numeric_cols = [
            col for col in config.columns
            if col in df.columns and df[col].dtype.is_numeric()
        ]
    else:
        numeric_cols = [
            col for col, dtype in df.schema.items()
            if dtype.is_numeric()
        ]

    if not numeric_cols:
        logger.debug("No numeric columns found for feature engineering")
        return df

    # Build feature expressions
    feature_exprs = []

    for col in numeric_cols:
        # Rolling statistics
        feature_exprs.extend(
            _build_rolling_stats_exprs(col, config.rolling_windows, config.include_stats, len(df))
        )

        # Lag features
        if config.include_lags:
            feature_exprs.extend(_build_lag_exprs(col, config.lag_periods))

        # Rate of change
        if config.include_rate_of_change:
            feature_exprs.extend(_build_rate_of_change_exprs(col, config.lag_periods))

        # Exponentially weighted features
        if config.include_ewm:
            feature_exprs.extend(_build_ewm_exprs(col, config.ewm_spans))

    if feature_exprs:
        df = df.with_columns(feature_exprs)

    return df


def _build_rolling_stats_exprs(
    col: str,
    windows: list[int],
    stats: list[str],
    data_len: int,
) -> list[pl.Expr]:
    """Build expressions for rolling statistics."""
    exprs = []

    for window in windows:
        if window > data_len:
            continue  # Skip windows larger than data

        if "mean" in stats:
            exprs.append(
                pl.col(col).rolling_mean(window).alias(f"{col}_rolling_mean_{window}")
            )
        if "std" in stats:
            exprs.append(
                pl.col(col).rolling_std(window).alias(f"{col}_rolling_std_{window}")
            )
        if "min" in stats:
            exprs.append(
                pl.col(col).rolling_min(window).alias(f"{col}_rolling_min_{window}")
            )
        if "max" in stats:
            exprs.append(
                pl.col(col).rolling_max(window).alias(f"{col}_rolling_max_{window}")
            )
        if "sum" in stats:
            exprs.append(
                pl.col(col).rolling_sum(window).alias(f"{col}_rolling_sum_{window}")
            )
        if "median" in stats:
            exprs.append(
                pl.col(col).rolling_median(window).alias(f"{col}_rolling_median_{window}")
            )

    return exprs


def _build_lag_exprs(col: str, lag_periods: list[int]) -> list[pl.Expr]:
    """Build expressions for lag features."""
    return [
        pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")
        for lag in lag_periods
    ]


def _build_rate_of_change_exprs(col: str, lag_periods: list[int]) -> list[pl.Expr]:
    """Build expressions for rate of change features."""
    return [
        ((pl.col(col) - pl.col(col).shift(lag)) / pl.col(col).shift(lag).abs().clip(1e-10, None))
        .alias(f"{col}_roc_{lag}")
        for lag in lag_periods
    ]


def _build_ewm_exprs(col: str, spans: list[int]) -> list[pl.Expr]:
    """Build expressions for exponentially weighted moving features."""
    return [
        pl.col(col).ewm_mean(span=span).alias(f"{col}_ewm_mean_{span}")
        for span in spans
    ]


def compute_anomaly_features(
    df: pl.DataFrame,
    value_columns: list[str] | None = None,
    time_column: str | None = None,
    windows: list[int] | None = None,
) -> pl.DataFrame:
    """Compute features optimized for anomaly detection.

    Creates features that are useful for detecting anomalies:
    - Z-scores within rolling windows
    - Deviation from rolling mean (absolute)
    - Min-max scaled position within rolling window

    Args:
        df: Input DataFrame
        value_columns: Columns to compute features for (None = all numeric)
        time_column: Optional time column for time-based features
        windows: Rolling window sizes (default: [10, 20, 50])

    Returns:
        DataFrame with anomaly detection features
    """
    if windows is None:
        windows = [10, 20, 50]

    if len(df) == 0:
        return df

    # Determine columns
    if value_columns:
        cols = [c for c in value_columns if c in df.columns]
    else:
        cols = [
            col for col, dtype in df.schema.items()
            if dtype.is_numeric() and col != time_column
        ]

    if not cols:
        return df

    feature_exprs = []

    for col in cols:
        for window in windows:
            if window > len(df):
                continue

            # Z-score within window
            rolling_mean = pl.col(col).rolling_mean(window)
            rolling_std = pl.col(col).rolling_std(window)
            feature_exprs.append(
                ((pl.col(col) - rolling_mean) / rolling_std.clip(1e-10, None))
                .alias(f"{col}_zscore_{window}")
            )

            # Deviation from rolling mean (absolute)
            feature_exprs.append(
                (pl.col(col) - rolling_mean).abs().alias(f"{col}_deviation_{window}")
            )

            # Distance from rolling min/max (normalized)
            rolling_min = pl.col(col).rolling_min(window)
            rolling_max = pl.col(col).rolling_max(window)
            range_expr = (rolling_max - rolling_min).clip(1e-10, None)
            feature_exprs.append(
                ((pl.col(col) - rolling_min) / range_expr)
                .alias(f"{col}_minmax_scaled_{window}")
            )

    if feature_exprs:
        df = df.with_columns(feature_exprs)

    return df


def get_feature_names(
    columns: list[str],
    config: RollingFeatureConfig,
) -> list[str]:
    """Get list of feature names that would be generated.

    Useful for pre-allocating or validating expected features.

    Args:
        columns: Input column names
        config: Feature configuration

    Returns:
        List of feature column names
    """
    feature_names = []

    for col in columns:
        # Rolling stats
        for window in config.rolling_windows:
            for stat in config.include_stats:
                feature_names.append(f"{col}_rolling_{stat}_{window}")

        # Lags
        if config.include_lags:
            for lag in config.lag_periods:
                feature_names.append(f"{col}_lag_{lag}")

        # Rate of change
        if config.include_rate_of_change:
            for lag in config.lag_periods:
                feature_names.append(f"{col}_roc_{lag}")

        # EWM
        if config.include_ewm:
            for span in config.ewm_spans:
                feature_names.append(f"{col}_ewm_mean_{span}")

    return feature_names
