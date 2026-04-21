"""Tests for formatting helpers used by runtime providers."""

import pytest

from server.services.runtime_service.providers.base import (
    format_bytes,
    format_duration,
)


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (None, None),
        (-1, None),
        (0, "0s"),
        (45, "45s"),
        (60, "1m"),
        (119, "1m"),
        (3600, "1h"),
        (3723, "1h 2m"),
        (86400, "1d"),
        (90061, "1d 1h"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (-1, None),
        (0, "0 B"),
        (1024, "1.0 KB"),
        (1024 * 1024 * 3, "3.0 MB"),
    ],
)
def test_format_bytes(value, expected):
    assert format_bytes(value) == expected
