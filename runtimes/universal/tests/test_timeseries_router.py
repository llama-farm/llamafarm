"""Tests for timeseries router backend availability checks."""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.timeseries.router import router, set_state


@pytest.fixture
def test_app():
    """Create a test FastAPI app with the timeseries router."""
    import asyncio

    app = FastAPI()
    app.include_router(router)
    set_state(models={}, model_load_lock=asyncio.Lock())
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestBackendAvailability:
    """Tests for backend availability checks in router endpoints."""

    def test_backends_list_includes_availability(self, client):
        """GET /v1/timeseries/backends includes available/unavailable_reason fields."""
        response = client.get("/v1/timeseries/backends")
        assert response.status_code == 200
        data = response.json()
        for backend in data["backends"]:
            assert "available" in backend
            assert "unavailable_reason" in backend

    def test_fit_unavailable_backend_returns_422(self, client):
        """POST /v1/timeseries/fit returns 422 when backend deps are missing."""
        with patch(
            "routers.timeseries.router.check_backend_available",
            return_value=(False, "Required dependency 'statsforecast' is not installed"),
        ):
            response = client.post(
                "/v1/timeseries/fit",
                json={
                    "backend": "arima",
                    "data": [
                        {"timestamp": "2024-01-01", "value": 100},
                        {"timestamp": "2024-01-02", "value": 120},
                    ],
                },
            )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "not available" in detail
        # Should NOT leak internal dependency details like pip install commands
        assert "pip install" not in detail

    def test_fit_available_backend_does_not_return_422(self, client):
        """POST /v1/timeseries/fit passes availability check for available backends."""
        with patch(
            "routers.timeseries.router.check_backend_available",
            return_value=(True, None),
        ), patch(
            "routers.timeseries.router._get_timeseries_model",
            side_effect=Exception("stop here"),
        ):
            response = client.post(
                "/v1/timeseries/fit",
                json={
                    "backend": "arima",
                    "data": [
                        {"timestamp": "2024-01-01", "value": 100},
                        {"timestamp": "2024-01-02", "value": 120},
                    ],
                },
            )
        # Should NOT be 422 (availability check passed)
        assert response.status_code != 422

    def test_backends_shows_unavailable_reason(self, client):
        """GET /v1/timeseries/backends shows reason when backend is unavailable."""
        with patch(
            "routers.timeseries.router.get_backends_info",
        ) as mock_info:
            from models.timeseries_model import BackendInfo

            mock_info.return_value = [
                BackendInfo(
                    name="arima",
                    description="ARIMA",
                    requires_training=True,
                    supports_confidence_intervals=True,
                    speed="medium",
                    available=False,
                    unavailable_reason="Required dependency 'statsforecast' is not installed",
                ),
            ]
            response = client.get("/v1/timeseries/backends")

        assert response.status_code == 200
        backends = response.json()["backends"]
        arima = backends[0]
        assert arima["available"] is False
        assert arima["unavailable_reason"] is not None
