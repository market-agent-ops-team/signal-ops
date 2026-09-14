import socket

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def offline_network(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        return

    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("requests.sessions.Session.request", blocked)
    monkeypatch.setattr("curl_cffi.requests.Session.request", blocked)


@pytest.fixture
def prices():
    day = np.arange(240)
    close = 100 + day * 0.04 + 4 * np.sin(day / 4)
    return pd.DataFrame({
        "ticker": "INFY.NS",
        "date": pd.bdate_range("2023-01-02", periods=len(day)),
        "open": close - 0.3,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 10000 + (day % 13) * 500,
    })
