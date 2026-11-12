import importlib

import pytest
from fastapi.testclient import TestClient

import api.app


@pytest.fixture
def reload_app(monkeypatch):
    """MODEL_IMPL 을 바꿔 api.app 을 다시 불러오고, 끝나면 rule 로 되돌린다."""
    def _reload(impl: str):
        monkeypatch.setenv("MODEL_IMPL", impl)
        return importlib.reload(api.app)

    yield _reload
    monkeypatch.setenv("MODEL_IMPL", "rule")
    importlib.reload(api.app)


def test_scikit_impl_reports_its_name(reload_app):
    """MODEL_IMPL=scikit 이면 /health 와 /eta 의 model 이 scikit 이다."""
    module = reload_app("scikit")
    client = TestClient(module.app)

    assert client.get("/health").json() == {"status": "ok", "model": "scikit"}
    res = client.post("/eta", json={"distance_km": 2.0, "pending_orders": 3})
    assert res.status_code == 200
    assert res.json()["model"] == "scikit"
    assert isinstance(res.json()["minutes"], float)


def test_unknown_impl_fails_at_import(reload_app):
    """존재하지 않는 MODEL_IMPL 이면 앱을 불러오는 시점에 실패한다."""
    with pytest.raises(ModuleNotFoundError):
        reload_app("does_not_exist")
