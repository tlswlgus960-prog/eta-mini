import os

import pytest

# 예상값을 고정하기 위해 기준선(rule) 구현체로 앱을 불러온다.
os.environ["MODEL_IMPL"] = "rule"

from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
