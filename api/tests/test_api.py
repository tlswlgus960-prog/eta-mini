import pytest


# ---------- /health ----------

def test_health_returns_ok_and_model(client):
    """/health 는 200 과 status=ok, 현재 구현체 이름을 돌려준다."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "model": "rule"}


# ---------- /eta 정상 입력 ----------

def test_eta_response_shape(client):
    """/eta 응답은 minutes(숫자)와 model(문자열) 두 필드만 가진다."""
    res = client.post("/eta", json={"distance_km": 1.0, "pending_orders": 1})
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"minutes", "model"}
    assert isinstance(body["minutes"], (int, float))
    assert body["model"] == "rule"


@pytest.mark.parametrize(
    ("distance_km", "pending_orders", "expected"),
    [
        (2.0, 3, 22.0),   # 10 + 3*2 + 2*3
        (1.5, 0, 14.5),   # 소수 거리
        (5, 4, 33.0),     # 정수로 보낸 거리
    ],
)
def test_eta_known_values(client, distance_km, pending_orders, expected):
    """알려진 입력에 대해 기준선 공식(10 + 3*km + 2*주문)의 값을 돌려준다."""
    res = client.post("/eta", json={"distance_km": distance_km, "pending_orders": pending_orders})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(expected)


# ---------- /eta 경계값 ----------

def test_eta_zero_inputs(client):
    """거리 0, 대기 주문 0 이면 기본 10분을 돌려준다."""
    res = client.post("/eta", json={"distance_km": 0, "pending_orders": 0})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(10.0)


def test_eta_large_inputs(client):
    """아주 큰 거리·주문 수도 200 으로 처리하고 공식대로 계산한다."""
    res = client.post("/eta", json={"distance_km": 10_000, "pending_orders": 100_000})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(10 + 3 * 10_000 + 2 * 100_000)


def test_eta_increases_with_distance_and_orders(client):
    """거리나 대기 주문이 늘면 예상 시간도 늘어난다."""
    def minutes(km, orders):
        return client.post("/eta", json={"distance_km": km, "pending_orders": orders}).json()["minutes"]

    assert minutes(3, 2) > minutes(2, 2)
    assert minutes(2, 3) > minutes(2, 2)


# ---------- /eta 잘못된 입력 → 422 ----------

@pytest.mark.parametrize(
    "payload",
    [
        {"distance_km": -1, "pending_orders": 0},         # 음수 거리
        {"distance_km": 1, "pending_orders": -1},         # 음수 주문
        {"distance_km": "far", "pending_orders": 0},      # 숫자가 아닌 거리
        {"distance_km": 1, "pending_orders": "many"},     # 숫자가 아닌 주문
        {"distance_km": 1, "pending_orders": 1.5},        # 정수가 아닌 주문 수
        {"distance_km": None, "pending_orders": 0},       # null
        {"pending_orders": 0},                            # 거리 누락
        {"distance_km": 1},                               # 주문 누락
        {},                                               # 빈 본문
    ],
)
def test_eta_invalid_input_returns_422(client, payload):
    """음수·숫자 아님·정수 아님·누락된 필드는 422 를 돌려준다."""
    res = client.post("/eta", json=payload)
    assert res.status_code == 422


def test_eta_rejects_get(client):
    """/eta 는 POST 전용이라 GET 은 405 를 돌려준다."""
    assert client.get("/eta").status_code == 405
