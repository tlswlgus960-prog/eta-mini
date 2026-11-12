import pytest


# ---------- / ----------

def test_index_returns_html_intro(client):
    """/ 는 200 과 HTML 소개 페이지를 돌려주고, 엔드포인트와 현재 구현체를 안내한다."""
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    body = res.text
    assert body.lstrip().lower().startswith("<!doctype html>")
    assert "delivery-eta" in body
    for path in ("/health", "/eta", "/docs"):
        assert path in body
    assert "<code>rule</code>" in body


def test_index_rejects_post(client):
    """/ 는 GET 전용이라 POST 는 405 를 돌려준다."""
    assert client.post("/").status_code == 405


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


# ---------- /eta 경계값 (추가) ----------

def test_eta_just_below_zero_distance_returns_422(client):
    """0 바로 아래 거리(-1e-9)도 음수이므로 422 를 돌려준다."""
    res = client.post("/eta", json={"distance_km": -1e-9, "pending_orders": 0})
    assert res.status_code == 422


def test_eta_negative_zero_distance_is_zero(client):
    """-0.0 거리는 0 으로 취급되어 기본 10분을 돌려준다."""
    res = client.post("/eta", json={"distance_km": -0.0, "pending_orders": 0})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(10.0)


def test_eta_huge_integer_orders(client):
    """64비트를 넘는 대기 주문 수(2**70)도 200 으로 처리하고 공식대로 계산한다."""
    res = client.post("/eta", json={"distance_km": 1, "pending_orders": 2**70})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(10 + 3 * 1 + 2 * 2**70)


def test_eta_minutes_is_float_for_integer_inputs(client):
    """정수 입력이어도 minutes 는 실수(float)로 돌려준다."""
    res = client.post("/eta", json={"distance_km": 0, "pending_orders": 0})
    assert isinstance(res.json()["minutes"], float)


# ---------- /eta 입력 변환 정책 (현재 계약을 고정) ----------

@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"distance_km": "2.5", "pending_orders": "3"}, 23.5),          # 숫자 문자열
        ({"distance_km": 1, "pending_orders": 2.0}, 17.0),              # 소수부가 0인 주문 수
        ({"distance_km": True, "pending_orders": False}, 13.0),         # 불리언 → 1, 0
        ({"distance_km": 1, "pending_orders": 1, "foo": "bar"}, 15.0),  # 정의되지 않은 필드는 무시
    ],
)
def test_eta_lenient_inputs_are_coerced(client, payload, expected):
    """숫자 문자열·소수부 0 실수·불리언은 숫자로 바꾸고, 모르는 필드는 무시한다."""
    res = client.post("/eta", json=payload)
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(expected)


# ---------- /eta 비정상 수치 (알려진 결함: xfail) ----------

@pytest.mark.xfail(strict=True, reason="NaN 입력 시 422 대신 500 이 난다 (앱 결함)")
def test_eta_nan_distance_returns_422(client):
    """JSON 리터럴 NaN 거리는 422 를 돌려준다."""
    res = client.post(
        "/eta",
        content='{"distance_km": NaN, "pending_orders": 0}',
        headers={"content-type": "application/json"},
    )
    assert res.status_code == 422


@pytest.mark.xfail(strict=True, reason="무한대 입력을 받아 minutes=null 을 돌려준다 (앱 결함)")
@pytest.mark.parametrize("raw_distance", ["Infinity", '"inf"'])
def test_eta_infinite_distance_returns_422(client, raw_distance):
    """무한대 거리(Infinity, "inf")는 422 를 돌려준다."""
    res = client.post(
        "/eta",
        content=f'{{"distance_km": {raw_distance}, "pending_orders": 0}}',
        headers={"content-type": "application/json"},
    )
    assert res.status_code == 422


@pytest.mark.xfail(strict=True, reason="계산이 무한대로 넘쳐 minutes=null 을 돌려준다 (앱 결함)")
def test_eta_overflowing_distance_never_returns_null_minutes(client):
    """유한하지만 아주 큰 거리(1e308)에서도 200 이면 minutes 는 null 이 아니다."""
    res = client.post("/eta", json={"distance_km": 1e308, "pending_orders": 0})
    assert res.status_code == 422 or res.json()["minutes"] is not None


# ---------- /eta 요청 본문 형식 → 422 ----------

@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        ("[1, 2]", "application/json"),                                 # 객체가 아닌 JSON
        ("not json", "application/json"),                               # 깨진 JSON
        ("", "application/json"),                                       # 빈 본문
        ('{"distance_km": 1, "pending_orders": 1}', "text/plain"),      # JSON 이 아닌 Content-Type
    ],
)
def test_eta_malformed_body_returns_422(client, content, content_type):
    """객체가 아닌 JSON·깨진 JSON·빈 본문·잘못된 Content-Type 은 422 를 돌려준다."""
    res = client.post("/eta", content=content, headers={"content-type": content_type})
    assert res.status_code == 422


def test_eta_no_body_returns_422(client):
    """본문 없이 POST 하면 422 를 돌려준다."""
    assert client.post("/eta").status_code == 422


# ---------- /eta 422 응답 형식 ----------

@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"distance_km": -1, "pending_orders": 0}, "distance_km"),
        ({"distance_km": 1, "pending_orders": -1}, "pending_orders"),
        ({"distance_km": 1}, "pending_orders"),
    ],
)
def test_eta_422_detail_points_to_field(client, payload, field):
    """422 응답의 detail 은 문제가 된 필드를 loc 로 가리킨다."""
    res = client.post("/eta", json=payload)
    assert res.status_code == 422
    locs = [tuple(err["loc"]) for err in res.json()["detail"]]
    assert ("body", field) in locs


# ---------- 메서드 ----------

def test_health_rejects_post(client):
    """/health 는 GET 전용이라 POST 는 405 를 돌려준다."""
    assert client.post("/health").status_code == 405
