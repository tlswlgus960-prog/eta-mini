"""Python API 와 다른 언어 서비스(Go, Rust)의 예측이 같은지 비교한다.

기준은 학습한 원본 모델(MODEL_IMPL=sklearn)로 띄운 Python API 다.
같은 입력을 기준과 각 대상의 POST /eta 로 보내고, minutes 의 최대 오차가 TOLERANCE 분 미만이면 PASS.
대상 중 하나라도 FAIL 이면 exit 1 로 끝나 CI 가 멈춘다.

pytest 테스트가 아니다. 서버가 모두 떠 있어야 한다. 표준 라이브러리만 쓴다.
    uv run python -m training.check_parity

환경 변수
    PY_URL   기준 Python API (기본 http://127.0.0.1:8080)
    TARGETS  이름=URL 을 쉼표로 구분 (기본 go=http://127.0.0.1:8081,rust=http://127.0.0.1:8082)
"""
import json
import os
import random
import sys
import urllib.error
import urllib.request

TOLERANCE = 1e-4  # 분
TIMEOUT = 5       # 초
EXPECTED_BASELINE_MODEL = "sklearn"

PY_URL = os.getenv("PY_URL", "http://127.0.0.1:8080")
TARGETS = os.getenv("TARGETS", "go=http://127.0.0.1:8081,rust=http://127.0.0.1:8082")


def parse_targets(spec: str) -> dict[str, str]:
    """'go=http://...,rust=http://...' 를 {이름: URL} 로 바꾼다."""
    targets = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, url = item.partition("=")
        if not sep or not name.strip() or not url.strip():
            raise ValueError(f"TARGETS 형식 오류: {item!r} (이름=URL 이어야 한다)")
        targets[name.strip()] = url.strip().rstrip("/")
    if not targets:
        raise ValueError("TARGETS 가 비어 있다")
    return targets


def build_inputs() -> list[tuple[float, int]]:
    """고정 입력 3건 + seed 0 무작위 20건 (거리 0.5~8km, 주문 0~10건)."""
    fixed = [(3.0, 2), (0.0, 0), (8.0, 10)]
    rng = random.Random(0)
    rand = [(rng.uniform(0.5, 8.0), rng.randint(0, 10)) for _ in range(20)]
    return fixed + rand


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as res:
        return json.load(res)


def predict(base_url: str, distance_km: float, pending_orders: int) -> float:
    """POST /eta 를 호출해 minutes 를 돌려준다."""
    body = json.dumps({"distance_km": distance_km, "pending_orders": pending_orders}).encode()
    req = urllib.request.Request(f"{base_url}/eta", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
        return float(json.load(res)["minutes"])


def main() -> int:
    try:
        targets = parse_targets(TARGETS)
    except ValueError as e:
        print(f"설정 오류: {e}")
        return 1
    base_url = PY_URL.rstrip("/")
    inputs = build_inputs()

    # 기준 서버가 원본 모델로 떠 있는지 먼저 확인한다. rule 로 떠 있으면 비교가 무의미하다.
    try:
        health = get_json(f"{base_url}/health")
        expected = [predict(base_url, d, o) for d, o in inputs]
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        print(f"기준 Python API({base_url}) 호출 실패: {e}")
        return 1
    if health.get("model") != EXPECTED_BASELINE_MODEL:
        print(f"기준 Python API 의 구현체가 {health.get('model')!r} 다. "
              f"MODEL_IMPL={EXPECTED_BASELINE_MODEL} 로 띄워야 한다.")
        return 1

    print(f"기준: python({base_url}, model={health['model']})  입력 {len(inputs)}건  허용 오차 {TOLERANCE:g}분")
    all_ok = True
    for name, url in targets.items():
        try:
            got = [predict(url, d, o) for d, o in inputs]
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            print(f"  {name:<6} FAIL  호출 실패 ({url}): {e}")
            all_ok = False
            continue
        diffs = [abs(g - w) for g, w in zip(got, expected)]
        worst = max(range(len(diffs)), key=diffs.__getitem__)
        ok = diffs[worst] < TOLERANCE
        all_ok &= ok
        d, o = inputs[worst]
        print(f"  {name:<6} {'PASS' if ok else 'FAIL'}  최대 오차 {diffs[worst]:.2e}분 "
              f"(입력 {d:.3f}km/{o}건: python {expected[worst]:.6f} vs {name} {got[worst]:.6f})")

    print("결과:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
