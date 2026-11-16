//! POST /eta 요청 본문 검증.
//! Python API(FastAPI + pydantic 의 느슨한 변환 규칙)와 같은 입력을 받고, 같은 입력을 422 로 거절한다.
//!
//!   distance_km    실수, 0 이상. 숫자·숫자 문자열("2.5", " 1_000 ")·불리언(true=1)을 받는다.
//!   pending_orders 정수, 0 이상. 정수·소수부가 0 인 실수(2.0, 1e2)·정수 문자열("3", "3.0")·불리언을 받는다.
//!   정의되지 않은 필드는 무시하고, 같은 키가 두 번 나오면 마지막 값을 쓴다.
//!
//! axum 의 Json 추출기는 형식 오류에 400·415 를 돌려주므로 쓰지 않고, 본문을 직접 파싱해 모든 입력 오류를 422 로 맞춘다.
//!
//! Python 과 의도적으로 다른 점: NaN·무한대는 422 로 거절한다.
//! Python 은 이를 받아 500 이나 minutes=null 을 돌려주며, api/tests 에 xfail(알려진 결함)로 기록돼 있다.

use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;
use serde_json::{Value, json};

/// FastAPI 422 응답의 detail 항목과 같은 모양.
#[derive(Debug, Serialize)]
pub struct FieldError {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub loc: Vec<Value>,
    pub msg: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EtaInput {
    pub distance_km: f64,
    /// 정수임을 검증한 값. 2**70 같은 큰 정수도 받으므로 f64 로 둔다.
    pub pending_orders: f64,
}

// 밑줄은 숫자 사이에만 (Python float("1_000") 규칙). \d 는 유니코드 숫자까지 받으므로 [0-9] 를 쓴다.
static FLOAT_STR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[+-]?([0-9](_?[0-9])*(\.([0-9](_?[0-9])*)?)?|\.[0-9](_?[0-9])*)([eE][+-]?[0-9](_?[0-9])*)?$").unwrap()
});
// 정수 문자열은 지수 표기를 받지 않고, 소수부는 0 만 허용한다 ("3.0" 은 되고 "3."·"3.5"·"1e2" 는 안 된다)
static INT_STR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[+-]?[0-9](_?[0-9])*(\.0+)?$").unwrap());

/// FastAPI 처럼 application/json 과 application/*+json 만 JSON 으로 읽는다.
pub fn is_json_content_type(content_type: Option<&str>) -> bool {
    let Some(ct) = content_type else { return false };
    let media_type = ct
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match media_type.split_once('/') {
        Some(("application", sub)) => sub == "json" || sub.ends_with("+json"),
        _ => false,
    }
}

/// 본문을 검증해 입력값을 돌려주거나, 422 로 돌려줄 오류 목록을 돌려준다.
pub fn parse_eta_body(
    content_type: Option<&str>,
    body: &[u8],
) -> Result<EtaInput, Vec<FieldError>> {
    if body.is_empty() {
        return Err(vec![FieldError {
            kind: "missing",
            loc: vec![json!("body")],
            msg: "Field required",
            input: None,
        }]);
    }
    if !is_json_content_type(content_type) {
        return Err(vec![not_object(Value::String(
            String::from_utf8_lossy(body).into_owned(),
        ))]);
    }
    // serde_json 은 뒤에 붙은 값, NaN 리터럴, 범위를 넘는 실수(1e400)를 모두 오류로 낸다.
    let raw: Value = serde_json::from_slice(body).map_err(|_| {
        vec![FieldError {
            kind: "json_invalid",
            loc: vec![json!("body")],
            msg: "JSON decode error",
            input: None,
        }]
    })?;
    let Value::Object(obj) = raw else {
        return Err(vec![not_object(raw)]);
    };

    let mut errors = Vec::new();
    let distance_km = match obj.get("distance_km") {
        None => Err(missing("distance_km")),
        Some(v) => to_float("distance_km", v),
    };
    let pending_orders = match obj.get("pending_orders") {
        None => Err(missing("pending_orders")),
        Some(v) => to_int("pending_orders", v),
    };
    let distance_km = distance_km.map_err(|e| errors.push(e)).ok();
    let pending_orders = pending_orders.map_err(|e| errors.push(e)).ok();
    match (distance_km, pending_orders) {
        (Some(distance_km), Some(pending_orders)) => Ok(EtaInput {
            distance_km,
            pending_orders,
        }),
        _ => Err(errors),
    }
}

/// pydantic float(lax) 규칙으로 변환하고 0 이상·유한 조건을 확인한다.
fn to_float(field: &'static str, v: &Value) -> Result<f64, FieldError> {
    let f = match v {
        Value::Bool(b) => bool_to_f64(*b),
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| type_err(field, "float_parsing", "Input should be a valid number", v))?,
        Value::String(s) => {
            const MSG: &str = "Input should be a valid number, unable to parse string as a number";
            let s = s.trim();
            if !FLOAT_STR.is_match(s) {
                return Err(type_err(field, "float_parsing", MSG, v));
            }
            s.replace('_', "")
                .parse::<f64>()
                .map_err(|_| type_err(field, "float_parsing", MSG, v))?
        }
        _ => {
            return Err(type_err(
                field,
                "float_type",
                "Input should be a valid number",
                v,
            ));
        }
    };
    check_range(field, f, v)
}

/// pydantic int(lax) 규칙으로 변환하고 0 이상·유한 조건을 확인한다.
fn to_int(field: &'static str, v: &Value) -> Result<f64, FieldError> {
    let f = match v {
        Value::Bool(b) => bool_to_f64(*b),
        Value::Number(n) => {
            // u64·i64 를 넘는 정수 리터럴은 serde_json 이 f64 로 읽는다
            let f = n.as_f64().ok_or_else(|| {
                type_err(field, "int_parsing", "Input should be a valid integer", v)
            })?;
            if f.is_finite() && f.fract() != 0.0 {
                return Err(type_err(
                    field,
                    "int_from_float",
                    "Input should be a valid integer, got a number with a fractional part",
                    v,
                ));
            }
            f
        }
        Value::String(s) => {
            const MSG: &str =
                "Input should be a valid integer, unable to parse string as an integer";
            let s = s.trim();
            if !INT_STR.is_match(s) {
                return Err(type_err(field, "int_parsing", MSG, v));
            }
            let digits = s.replace('_', "");
            let whole = digits.split('.').next().unwrap_or("");
            whole
                .parse::<f64>()
                .map_err(|_| type_err(field, "int_parsing", MSG, v))?
        }
        _ => {
            return Err(type_err(
                field,
                "int_type",
                "Input should be a valid integer",
                v,
            ));
        }
    };
    check_range(field, f, v)
}

fn check_range(field: &'static str, f: f64, input: &Value) -> Result<f64, FieldError> {
    if !f.is_finite() {
        return Err(type_err(
            field,
            "finite_number",
            "Input should be a finite number",
            input,
        ));
    }
    if f < 0.0 {
        return Err(type_err(
            field,
            "greater_than_equal",
            "Input should be greater than or equal to 0",
            input,
        ));
    }
    Ok(f + 0.0) // -0 → 0
}

fn bool_to_f64(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

fn type_err(
    field: &'static str,
    kind: &'static str,
    msg: &'static str,
    input: &Value,
) -> FieldError {
    FieldError {
        kind,
        loc: vec![json!("body"), json!(field)],
        msg,
        input: Some(input.clone()),
    }
}

fn missing(field: &'static str) -> FieldError {
    FieldError {
        kind: "missing",
        loc: vec![json!("body"), json!(field)],
        msg: "Field required",
        input: None,
    }
}

fn not_object(input: Value) -> FieldError {
    FieldError {
        kind: "model_attributes_type",
        loc: vec![json!("body")],
        msg: "Input should be a valid dictionary or object to extract fields from",
        input: Some(input),
    }
}
