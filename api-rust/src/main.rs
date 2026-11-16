//! delivery-eta 의 Rust 구현. api/app.py 와 같은 HTTP 계약을 따른다.
//!
//!   GET  /health → {"status": "ok", "model": MODEL_NAME}
//!   POST /eta    {"distance_km": 실수 ≥ 0, "pending_orders": 정수 ≥ 0}
//!                → {"minutes": 실수, "model": MODEL_NAME}, 잘못된 입력은 422 {"detail": [...]}
//!
//! 예측은 ONNX 모델(입력 "X" [1, 2] float32 → 출력 "variable" [1, 1])을 ort 로 실행한다.
//! 정규화는 모델 안에 있다. ort 는 load-dynamic 으로 ORT_LIB 의 공유 라이브러리를 읽는다.
//!
//! 환경 변수
//!   PORT        듣는 포트 (기본 8080)
//!   MODEL_PATH  ONNX 모델 경로 (기본 artifacts/model_sklearn.onnx)
//!   ORT_LIB     ONNX Runtime 공유 라이브러리 경로 (기본 libonnxruntime.so)
//!   MODEL_NAME  응답의 model 값 (기본 rust-onnx)

mod validate;

use std::sync::{Arc, Mutex};

use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use ort::{session::Session, value::Tensor};
use serde_json::json;

use validate::{FieldError, parse_eta_body};

struct AppState {
    // Session::run 이 &mut self 를 받으므로 요청 사이에 잠금으로 나눠 쓴다. 모델이 작아 경합은 짧다.
    session: Mutex<Session>,
    model_name: String,
}

/// ORT_LIB 의 ONNX Runtime 을 읽어(load-dynamic) 모델 세션을 만든다.
fn load_session(lib_path: &str, model_path: &str) -> Result<Session, Box<dyn std::error::Error>> {
    let env = ort::init_from(lib_path)?.build()?;
    Ok(Session::builder(&env)?.commit_from_file(model_path)?)
}

/// 배달 예상 시간(분)을 계산한다. Python onnx 구현체처럼 float32 로 넣는다.
fn predict(session: &Mutex<Session>, distance_km: f64, pending_orders: f64) -> ort::Result<f64> {
    let x = Tensor::from_array(([1usize, 2], vec![distance_km as f32, pending_orders as f32]))?;
    let mut session = session
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let outputs = session.run(ort::inputs!["X" => x])?;
    let (_, data) = outputs["variable"].try_extract_tensor::<f32>()?;
    Ok(f64::from(data[0]))
}

fn json_response(status: StatusCode, body: serde_json::Value) -> Response {
    (status, Json(body)).into_response()
}

fn unprocessable(errors: Vec<FieldError>) -> Response {
    json_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        json!({ "detail": errors }),
    )
}

/// FastAPI 처럼 허용하지 않는 메서드에 405 {"detail": "Method Not Allowed"} 를 돌려준다.
fn method_not_allowed(allow: &'static str) -> Response {
    let mut res = json_response(
        StatusCode::METHOD_NOT_ALLOWED,
        json!({ "detail": "Method Not Allowed" }),
    );
    res.headers_mut()
        .insert(header::ALLOW, HeaderValue::from_static(allow));
    res
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    json_response(
        StatusCode::OK,
        json!({ "status": "ok", "model": state.model_name }),
    )
}

async fn eta(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok());
    let input = match parse_eta_body(content_type, &body) {
        Ok(input) => input,
        Err(errors) => return unprocessable(errors),
    };
    match predict(&state.session, input.distance_km, input.pending_orders) {
        Ok(minutes) if minutes.is_finite() => json_response(
            StatusCode::OK,
            json!({ "minutes": minutes, "model": state.model_name }),
        ),
        // float32 범위를 넘는 입력(예: 1e308 km)은 결과가 무한대가 된다. null 을 돌려주지 않고 422 로 거절한다.
        Ok(_) => unprocessable(vec![FieldError {
            kind: "value_error",
            loc: vec![json!("body")],
            msg: "Input is out of the model's numeric range",
            input: None,
        }]),
        Err(e) => {
            eprintln!("예측 실패: {e}");
            json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({ "detail": "Internal Server Error" }),
            )
        }
    }
}

fn app(state: Arc<AppState>) -> Router {
    Router::new()
        .route(
            "/health",
            get(health).fallback(|| async { method_not_allowed("GET") }),
        )
        .route(
            "/eta",
            post(eta).fallback(|| async { method_not_allowed("POST") }),
        )
        .fallback(|| async {
            json_response(StatusCode::NOT_FOUND, json!({ "detail": "Not Found" }))
        })
        .with_state(state)
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

#[tokio::main]
async fn main() {
    let lib_path = env_or("ORT_LIB", "libonnxruntime.so");
    let model_path = env_or("MODEL_PATH", "artifacts/model_sklearn.onnx");
    let session = load_session(&lib_path, &model_path).unwrap_or_else(|e| {
        eprintln!("ONNX 모델 로드 실패 (ORT_LIB={lib_path}, MODEL_PATH={model_path}): {e}");
        std::process::exit(1);
    });
    let state = Arc::new(AppState {
        session: Mutex::new(session),
        model_name: env_or("MODEL_NAME", "rust-onnx"),
    });

    let addr = format!("0.0.0.0:{}", env_or("PORT", "8080"));
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("{addr} 에서 들을 수 없다: {e}");
            std::process::exit(1);
        });
    eprintln!("delivery-eta (rust) listening on {addr}");
    axum::serve(listener, app(state))
        .await
        .expect("서버 실행 실패");
}
