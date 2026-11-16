# delivery-eta Rust 서비스. 저장소 루트를 빌드 컨텍스트로 쓴다.
#   docker build -f api-rust/Dockerfile -t eta-api-rust .
#   docker run --rm -p 8082:8080 eta-api-rust
# 사전 조건: artifacts/model_sklearn.onnx (python -m training.export_onnx 로 만든다)
#
# 베이스 이미지는 Docker Hub 익명 pull 제한(429 Too Many Requests)을 피하려고
# 기본값으로 Google 의 Docker Hub 미러에서 받는다. Docker Hub 를 직접 쓰려면:
#   docker build --build-arg BASE_REGISTRY=docker.io/library -f api-rust/Dockerfile -t eta-api-rust .
ARG BASE_REGISTRY=mirror.gcr.io/library

# ── 1) ONNX Runtime: Python 쪽(uv.lock)과 같은 버전을 TARGETARCH 에 맞춰 받는다 ──
FROM ${BASE_REGISTRY}/rust:1.99-bookworm AS ort
ARG TARGETARCH
COPY uv.lock /tmp/uv.lock
RUN set -eu; \
    ver="$(awk -F'"' '{ sub(/\r$/, "") } $0 == "name = \"onnxruntime\"" { getline; sub(/\r$/, ""); print $2; exit }' /tmp/uv.lock)"; \
    echo "$ver" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' || { echo "uv.lock 에서 onnxruntime 버전을 찾지 못했다: '$ver'" >&2; exit 1; }; \
    case "$TARGETARCH" in \
      amd64) arch=x64 ;; \
      arm64) arch=aarch64 ;; \
      *) echo "지원하지 않는 TARGETARCH: '$TARGETARCH'" >&2; exit 1 ;; \
    esac; \
    url="https://github.com/microsoft/onnxruntime/releases/download/v${ver}/onnxruntime-linux-${arch}-${ver}.tgz"; \
    echo "ONNX Runtime ${ver} (${arch}): ${url}"; \
    curl -fsSL "$url" | tar -xz -C /tmp; \
    mkdir -p /opt/ort; \
    cp "/tmp/onnxruntime-linux-${arch}-${ver}/lib/libonnxruntime.so.${ver}" /opt/ort/libonnxruntime.so

# ── 2) Rust 빌드. ort 는 load-dynamic 이라 빌드 중에 런타임을 내려받지 않는다 ──
FROM ${BASE_REGISTRY}/rust:1.99-bookworm AS build
WORKDIR /src
# 의존성만 먼저 빌드해 캐시한다 (소스만 바뀌면 이 단계를 다시 하지 않는다)
COPY api-rust/Cargo.toml api-rust/Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release --locked && rm -rf src
COPY api-rust/src ./src
RUN touch src/main.rs && cargo build --release --locked

# ── 3) 실행 이미지 ──
FROM ${BASE_REGISTRY}/debian:bookworm-slim
RUN useradd --system --uid 10001 app
WORKDIR /app
COPY --from=ort /opt/ort/libonnxruntime.so /usr/local/lib/libonnxruntime.so
COPY --from=build /src/target/release/eta-api /usr/local/bin/eta-api
COPY artifacts/model_sklearn.onnx /app/model_sklearn.onnx
ENV ORT_LIB=/usr/local/lib/libonnxruntime.so \
    MODEL_PATH=/app/model_sklearn.onnx \
    PORT=8080
USER app
EXPOSE 8080
CMD ["eta-api"]
