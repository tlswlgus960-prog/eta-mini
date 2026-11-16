# delivery-eta Go 서비스. 저장소 루트를 빌드 컨텍스트로 쓴다.
#   docker build -f api-go/Dockerfile -t eta-api-go .
#   docker run --rm -p 8081:8080 eta-api-go
# 사전 조건: artifacts/model_sklearn.onnx (python -m training.export_onnx 로 만든다)

# ── 1) ONNX Runtime: Python 쪽(uv.lock)과 같은 버전을 TARGETARCH 에 맞춰 받는다 ──
FROM golang:1.27-bookworm AS ort
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

# ── 2) Go 빌드 (onnxruntime_go 가 cgo 를 쓴다) ──
FROM golang:1.27-bookworm AS build
WORKDIR /src
COPY api-go/go.mod api-go/go.sum ./
RUN go mod download
COPY api-go/*.go ./
RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/eta-api .

# ── 3) 실행 이미지 ──
FROM debian:bookworm-slim
RUN useradd --system --uid 10001 app
WORKDIR /app
COPY --from=ort /opt/ort/libonnxruntime.so /usr/local/lib/libonnxruntime.so
COPY --from=build /out/eta-api /usr/local/bin/eta-api
COPY artifacts/model_sklearn.onnx /app/model_sklearn.onnx
ENV ORT_LIB_PATH=/usr/local/lib/libonnxruntime.so \
    MODEL_PATH=/app/model_sklearn.onnx \
    PORT=8080
USER app
EXPOSE 8080
CMD ["eta-api"]
