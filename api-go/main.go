// delivery-eta 의 Go 구현. api/app.py 와 같은 HTTP 계약을 따른다.
//
//	GET  /health → {"status": "ok", "model": MODEL_NAME}
//	POST /eta    {"distance_km": 실수 ≥ 0, "pending_orders": 정수 ≥ 0}
//	             → {"minutes": 실수, "model": MODEL_NAME}, 잘못된 입력은 422 {"detail": [...]}
//
// 예측은 ONNX 모델(입력 "X" [1, 2] float32 → 출력 "variable" [1, 1])을 ONNX Runtime 으로 실행한다.
// 정규화는 모델 안에 있다.
//
// 환경 변수
//
//	PORT          듣는 포트 (기본 8080)
//	MODEL_PATH    ONNX 모델 경로 (기본 artifacts/model_sklearn.onnx)
//	ORT_LIB_PATH  ONNX Runtime 공유 라이브러리 경로 (기본 libonnxruntime.so)
//	MODEL_NAME    응답의 model 값 (기본 go-onnx)
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

const maxBodyBytes = 1 << 20

type predictor struct {
	session *ort.DynamicAdvancedSession
}

func newPredictor(libPath, modelPath string) (*predictor, error) {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("ONNX Runtime 초기화 실패 (%s): %w", libPath, err)
	}
	// DynamicAdvancedSession 은 Run 마다 텐서를 받으므로 여러 요청이 동시에 써도 된다.
	s, err := ort.NewDynamicAdvancedSession(modelPath, []string{"X"}, []string{"variable"}, nil)
	if err != nil {
		return nil, fmt.Errorf("모델 로드 실패 (%s): %w", modelPath, err)
	}
	return &predictor{session: s}, nil
}

// predict 는 배달 예상 시간(분)을 계산한다. Python onnx 구현체처럼 float32 로 넣는다.
func (p *predictor) predict(distanceKm, pendingOrders float64) (float64, error) {
	x, err := ort.NewTensor(ort.NewShape(1, 2), []float32{float32(distanceKm), float32(pendingOrders)})
	if err != nil {
		return 0, err
	}
	defer x.Destroy()
	y, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		return 0, err
	}
	defer y.Destroy()
	if err := p.session.Run([]ort.Value{x}, []ort.Value{y}); err != nil {
		return 0, err
	}
	return float64(y.GetData()[0]), nil
}

// pyFloat 은 Python 처럼 정수값 실수도 13.0 으로 직렬화한다 (encoding/json 은 13 으로 쓴다).
type pyFloat float64

func (f pyFloat) MarshalJSON() ([]byte, error) {
	s := strconv.FormatFloat(float64(f), 'g', -1, 64)
	if !strings.ContainsAny(s, ".eEn") { // n: NaN/Inf 는 여기 오기 전에 걸러진다
		s += ".0"
	}
	return []byte(s), nil
}

type etaResponse struct {
	Minutes pyFloat `json:"minutes"`
	Model   string  `json:"model"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("응답 쓰기 실패: %v", err)
	}
}

// route 는 FastAPI 처럼 허용하지 않는 메서드에 405 {"detail": "Method Not Allowed"} 를 돌려준다.
func route(method string, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method {
			w.Header().Set("Allow", method)
			writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"detail": "Method Not Allowed"})
			return
		}
		h(w, r)
	}
}

func newMux(p *predictor, modelName string) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", route(http.MethodGet, func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "model": modelName})
	}))
	mux.HandleFunc("/eta", route(http.MethodPost, func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
		if err != nil {
			writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"detail": "Request body too large"})
			return
		}
		in, errs := parseEtaBody(r.Header.Get("Content-Type"), body)
		if len(errs) > 0 {
			writeJSON(w, http.StatusUnprocessableEntity, map[string]any{"detail": errs})
			return
		}
		minutes, err := p.predict(in.DistanceKm, in.PendingOrders)
		if err != nil {
			log.Printf("예측 실패: %v", err)
			writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": "Internal Server Error"})
			return
		}
		// float32 범위를 넘는 입력(예: 1e308 km)은 결과가 무한대가 된다. null 을 돌려주지 않고 422 로 거절한다.
		if math.IsNaN(minutes) || math.IsInf(minutes, 0) {
			writeJSON(w, http.StatusUnprocessableEntity, map[string]any{"detail": []fieldError{{
				Type: "value_error", Loc: []any{"body"}, Msg: "Input is out of the model's numeric range"}}})
			return
		}
		writeJSON(w, http.StatusOK, etaResponse{Minutes: pyFloat(minutes), Model: modelName})
	}))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusNotFound, map[string]string{"detail": "Not Found"})
	})
	return mux
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	p, err := newPredictor(getenv("ORT_LIB_PATH", "libonnxruntime.so"), getenv("MODEL_PATH", "artifacts/model_sklearn.onnx"))
	if err != nil {
		log.Fatal(err)
	}
	addr := ":" + getenv("PORT", "8080")
	log.Printf("delivery-eta (go) listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, newMux(p, getenv("MODEL_NAME", "go-onnx"))))
}
