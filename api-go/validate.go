package main

// POST /eta 요청 본문 검증.
// Python API(FastAPI + pydantic 의 느슨한 변환 규칙)와 같은 입력을 받고, 같은 입력을 422 로 거절한다.
//
//   distance_km    실수, 0 이상. 숫자·숫자 문자열("2.5", " 1_000 ")·불리언(true=1)을 받는다.
//   pending_orders 정수, 0 이상. 정수·소수부가 0 인 실수(2.0, 1e2)·정수 문자열("3", "3.0")·불리언을 받는다.
//   정의되지 않은 필드는 무시하고, 같은 키가 두 번 나오면 마지막 값을 쓴다.
//
// Python 과 의도적으로 다른 점: NaN·무한대는 422 로 거절한다.
// Python 은 이를 받아 500 이나 minutes=null 을 돌려주며, api/tests 에 xfail(알려진 결함)로 기록돼 있다.

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"mime"
	"regexp"
	"strconv"
	"strings"
)

// fieldError 는 FastAPI 422 응답의 detail 항목과 같은 모양이다.
type fieldError struct {
	Type  string `json:"type"`
	Loc   []any  `json:"loc"`
	Msg   string `json:"msg"`
	Input any    `json:"input,omitempty"`
}

type etaInput struct {
	DistanceKm    float64
	PendingOrders float64 // 정수임을 검증한 값. 2**70 같은 큰 정수도 받으므로 float64 로 둔다.
}

var (
	// 밑줄은 숫자 사이에만 (Python float("1_000") 규칙)
	floatStrRe = regexp.MustCompile(`^[+-]?(\d(_?\d)*(\.(\d(_?\d)*)?)?|\.\d(_?\d)*)([eE][+-]?\d(_?\d)*)?$`)
	// 정수 문자열은 지수 표기를 받지 않고, 소수부는 0 만 허용한다 ("3.0" 은 되고 "3."·"3.5"·"1e2" 는 안 된다)
	intStrRe = regexp.MustCompile(`^[+-]?\d(_?\d)*(\.0+)?$`)
)

// isJSONContentType 은 FastAPI 처럼 application/json 과 application/*+json 만 JSON 으로 읽는다.
func isJSONContentType(ct string) bool {
	mt, _, err := mime.ParseMediaType(ct)
	if err != nil {
		return false
	}
	main, sub, ok := strings.Cut(mt, "/")
	return ok && main == "application" && (sub == "json" || strings.HasSuffix(sub, "+json"))
}

// parseEtaBody 는 본문을 검증해 입력값을 돌려주거나, 422 로 돌려줄 오류 목록을 돌려준다.
func parseEtaBody(contentType string, body []byte) (etaInput, []fieldError) {
	if len(body) == 0 {
		return etaInput{}, []fieldError{{Type: "missing", Loc: []any{"body"}, Msg: "Field required"}}
	}
	if !isJSONContentType(contentType) {
		return etaInput{}, []fieldError{notObject(string(body))}
	}

	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber() // 2**70 같은 정수와 1.0 / 1 의 구분을 잃지 않도록 원문 그대로 받는다
	var raw any
	if err := dec.Decode(&raw); err != nil {
		return etaInput{}, []fieldError{jsonInvalid(err)}
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) { // {"a":1} 뒤에 다른 값이 붙은 경우
		return etaInput{}, []fieldError{jsonInvalid(errors.New("extra data after JSON value"))}
	}

	obj, ok := raw.(map[string]any)
	if !ok {
		return etaInput{}, []fieldError{notObject(raw)}
	}

	var in etaInput
	var errs []fieldError
	if v, present := obj["distance_km"]; !present {
		errs = append(errs, missing("distance_km"))
	} else if f, e := toFloat("distance_km", v); e != nil {
		errs = append(errs, *e)
	} else {
		in.DistanceKm = f
	}
	if v, present := obj["pending_orders"]; !present {
		errs = append(errs, missing("pending_orders"))
	} else if n, e := toInt("pending_orders", v); e != nil {
		errs = append(errs, *e)
	} else {
		in.PendingOrders = n
	}
	return in, errs
}

// toFloat 은 pydantic float(lax) 규칙으로 변환하고 0 이상·유한 조건을 확인한다.
func toFloat(field string, v any) (float64, *fieldError) {
	var f float64
	switch x := v.(type) {
	case bool:
		f = boolToFloat(x)
	case json.Number:
		var err error
		if f, err = strconv.ParseFloat(string(x), 64); err != nil && !isRangeErr(err) {
			return 0, typeErr(field, "float_parsing", "Input should be a valid number", v)
		}
	case string:
		s := strings.TrimSpace(x)
		if !floatStrRe.MatchString(s) {
			return 0, typeErr(field, "float_parsing", "Input should be a valid number, unable to parse string as a number", v)
		}
		var err error
		if f, err = strconv.ParseFloat(strings.ReplaceAll(s, "_", ""), 64); err != nil && !isRangeErr(err) {
			return 0, typeErr(field, "float_parsing", "Input should be a valid number, unable to parse string as a number", v)
		}
	default:
		return 0, typeErr(field, "float_type", "Input should be a valid number", v)
	}
	return checkRange(field, f, v)
}

// toInt 은 pydantic int(lax) 규칙으로 변환하고 0 이상·유한 조건을 확인한다.
func toInt(field string, v any) (float64, *fieldError) {
	var f float64
	switch x := v.(type) {
	case bool:
		f = boolToFloat(x)
	case json.Number:
		var err error
		if f, err = strconv.ParseFloat(string(x), 64); err != nil && !isRangeErr(err) {
			return 0, typeErr(field, "int_parsing", "Input should be a valid integer", v)
		}
		if !math.IsInf(f, 0) && f != math.Trunc(f) {
			return 0, typeErr(field, "int_from_float", "Input should be a valid integer, got a number with a fractional part", v)
		}
	case string:
		s := strings.TrimSpace(x)
		if !intStrRe.MatchString(s) {
			return 0, typeErr(field, "int_parsing", "Input should be a valid integer, unable to parse string as an integer", v)
		}
		s, _, _ = strings.Cut(strings.ReplaceAll(s, "_", ""), ".")
		var err error
		if f, err = strconv.ParseFloat(s, 64); err != nil && !isRangeErr(err) {
			return 0, typeErr(field, "int_parsing", "Input should be a valid integer, unable to parse string as an integer", v)
		}
	default:
		return 0, typeErr(field, "int_type", "Input should be a valid integer", v)
	}
	return checkRange(field, f, v)
}

func checkRange(field string, f float64, input any) (float64, *fieldError) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, typeErr(field, "finite_number", "Input should be a finite number", input)
	}
	if f < 0 {
		return 0, typeErr(field, "greater_than_equal", "Input should be greater than or equal to 0", input)
	}
	return f + 0, nil // -0 → 0
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// isRangeErr: 1e400 처럼 범위를 넘는 값은 ParseFloat 이 ±Inf 와 함께 돌려준다. 이후 finite 검사에서 422 가 된다.
func isRangeErr(err error) bool {
	var ne *strconv.NumError
	return errors.As(err, &ne) && errors.Is(ne.Err, strconv.ErrRange)
}

func typeErr(field, typ, msg string, input any) *fieldError {
	return &fieldError{Type: typ, Loc: []any{"body", field}, Msg: msg, Input: input}
}

func missing(field string) fieldError {
	return fieldError{Type: "missing", Loc: []any{"body", field}, Msg: "Field required"}
}

func notObject(input any) fieldError {
	return fieldError{Type: "model_attributes_type", Loc: []any{"body"},
		Msg: "Input should be a valid dictionary or object to extract fields from", Input: input}
}

func jsonInvalid(err error) fieldError {
	return fieldError{Type: "json_invalid", Loc: []any{"body"}, Msg: "JSON decode error: " + err.Error()}
}
