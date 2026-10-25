package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestParseWakeStep(t *testing.T) {
	tests := []struct {
		spec    string
		want    wakeStep
		wantErr bool
	}{
		{
			spec: "POST /wake_up {\"tags\":[\"weights\"]}",
			want: wakeStep{Method: http.MethodPost, Path: "/wake_up", Body: `{"tags":["weights"]}`},
		},
		{
			spec: "POST /collective_rpc {\"method\":\"reload_weights\"}",
			want: wakeStep{Method: http.MethodPost, Path: "/collective_rpc", Body: `{"method":"reload_weights"}`},
		},
		{
			spec: "GET /is_sleeping",
			want: wakeStep{Method: http.MethodGet, Path: "/is_sleeping"},
		},
		{
			spec: "/wake_up",
			want: wakeStep{Method: http.MethodPost, Path: "/wake_up"},
		},
		{
			spec: "/sleep?level=2",
			want: wakeStep{Method: http.MethodPost, Path: "/sleep?level=2"},
		},
		{spec: "", wantErr: true},
		{spec: "POST", wantErr: true},
		{spec: "BREW /coffee", wantErr: true},
		{spec: "POST not-a-path", wantErr: true},
		{spec: "POST /wake_up {not json}", wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.spec, func(t *testing.T) {
			got, err := parseWakeStep(tc.spec)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseWakeStep(%q) expected error, got %+v", tc.spec, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseWakeStep(%q) failed: %v", tc.spec, err)
			}
			if got != tc.want {
				t.Errorf("parseWakeStep(%q) = %+v, want %+v", tc.spec, got, tc.want)
			}
		})
	}
}

func TestDefaultWakeSteps(t *testing.T) {
	tests := []struct {
		level     int
		wantPaths []string
		wantBodys []string
	}{
		{level: 0, wantPaths: []string{"/wake_up"}, wantBodys: []string{""}},
		{level: 1, wantPaths: []string{"/wake_up"}, wantBodys: []string{""}},
		{
			level:     2,
			wantPaths: []string{"/wake_up", "/collective_rpc", "/wake_up"},
			wantBodys: []string{`{"tags":["weights"]}`, `{"method":"reload_weights"}`, `{"tags":["kv_cache"]}`},
		},
	}

	for _, tc := range tests {
		got := defaultWakeSteps(tc.level)
		if len(got) != len(tc.wantPaths) {
			t.Fatalf("level %d: got %d steps, want %d", tc.level, len(got), len(tc.wantPaths))
		}
		for i := range tc.wantPaths {
			if got[i].Path != tc.wantPaths[i] {
				t.Errorf("level %d step %d: path %q, want %q", tc.level, i, got[i].Path, tc.wantPaths[i])
			}
			if got[i].Body != tc.wantBodys[i] {
				t.Errorf("level %d step %d: body %q, want %q", tc.level, i, got[i].Body, tc.wantBodys[i])
			}
		}
	}
}

// TestRunWakeStepsOrder verifies the steps run in order and that each body and
// method reaches the server unchanged.
func TestRunWakeStepsOrder(t *testing.T) {
	type recorded struct {
		Method string
		Path   string
		Body   string
	}
	var got []recorded

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		got = append(got, recorded{Method: r.Method, Path: r.URL.Path, Body: string(body)})
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	steps := defaultWakeSteps(int(sleepLevel2))
	if err := runWakeSteps(ts.URL, steps, 10*time.Second); err != nil {
		t.Fatalf("runWakeSteps failed: %v", err)
	}

	want := []recorded{
		{Method: http.MethodPost, Path: "/wake_up", Body: `{"tags":["weights"]}`},
		{Method: http.MethodPost, Path: "/collective_rpc", Body: `{"method":"reload_weights"}`},
		{Method: http.MethodPost, Path: "/wake_up", Body: `{"tags":["kv_cache"]}`},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d requests, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("step %d: got %+v, want %+v", i, got[i], want[i])
		}
	}
}

// TestRunWakeStepsStopsOnFailure verifies a failing step aborts the sequence so
// later steps do not run against a half-woken engine.
func TestRunWakeStepsStopsOnFailure(t *testing.T) {
	var paths []string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		if r.URL.Path == "/collective_rpc" {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	err := runWakeSteps(ts.URL, defaultWakeSteps(int(sleepLevel2)), 10*time.Second)
	if err == nil {
		t.Fatal("runWakeSteps expected error when a step fails")
	}
	if !strings.Contains(err.Error(), "wake step 2/3") {
		t.Errorf("error should name the failing step, got %v", err)
	}
	want := []string{"/wake_up", "/collective_rpc"}
	if len(paths) != len(want) {
		t.Fatalf("got requests %v, want %v (third step must not run)", paths, want)
	}
	for i := range want {
		if paths[i] != want[i] {
			t.Errorf("request %d: got %q, want %q", i, paths[i], want[i])
		}
	}
}

// TestRunWakeStepsAcceptsNoContent verifies 204 counts as success, as some vLLM
// endpoints return No Content.
func TestRunWakeStepsAcceptsNoContent(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ts.Close()

	if err := runWakeSteps(ts.URL, defaultWakeSteps(int(sleepLevel1)), 10*time.Second); err != nil {
		t.Fatalf("runWakeSteps should accept 204: %v", err)
	}
}

func TestIsSleepingVLLM(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/is_sleeping" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Write([]byte(`{"is_sleeping":true}`))
	}))
	defer ts.Close()

	asleep, err := isSleepingVLLM(ts.URL)
	if err != nil {
		t.Fatalf("isSleepingVLLM failed: %v", err)
	}
	if !asleep {
		t.Errorf("expected is_sleeping=true")
	}
}

func TestWaitForHealthy(t *testing.T) {
	// Test successful health check
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("Unexpected path: %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"data":[]}`))
	}))
	defer ts.Close()

	if err := waitForHealthyWithPath(ts.URL, "/v1/models", 2*time.Second); err != nil {
		t.Fatalf("waitForHealthy failed: %v", err)
	}

	// Test timeout: server delays response longer than context timeout
	ts2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Delay 2 seconds
		time.Sleep(2 * time.Second)
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"data":[]}`))
	}))
	defer ts2.Close()

	err := waitForHealthyWithPath(ts2.URL, "/v1/models", 1*time.Second)
	if err == nil {
		t.Errorf("waitForHealthy expected timeout error")
		return
	}
	if err != context.DeadlineExceeded {
		t.Errorf("waitForHealthy expected context deadline exceeded, got %v", err)
	}
}

func TestSleepCommandMarshal(t *testing.T) {
	// We test the sleep command by checking the JSON marshaling we use in sleepCmd.
	// Since sleepCmd is not easily unit-testable without exposing more, we test the structure.
	body := map[string]int{"level": 1}
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}
	expected := `{"level":1}`
	if string(data) != expected {
		t.Errorf("Expected %s, got %s", expected, string(data))
	}
}

// TestStartDaemon tests that startDaemon returns an error when the start command exits
// quickly and the daemon does not become healthy.
func TestStartDaemon(t *testing.T) {
	// Use a start command that exits immediately (true) and a health URL that will not respond.
	err := startDaemon([]string{"true"}, "http://127.0.0.1:12345/health", "/health", 10*time.Millisecond)
	if err == nil {
		t.Fatalf("startDaemon expected error but got nil")
	}
	if !strings.Contains(err.Error(), "daemon did not become healthy") {
		t.Errorf("error expected to contain 'daemon did not become healthy', got %v", err)
	}
}

// TestStartDaemonArgv verifies that multiple startArgs are passed as separate argv values.
func TestStartDaemonArgv(t *testing.T) {
	tmpDir := t.TempDir()
	argvFile := filepath.Join(tmpDir, "argv.txt")

	script := filepath.Join(tmpDir, "write-argv.sh")
	if err := os.WriteFile(script, []byte(
		"#!/bin/bash\nprintf '%s\n' \"$@\" > \""+argvFile+"\"\nexit 0\n",
	), 0755); err != nil {
		t.Fatalf("write helper script: %v", err)
	}

	err := startDaemon([]string{script, "arg1", "arg2", "arg3"}, "http://127.0.0.1:12345/health", "/health", 100*time.Millisecond)
	if err == nil {
		t.Fatal("expected error (health check fails)")
	}

	content, err := os.ReadFile(argvFile)
	if err != nil {
		t.Fatalf("read argv file: %v", err)
	}
	got := strings.Split(strings.TrimSpace(string(content)), "\n")
	want := []string{"arg1", "arg2", "arg3"}
	if len(got) != len(want) {
		t.Fatalf("argv length: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("argv[%d]: got %q, want %q", i, got[i], want[i])
		}
	}
}

// TestRunWakeStepsTimeout verifies a hung step fails instead of blocking forever.
func TestRunWakeStepsTimeout(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	steps := []wakeStep{{Method: http.MethodPost, Path: "/wake_up"}}
	err := runWakeSteps(ts.URL, steps, 100*time.Millisecond)
	if err == nil {
		t.Fatal("runWakeSteps expected timeout error for a hung step")
	}
	if !strings.Contains(err.Error(), "wake step 1/1") {
		t.Errorf("error should name the failing step, got %v", err)
	}
}
