package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestWakeUpVLLM(t *testing.T) {
	// Test successful wake up
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/wake_up" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	if err := wakeUpVLLM(ts.URL); err != nil {
		t.Fatalf("wakeUpVLLM failed: %v", err)
	}

	// Test failure when server returns error
	ts2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/wake_up" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer ts2.Close()

	if err := wakeUpVLLM(ts2.URL); err == nil {
		t.Errorf("wakeUpVLLM expected error for non-200 response")
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

func TestReloadWeightsVLLM(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/collective_rpc" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		body := make([]byte, r.ContentLength)
		if _, err := r.Body.Read(body); err == nil {
			if string(body) != `{"method":"reload_weights"}` {
				t.Errorf("Unexpected body: %s", body)
			}
		}
		w.Write([]byte(`{"results":[]}`))
	}))
	defer ts.Close()

	if err := reloadWeightsVLLM(ts.URL); err != nil {
		t.Fatalf("reloadWeightsVLLM failed: %v", err)
	}

	ts2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer ts2.Close()

	if err := reloadWeightsVLLM(ts2.URL); err == nil {
		t.Errorf("reloadWeightsVLLM expected error for non-200 response")
	}
}

func TestSleepVLLM(t *testing.T) {
	for _, level := range []int{1, 2} {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost || r.URL.Path != "/sleep" {
				t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
			}
			// vLLM reads the level from the query string only.
			if got := r.URL.Query().Get("level"); got != strconv.Itoa(level) {
				t.Errorf("level query param: got %q, want %d", got, level)
			}
			w.WriteHeader(http.StatusOK)
		}))

		err := sleepVLLM(ts.URL, level)
		ts.Close()
		if err != nil {
			t.Fatalf("sleepVLLM(level=%d) failed: %v", level, err)
		}
	}

	ts2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer ts2.Close()

	if err := sleepVLLM(ts2.URL, 2); err == nil {
		t.Errorf("sleepVLLM expected error for non-200 response")
	}
}

func TestRestoreAfterWake(t *testing.T) {
	// Level 1 needs no extra calls after a plain wake_up.
	calls := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	if err := restoreAfterWake(ts.URL, 1); err != nil {
		t.Fatalf("restoreAfterWake(level=1) failed: %v", err)
	}
	if calls != 0 {
		t.Errorf("level 1 should not make extra calls, got %d", calls)
	}

	// Level 2 must run the documented sequence in order.
	var got []string
	ts2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = append(got, r.Method+" "+r.URL.Path+"?"+r.URL.RawQuery)
		w.WriteHeader(http.StatusOK)
	}))
	defer ts2.Close()

	if err := restoreAfterWake(ts2.URL, 2); err != nil {
		t.Fatalf("restoreAfterWake(level=2) failed: %v", err)
	}
	want := []string{
		"POST /wake_up?tags=weights",
		"POST /collective_rpc?",
		"POST /wake_up?tags=kv_cache",
	}
	if len(got) != len(want) {
		t.Fatalf("call count: got %d %v, want %d", len(got), got, len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("call[%d]: got %q, want %q", i, got[i], want[i])
		}
	}
}

func TestGetSleepState(t *testing.T) {
	// A server whose /metrics reports the given sleep_state as active and
	// every other state as 0, matching vLLM's gauge semantics.
	metricsServer := func(t *testing.T, active string) *httptest.Server {
		t.Helper()
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/metrics" {
				t.Errorf("unexpected path %q", r.URL.Path)
			}
			w.Write([]byte("# HELP vllm:engine_sleep_state Engine sleep state\n"))
			for _, s := range []string{"awake", "weights_offloaded", "discard_all"} {
				v := 0
				if s == active {
					v = 1
				}
				fmt.Fprintf(w, "vllm:engine_sleep_state{sleep_state=%q} %d.0\n", s, v)
			}
		}))
	}

	tests := []struct {
		name   string
		active string
		want   int
	}{
		{"awake", "awake", -1},
		{"level 1", "weights_offloaded", 1},
		{"level 2", "discard_all", 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ts := metricsServer(t, tc.active)
			defer ts.Close()
			if got := getSleepState(ts.URL); got != tc.want {
				t.Errorf("getSleepState() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestGetSleepStateFallback(t *testing.T) {
	// /metrics unavailable, /is_sleeping reports awake: -1.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/is_sleeping":
			w.Write([]byte(`{"is_sleeping":false}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()
	if got := getSleepState(ts.URL); got != -1 {
		t.Errorf("awake fallback: getSleepState() = %d, want -1", got)
	}

	// /metrics unavailable, /is_sleeping reports asleep at an unknown level: 2.
	ts2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/is_sleeping":
			w.Write([]byte(`{"is_sleeping":true}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts2.Close()
	if got := getSleepState(ts2.URL); got != 2 {
		t.Errorf("asleep fallback: getSleepState() = %d, want 2", got)
	}
}

func TestGetSleepStateIndeterminate(t *testing.T) {
	// Server up but neither endpoint is usable: must report 2 (conservative).
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer ts.Close()
	if got := getSleepState(ts.URL); got != 2 {
		t.Errorf("indeterminate: getSleepState() = %d, want 2", got)
	}

	// Server unreachable entirely: also 2.
	if got := getSleepState("http://127.0.0.1:1"); got != 2 {
		t.Errorf("unreachable: getSleepState() = %d, want 2", got)
	}

	// /metrics returns 200 but with no engine_sleep_state lines: 2.
	ts3 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/metrics":
			w.Write([]byte("# HELP something_else a metric\nsomething_else 1\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts3.Close()
	if got := getSleepState(ts3.URL); got != 2 {
		t.Errorf("no metric line: getSleepState() = %d, want 2", got)
	}
}

func TestWakeEngine(t *testing.T) {
	// A level-2 sleep must use the documented restore sequence and must not
	// send a plain /wake_up first: after a level-2 sleep there is no weight
	// backup to restore from.
	var got []string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = append(got, r.Method+" "+r.URL.Path+"?"+r.URL.RawQuery)
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()
	if err := wakeEngine(ts.URL, 2); err != nil {
		t.Fatalf("wakeEngine(level=2) failed: %v", err)
	}
	want := []string{
		"POST /wake_up?tags=weights",
		"POST /collective_rpc?",
		"POST /wake_up?tags=kv_cache",
	}
	if len(got) != len(want) {
		t.Fatalf("call count: got %d %v, want %d %v", len(got), got, len(want), want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("call[%d]: got %q, want %q", i, got[i], want[i])
		}
	}

	// Levels other than 2 wake in a single call.
	ts2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/wake_up" || r.URL.RawQuery != "" {
			t.Errorf("unexpected request: %s %s?%s", r.Method, r.URL.Path, r.URL.RawQuery)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts2.Close()
	for _, level := range []int{-1, 0, 1} {
		if err := wakeEngine(ts2.URL, level); err != nil {
			t.Fatalf("wakeEngine(level=%d) failed: %v", level, err)
		}
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
