package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// vllmWrapper serves as a cmd/cmdStop wrapper for vLLM with sleep mode.
func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <command> [args]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Commands:\n")
		fmt.Fprintf(os.Stderr, "  serve    Start as a forward proxy (for cmd)\n")
		fmt.Fprintf(os.Stderr, "  sleep    Put vLLM to sleep (for cmdStop)\n")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "serve":
		serveCmd(os.Args[2:])
	case "sleep":
		sleepCmd(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", os.Args[1])
		os.Exit(1)
	}
}

// serveCmd implements the serve subcommand.
func serveCmd(args []string) {
	var (
		vllmURL     string
		listenAddr  string
		sleepLevel  int
		healthPath  string
		waitTimeout time.Duration
		journalUnit string
	)
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	fs.StringVar(&vllmURL, "vllm-url", "", "Base URL of vLLM server (e.g., http://127.0.0.1:8000)")
	fs.StringVar(&listenAddr, "listen", "", "Address to listen on (e.g., :$PORT)")
	fs.IntVar(&sleepLevel, "sleep-level", 1, "Sleep level to use when sleeping (default 1)")
	fs.StringVar(&healthPath, "health-path", "/health", "Health check path (default /health)")
	fs.DurationVar(&waitTimeout, "wait-timeout", 120*time.Second, "Timeout waiting for daemon to become healthy")
	fs.StringVar(&journalUnit, "journal-unit", "", "User systemd unit whose logs should be forwarded to stdout")
	fs.Parse(args)
	startArgs := fs.Args()

	if vllmURL == "" {
		log.Fatalf("--vllm-url is required")
	}
	if listenAddr == "" {
		log.Fatalf("--listen is required")
	}
	if len(startArgs) == 0 {
		log.Fatalf("a daemon command after -- is required")
	}

	// Ensure vLLM URL does not have trailing slash.
	vllmURL = strings.TrimRight(vllmURL, "/")

	journalCtx, stopJournal := context.WithCancel(context.Background())
	defer stopJournal()
	if journalUnit != "" {
		if err := startJournalForwarder(journalCtx, journalUnit); err != nil {
			log.Printf("Warning: failed to forward logs from %s: %v", journalUnit, err)
		}
	}

	// Step 1: Ensure the daemon is running and awake.
	// First, check if we can reach the daemon (liveness).
	if err := checkHealthy(vllmURL, healthPath); err != nil {
		// Not reachable, try to wake up (in case it's asleep but we couldn't reach? Actually, if not reachable, waking won't work)
		log.Printf("vLLM daemon not reachable (%v), attempting to wake up", err)
		if err := wakeUpVLLM(vllmURL); err != nil {
			// Wake up failed (e.g., connection refused), assume daemon not running, try to start it.
			log.Printf("Wake up failed: %v, attempting to start daemon", err)
			if err := startDaemon(startArgs, vllmURL, healthPath, waitTimeout); err != nil {
				log.Fatalf("Failed to start daemon: %v", err)
			}
		} else {
			// The daemon answered wake_up, so it was asleep. Use the level it
			// reports rather than the configured one; see getSleepState.
			if err := restoreAfterWake(vllmURL, getSleepState(vllmURL)); err != nil {
				log.Fatalf("Failed to restore weights after wake up: %v", err)
			}
			log.Printf("Wake up sent, waiting for healthy state")
			if err := waitForHealthyWithPath(vllmURL, healthPath, waitTimeout); err != nil {
				log.Fatalf("vLLM health check failed after wake up: %v", err)
			}
		}
	} else {
		log.Printf("vLLM daemon is reachable at %s%s, checking sleep state", vllmURL, healthPath)
		state := getSleepState(vllmURL)
		if state == -1 {
			log.Printf("vLLM daemon is awake; nothing to do")
		} else {
			if state != 2 && state != sleepLevel {
				log.Printf("Warning: vLLM reports sleep level %d but --sleep-level is %d; using the reported level", state, sleepLevel)
			}
			if err := wakeUpVLLM(vllmURL); err != nil {
				log.Fatalf("Wake up failed: %v", err)
			}
			log.Printf("Wake up started (sleep level %d)", state)
			if err := restoreAfterWake(vllmURL, state); err != nil {
				log.Fatalf("Failed to restore weights after level-2 wake: %v", err)
			}
			log.Printf("Waiting for vLLM to be healthy after wake up")
			if err := waitForHealthyWithPath(vllmURL, healthPath, waitTimeout); err != nil {
				log.Fatalf("vLLM health check failed after wake up: %v", err)
			}
			log.Printf("Wake up complete")
		}
	}

	// Step 2: Set up reverse proxy from listenAddr to vllmURL.
	proxyURL, err := url.Parse(vllmURL)
	if err != nil {
		log.Fatalf("Invalid vLLM URL %q: %v", vllmURL, err)
	}
	proxy := httputil.NewSingleHostReverseProxy(proxyURL)

	// Modify response to disable buffering for streaming. Upstream CORS headers
	// are deliberately left in place: this wrapper has no CORS middleware of its
	// own, and llama-swap strips them when it proxies through; see issue #85.
	proxy.ModifyResponse = func(resp *http.Response) error {
		if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
			resp.Header.Set("X-Accel-Buffering", "no")
		}
		return nil
	}

	// Create HTTP server.
	srv := &http.Server{
		Addr: listenAddr,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			proxy.ServeHTTP(w, r)
		}),
	}

	// Start server in a goroutine.
	go func() {
		log.Printf("Starting vllm-wrapper serve on %s -> %s", listenAddr, vllmURL)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("ListenAndServe: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown.
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	sig := <-c
	signal.Stop(c)

	if sig == syscall.SIGTERM {
		log.Printf("SIGTERM received, putting vLLM to sleep (level %d)", sleepLevel)
		if err := sleepVLLM(vllmURL, sleepLevel); err != nil {
			log.Printf("Warning: failed to put vLLM to sleep: %v", err)
		} else {
			log.Printf("Successfully put vLLM to sleep (level %d)", sleepLevel)
		}
	}

	log.Println("Shutting down vllm-wrapper serve...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server shutdown failed: %v", err)
	}
	log.Println("Server stopped")
}

// sleepCmd implements the sleep subcommand.
func sleepCmd(args []string) {
	var (
		vllmURL    string
		sleepLevel int
		stopPID    int
	)
	fs := flag.NewFlagSet("sleep", flag.ExitOnError)
	fs.StringVar(&vllmURL, "vllm-url", "", "Base URL of vLLM server (e.g., http://127.0.0.1:8000)")
	fs.IntVar(&sleepLevel, "sleep-level", 1, "Sleep level to use (default 1)")
	fs.IntVar(&stopPID, "stop-pid", 0, "PID of the serve proxy to terminate after vLLM successfully enters sleep mode")
	fs.Parse(args)

	if vllmURL == "" {
		log.Fatalf("--vllm-url is required")
	}
	vllmURL = strings.TrimRight(vllmURL, "/")

	if err := sleepVLLM(vllmURL, sleepLevel); err != nil {
		log.Fatalf("Failed to put vLLM to sleep: %v", err)
	}

	log.Printf("Successfully put vLLM to sleep (level %d)", sleepLevel)

	if stopPID > 0 {
		if err := syscall.Kill(stopPID, syscall.SIGTERM); err != nil {
			log.Fatalf("Failed to stop serve proxy process %d: %v", stopPID, err)
		}
		log.Printf("Sent SIGTERM to serve proxy process %d", stopPID)
	}
}

// sleepVLLM sends a POST to /sleep to put the vLLM daemon to sleep.
func sleepVLLM(vllmURL string, sleepLevel int) error {
	resp, err := http.Post(
		fmt.Sprintf("%s/sleep?level=%d", vllmURL, sleepLevel),
		"application/json",
		http.NoBody,
	)
	if err != nil {
		return fmt.Errorf("failed to send sleep request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("vLLM sleep request failed with status %d: %s", resp.StatusCode, resp.Status)
	}

	return nil
}

// wakeUpVLLM sends a POST to /wake_up to wake the vLLM daemon.
func wakeUpVLLM(vllmURL string) error {
	// The wake_up endpoint may not require a body; we'll send a POST with empty body.
	resp, err := http.Post(vllmURL+"/wake_up", "application/json", strings.NewReader(""))
	if err != nil {
		return fmt.Errorf("failed to POST /wake_up: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		// Some versions might return 204 No Content.
		return fmt.Errorf("/wake_up returned unexpected status %d: %s", resp.StatusCode, resp.Status)
	}
	return nil
}

// isSleepingVLLM queries GET /is_sleeping to get sleep state.
func isSleepingVLLM(vllmURL string) (bool, error) {
	resp, err := http.Get(vllmURL + "/is_sleeping")
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("/is_sleeping returned status %d: %s", resp.StatusCode, resp.Status)
	}
	var out struct {
		IsSleeping bool `json:"is_sleeping"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return false, fmt.Errorf("decoding /is_sleeping response: %w", err)
	}
	return out.IsSleeping, nil
}

// getSleepState reports the engine's sleep state as an integer:
//
//	-1  the engine is awake
//	 0  the engine is asleep (level 0 pause)
//	 1  the engine is asleep at level 1 (weights in CPU memory)
//	 2  the engine is asleep at level 2 (weights discarded), or the state
//	    could not be determined
//
// Level 2 is returned both when the engine reports a level-2 sleep and when
// its state is unknown: an unknown state must be treated as the most
// conservative case, because a level-2 restore on an awake engine is a
// harmless no-op while skipping it on a level-2 sleep leaves the model
// without weights. The level comes from vllm:engine_sleep_state in /metrics,
// which reports the engine's actual state rather than what this wrapper
// requested, so a mismatch between the serve and sleep subcommands cannot
// cause a wrong restore. Falls back to /is_sleeping when /metrics is
// unavailable or does not expose the metric.
func getSleepState(vllmURL string) int {
	if state, ok := sleepStateFromMetrics(vllmURL); ok {
		return state
	}
	if asleep, err := isSleepingVLLM(vllmURL); err == nil && !asleep {
		return -1
	}
	// Asleep, or the state could not be determined either way.
	return 2
}

// sleepStateFromMetrics parses vllm:engine_sleep_state from the /metrics
// endpoint. The metric is a gauge carrying a sleep_state label whose value is
// 1 for the current state: "awake", "weights_offloaded" (level 1) or
// "discard_all" (level 2). Reports false when the metric is unavailable, so
// callers can fall back to /is_sleeping.
func sleepStateFromMetrics(vllmURL string) (int, bool) {
	resp, err := http.Get(vllmURL + "/metrics")
	if err != nil {
		return 0, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, false
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "vllm:engine_sleep_state{") {
			continue
		}
		// Format: vllm:engine_sleep_state{sleep_state="awake"} 1.0
		end := strings.LastIndex(line, "}")
		if end < 0 {
			continue
		}
		value := strings.TrimSpace(line[end+1:])
		if v, err := strconv.ParseFloat(value, 64); err != nil || v != 1 {
			continue
		}
		switch {
		case strings.Contains(line, `sleep_state="awake"`):
			return -1, true
		case strings.Contains(line, `sleep_state="weights_offloaded"`):
			return 1, true
		case strings.Contains(line, `sleep_state="discard_all"`):
			return 2, true
		}
	}
	if err := scanner.Err(); err != nil {
		return 0, false
	}
	return 0, false
}

// restoreAfterWake finishes a level-2 wake-up.
func restoreAfterWake(vllmURL string, sleepLevel int) error {
	if sleepLevel != 2 {
		return nil
	}
	if err := wakeUpVLLMTags(vllmURL, "weights"); err != nil {
		return fmt.Errorf("waking weights: %w", err)
	}
	if err := reloadWeightsVLLM(vllmURL); err != nil {
		return err
	}
	log.Printf("Weights reloaded after level-2 wake")
	if err := wakeUpVLLMTags(vllmURL, "kv_cache"); err != nil {
		return fmt.Errorf("waking kv cache: %w", err)
	}
	return nil
}

// wakeUpVLLMTags sends POST /wake_up?tags=<tag> for a partial wake-up. Only
// the sleep endpoints understand "tags"; when the server rejects it we fall
// back to the plain wake_up used by callers that do not need the split.
func wakeUpVLLMTags(vllmURL string, tag string) error {
	resp, err := http.Post(
		fmt.Sprintf("%s/wake_up?tags=%s", vllmURL, url.QueryEscape(tag)),
		"application/json",
		http.NoBody,
	)
	if err != nil {
		return fmt.Errorf("failed to POST /wake_up?tags=%s: %w", tag, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		return fmt.Errorf("/wake_up?tags=%s returned unexpected status %d: %s", tag, resp.StatusCode, resp.Status)
	}
	return nil
}

// reloadWeightsVLLM asks the engine to reload model weights in place from
// disk. Required after a level-2 sleep, which discards the weights instead
// of backing them up; a plain wake_up only re-allocates memory.
func reloadWeightsVLLM(vllmURL string) error {
	resp, err := http.Post(
		vllmURL+"/collective_rpc",
		"application/json",
		strings.NewReader(`{"method":"reload_weights"}`),
	)
	if err != nil {
		return fmt.Errorf("failed to POST /collective_rpc reload_weights: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		return fmt.Errorf("/collective_rpc reload_weights returned status %d: %s", resp.StatusCode, resp.Status)
	}
	return nil
}

// waitForHealthyWithPath polls the vLLM daemon's health endpoint at the given path.
func waitForHealthyWithPath(vllmURL string, healthPath string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		// Create a request with context.
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, vllmURL+healthPath, nil)
		if err != nil {
			return err
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			// If context canceled, break.
			if ctx.Err() != nil {
				return ctx.Err()
			}
			// Wait a bit before retrying.
			time.Sleep(1 * time.Second)
			continue
		}
		if resp.StatusCode == http.StatusOK {
			resp.Body.Close()
			return nil
		}
		resp.Body.Close()
		// Wait a bit before retrying.
		time.Sleep(1 * time.Second)
	}
	return ctx.Err()
}

// checkHealthy sends a GET request to the health path and returns nil if the response status is 200 OK.
func checkHealthy(vllmURL string, healthPath string) error {
	resp, err := http.Get(vllmURL + healthPath)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	return nil
}

// startJournalForwarder forwards new log entries from a user systemd unit to stdout.
func startJournalForwarder(ctx context.Context, unit string) error {
	cmd := exec.CommandContext(
		ctx,
		"journalctl",
		"--user-unit="+unit,
		"--follow",
		"--lines=0",
		"--output=cat",
		"--no-pager",
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start journalctl for %s: %w", unit, err)
	}

	go func() {
		err := cmd.Wait()
		if ctx.Err() == nil && err != nil {
			log.Printf("Journal forwarding for %s stopped: %v", unit, err)
		}
	}()

	return nil
}

// startDaemon executes the start command and waits for the vLLM daemon to become healthy.
func startDaemon(startArgs []string, vllmURL string, healthPath string, waitTimeout time.Duration) error {
	cmd := exec.Command(startArgs[0], startArgs[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start daemon command: %w", err)
	}
	// Wait for healthy state.
	log.Printf("Started daemon with PID %d, waiting for healthy state", cmd.Process.Pid)
	err := waitForHealthyWithPath(vllmURL, healthPath, waitTimeout)
	if err != nil {
		// If we fail to become healthy, kill the started process.
		_ = cmd.Process.Kill()
		return fmt.Errorf("daemon did not become healthy: %w", err)
	}
	// Daemon is healthy, we don't wait for the command to exit (it should keep running).
	// We'll let it run; the wrapper will not kill it on exit.
	return nil
}
