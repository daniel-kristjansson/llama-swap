package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

// sleepLevel represents the vLLM sleep level.
type sleepLevel int

const (
	sleepLevel0 sleepLevel = 0
	sleepLevel1 sleepLevel = 1
	sleepLevel2 sleepLevel = 2
)

// wakeStep is one HTTP call in a wake sequence. It is parsed from a
// "METHOD /path [body]" string, for example:
//
//	POST /wake_up {"tags":["weights"]}
//	POST /collective_rpc {"method":"reload_weights"}
//	POST /wake_up {"tags":["kv_cache"]}
//	GET /is_sleeping
//
// The method defaults to POST when omitted. The path may include a query
// string. The optional body must be valid JSON and is sent verbatim.
type wakeStep struct {
	Method string
	Path   string
	Body   string
}

// String renders the step back into its flag syntax, for logging.
func (s wakeStep) String() string {
	if s.Body == "" {
		return s.Method + " " + s.Path
	}
	return s.Method + " " + s.Path + " " + s.Body
}

// parseWakeStep parses a "METHOD /path [body]" wake step definition.
func parseWakeStep(spec string) (wakeStep, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return wakeStep{}, fmt.Errorf("wake step is empty")
	}

	fields := strings.SplitN(spec, " ", 2)
	method := strings.ToUpper(fields[0])
	rest := ""
	if len(fields) == 2 {
		rest = strings.TrimSpace(fields[1])
	}

	// A first token that does not look like an HTTP method is treated as the
	// path, so plain "/wake_up" works and defaults to POST.
	if !strings.HasPrefix(method, "/") {
		switch method {
		case http.MethodGet, http.MethodPost, http.MethodPut, http.MethodPatch:
			// method consumed, rest is path plus optional body
		default:
			return wakeStep{}, fmt.Errorf("unsupported wake step method %q (use GET, POST, PUT or PATCH)", fields[0])
		}
	} else {
		method = http.MethodPost
		rest = strings.TrimSpace(spec)
	}

	if rest == "" {
		return wakeStep{}, fmt.Errorf("wake step %q is missing a path", spec)
	}

	step := wakeStep{Method: method}
	if idx := strings.Index(rest, " "); idx >= 0 {
		step.Path = strings.TrimSpace(rest[:idx])
		step.Body = strings.TrimSpace(rest[idx+1:])
	} else {
		step.Path = rest
	}

	if !strings.HasPrefix(step.Path, "/") {
		return wakeStep{}, fmt.Errorf("wake step path %q must start with \"/\"", step.Path)
	}
	if step.Body != "" && !json.Valid([]byte(step.Body)) {
		return wakeStep{}, fmt.Errorf("wake step %q body is not valid JSON", spec)
	}
	return step, nil
}

// wakeStepsFlag collects repeated --wake-step flags in order.
type wakeStepsFlag []wakeStep

// String implements flag.Value.
func (f *wakeStepsFlag) String() string {
	parts := make([]string, len(*f))
	for i, s := range *f {
		parts[i] = s.String()
	}
	return strings.Join(parts, "; ")
}

// Set implements flag.Value, appending each occurrence in order.
func (f *wakeStepsFlag) Set(spec string) error {
	step, err := parseWakeStep(spec)
	if err != nil {
		return err
	}
	*f = append(*f, step)
	return nil
}

// defaultWakeSteps returns the wake sequence implied by a sleep level when
// no explicit --wake-step flags are given.
//
// Level 2 discards the model weights, so the documented vLLM sequence is
// needed: allocate weight memory, reload the weights from disk, then
// allocate the KV cache. Waking without tags would allocate the KV cache
// before the weights land, raising peak VRAM. Level 0 and 1 both just need
// an untagged wake_up: level 0 frees nothing, and level 1 restores weights
// from its host-RAM backup.
func defaultWakeSteps(level int) []wakeStep {
	if level != int(sleepLevel2) {
		return []wakeStep{{Method: http.MethodPost, Path: "/wake_up"}}
	}
	return []wakeStep{
		{Method: http.MethodPost, Path: "/wake_up", Body: `{"tags":["weights"]}`},
		{Method: http.MethodPost, Path: "/collective_rpc", Body: `{"method":"reload_weights"}`},
		{Method: http.MethodPost, Path: "/wake_up", Body: `{"tags":["kv_cache"]}`},
	}
}

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
		vllmURL         string
		listenAddr      string
		sleepLevel      int
		healthPath      string
		waitTimeout     time.Duration
		journalUnit     string
		wakeSteps       wakeStepsFlag
		wakeStepTimeout time.Duration
	)
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	fs.StringVar(&vllmURL, "vllm-url", "", "Base URL of vLLM server (e.g., http://127.0.0.1:8000)")
	fs.StringVar(&listenAddr, "listen", "", "Address to listen on (e.g., :$PORT)")
	fs.IntVar(&sleepLevel, "sleep-level", 1, "Sleep level to use when sleeping: 0 pauses scheduling only, 1 offloads weights to host RAM, 2 discards weights (default 1)")
	fs.StringVar(&healthPath, "health-path", "/health", "Health check path (default /health)")
	fs.DurationVar(&waitTimeout, "wait-timeout", 120*time.Second, "Timeout waiting for daemon to become healthy")
	fs.StringVar(&journalUnit, "journal-unit", "", "User systemd unit whose logs should be forwarded to stdout")
	fs.Var(&wakeSteps, "wake-step", "HTTP call to run when waking a sleeping daemon, as 'METHOD /path [json-body]' (repeatable, in order). Defaults depend on --sleep-level; level 2 defaults to wake_up{tags:[weights]}, collective_rpc reload_weights, wake_up{tags:[kv_cache]}")
	fs.DurationVar(&wakeStepTimeout, "wake-step-timeout", 10*time.Minute, "Timeout for each individual --wake-step request")
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

	if len(wakeSteps) == 0 {
		wakeSteps = defaultWakeSteps(sleepLevel)
	}

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
		if err := runWakeSteps(vllmURL, wakeSteps, wakeStepTimeout); err != nil {
			// Wake up failed (e.g., connection refused), assume daemon not running, try to start it.
			log.Printf("Wake up failed: %v, attempting to start daemon", err)
			if err := startDaemon(startArgs, vllmURL, healthPath, waitTimeout); err != nil {
				log.Fatalf("Failed to start daemon: %v", err)
			}
		} else {
			log.Printf("Wake up sent, waiting for healthy state")
			if err := waitForHealthyWithPath(vllmURL, healthPath, waitTimeout); err != nil {
				log.Fatalf("vLLM health check failed after wake up: %v", err)
			}
		}
	} else {
		// Reachable (liveness ok). Note that /health answers 200 even while
		// the engine is asleep, so query the sleep state directly.
		log.Printf("vLLM daemon is reachable at %s%s, checking sleep state", vllmURL, healthPath)
		asleep, err := isSleepingVLLM(vllmURL)
		if err != nil {
			// /is_sleeping is a dev endpoint; if it is unavailable, fall back
			// to an unconditional wake, which is a no-op when already awake.
			log.Printf("Warning: could not query sleep state (%v), waking up unconditionally", err)
			asleep = true
		}
		if !asleep {
			log.Printf("vLLM daemon is awake; nothing to do")
		} else {
			if err := runWakeSteps(vllmURL, wakeSteps, wakeStepTimeout); err != nil {
				log.Fatalf("Wake up failed: %v", err)
			}
			log.Printf("Wake up complete")
			// A level-2 wake re-loads weights from disk and can take a long
			// time; use the full wait timeout rather than a short poll.
			log.Printf("Waiting for vLLM to be healthy after wake up")
			if err := waitForHealthyWithPath(vllmURL, healthPath, waitTimeout); err != nil {
				log.Fatalf("vLLM health check failed after wake up: %v", err)
			}
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

	// Prepare sleep request body.
	body := map[string]int{"level": sleepLevel}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		log.Fatalf("Failed to marshal sleep request: %v", err)
	}

	// Send POST to /sleep endpoint. NOTE: the vLLM /sleep endpoint reads
	// "level" from the query string, not the JSON body — send it there
	// (keeping the body for compatibility/inspection).
	resp, err := http.Post(fmt.Sprintf("%s/sleep?level=%d", vllmURL, sleepLevel), "application/json", strings.NewReader(string(jsonBody)))
	if err != nil {
		log.Fatalf("Failed to send sleep request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Fatalf("vLLM sleep request failed with status %d: %v", resp.StatusCode, resp.Status)
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
	body := map[string]int{"level": sleepLevel}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("failed to marshal sleep request: %w", err)
	}

	// Same as sleepCmd: "level" must ride in the query string.
	resp, err := http.Post(fmt.Sprintf("%s/sleep?level=%d", vllmURL, sleepLevel), "application/json", strings.NewReader(string(jsonBody)))
	if err != nil {
		return fmt.Errorf("failed to send sleep request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("vLLM sleep request failed with status %d: %s", resp.StatusCode, resp.Status)
	}

	return nil
}

// runWakeSteps executes the configured wake sequence in order and stops at
// the first failing step. Each step is an independent HTTP call, which lets
// callers express multi-step wakes such as vLLM's level-2 sequence without
// hardcoding endpoint names here.
func runWakeSteps(vllmURL string, steps []wakeStep, timeout time.Duration) error {
	if len(steps) == 0 {
		return fmt.Errorf("wake sequence is empty")
	}
	for i, step := range steps {
		log.Printf("Wake step %d/%d: %s", i+1, len(steps), step.String())
		if err := doWakeStep(vllmURL, step, timeout); err != nil {
			return fmt.Errorf("wake step %d/%d (%s): %w", i+1, len(steps), step.String(), err)
		}
	}
	return nil
}

// doWakeStep performs a single wake step, accepting 200 and 204 as success.
// A reload_weights step reads the weights from disk and can take minutes, so
// the caller-supplied timeout is generous rather than a short default.
func doWakeStep(vllmURL string, step wakeStep, timeout time.Duration) error {
	var bodyReader io.Reader
	if step.Body != "" {
		bodyReader = strings.NewReader(step.Body)
	}
	req, err := http.NewRequest(step.Method, vllmURL+step.Path, bodyReader)
	if err != nil {
		return fmt.Errorf("building request: %w", err)
	}
	if step.Body != "" {
		req.Header.Set("Content-Type", "application/json")
	}

	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		return fmt.Errorf("returned unexpected status %d: %s", resp.StatusCode, resp.Status)
	}
	return nil
}

// isSleepingVLLM queries GET /is_sleeping and reports whether the engine
// is currently asleep.
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
