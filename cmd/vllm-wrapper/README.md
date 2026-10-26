# vllm-wrapper

`vllm-wrapper` is a standalone helper program designed to be used as a model's `cmd` and `cmdStop` in llama-swap configurations for vLLM servers that have been started with `--enable-sleep-mode`.

It provides two subcommands:

- `serve`: Used as a model's `cmd`. Manages the vLLM daemon lifecycle: if the daemon is not running, starts it using the provided start command; if running but asleep, wakes it up; waits for it to be healthy, then runs a reverse proxy from a local port to the vLLM upstream.
- `sleep`: Used as a model's `cmdStop`. Sends a sleep request to the vLLM daemon to free VRAM while keeping the process alive.

## Why use this?

When using vLLM with llama-swap, you can leverage vLLM's sleep mode to drastically reduce swap-in times. Instead of stopping and starting the vLLM process (which incurs a cold start), you can put the vLLM daemon to sleep when not in use (via `cmdStop`) and wake it up when needed (via `cmd`). This keeps the vLLM process running, preserving the GPU context and allowing for near-instant wake-ups.

## Prerequisites

- vLLM server must be started with `--enable-sleep-mode`.
- The vLLM server must be reachable at the URL provided to the wrapper.
- To enable automatic start‑if‑not‑running, provide the daemon executable and arguments after `--`.

## Installation

Build the binary from source:

```bash
go build -o vllm-wrapper ./cmd/vllm-wrapper
```

Or install via `go install`:

```bash
go install ./cmd/vllm-wrapper
```

## Usage in llama-swap

### As a model's `cmd`

#### Daemon startup

For native vLLM installations or when `llama-swap` splits the command into separate arguments, use the `--` separator. Everything after `--` is treated as the daemon executable and its arguments, launched directly without `sh -c`:

```yaml
models:
  my-vllm-model:
    cmd: |
      vllm-wrapper serve
      --vllm-url http://127.0.0.1:8000
      --listen :${PORT}
      --wait-timeout 600s
      --
      ${vllm}
      serve
      ${model_path}
      --host 127.0.0.1
      --port 8000
      --max-model-len ${context_size}
      --enable-sleep-mode
```

Benefits of argv-based startup:
- Enables native vLLM without Docker.
- Supports vLLM executables defined through llama-swap macros.
- Allows readable multiline commands in YAML.
- Preserves individual vLLM options (can be commented out).
- Avoids shell parsing overhead.
- Correctly preserves arguments containing spaces.

### As a model's `cmdStop`

Configure your model's `cmdStop` to invoke `vllm-wrapper sleep`:

```yaml
models:
  my-vllm-model:
    cmdStop: |
      vllm-wrapper sleep
      --vllm-url http://127.0.0.1:8000
      --stop-pid ${PID}
    # Optional flags:
    #   --sleep-level: sleep level to use (default: 1)
    #   --stop-pid: PID of the serve proxy to terminate after a successful sleep request
```

When llama-swap stops the model, it will:
1. Send a sleep request to the vLLM daemon (POST to `/sleep?level=<level>`).
2. If `--stop-pid` is provided, send SIGTERM to the specified `vllm-wrapper serve` process after the sleep request succeeds.
3. Exit with status 0, leaving the vLLM daemon running but asleep while allowing llama-swap to complete the unload operation.

### Choosing a sleep level

Level 1 keeps the model weights in CPU memory, so waking is fast but needs enough RAM for a copy of the weights. Level 2 discards both the weights and the KV cache, freeing the most GPU memory and using almost no host memory, at the cost of reloading the weights from disk on wake.

`--sleep-level` selects the level when sleeping and is advisory on wake, where the wrapper uses the level vLLM reports. Passing it to both commands is still recommended so the configured intent matches:

```yaml
models:
  my-vllm-model:
    cmd: |
      vllm-wrapper serve
      --vllm-url http://127.0.0.1:8000
      --listen :${PORT}
      --sleep-level 2
      --
      ${vllm}
      serve
      ${model_path}
      --host 127.0.0.1
      --port 8000
      --enable-sleep-mode
    cmdStop: |
      vllm-wrapper sleep
      --vllm-url http://127.0.0.1:8000
      --sleep-level 2
      --stop-pid ${PID}
```

## Example Configuration

Here is a complete example using vLLM with sleep mode, demonstrating cold start on first swap‑in and fast wake‑up on subsequent swaps:

```yaml
models:
  qwen-7b-chat:
    cmd: |
      vllm-wrapper serve
      --vllm-url http://127.0.0.1:8000
      --listen :${PORT}
      --
      ${vllm}
      serve
      ${model_path}
      --host 127.0.0.1
      --port 8000
      --enable-sleep-mode
    cmdStop: vllm-wrapper sleep --vllm-url http://127.0.0.1:8000 --stop-pid ${PID}
    # You may also want to set a TTL to automatically unload after a period of inactivity:
    ttl: 3600   # unload after 1 hour of inactivity
```

## Systemd setup for native vLLM startup

`llama-swap` runs as the system service user `llama`, while vLLM is started as a transient user service with `systemd-run --user`.

Enable the user systemd manager:

```bash
sudo loginctl enable-linger llama

uid=$(id -u llama)
sudo systemctl start "user@${uid}.service"
```

Expose the user D-Bus session to `llama.service`:

```bash
sudo mkdir -p /etc/systemd/system/llama.service.d

sudo tee /etc/systemd/system/llama.service.d/user-bus.conf >/dev/null <<EOF
[Unit]
Wants=user@${uid}.service
After=user@${uid}.service

[Service]
Environment=XDG_RUNTIME_DIR=/run/user/${uid}
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/${uid}/bus
EOF

sudo systemctl daemon-reload
sudo systemctl restart llama.service
```

Use `systemd-run` in the `llama-swap` model command:

```yaml
cmd: |
  vllm-wrapper serve
  --vllm-url http://127.0.0.1:18000
  --listen :${PORT}
  --wait-timeout 5m
  --journal-unit: vllm-qwen.service
  # optional systemd user unit whose new journal entries are forwarded to the wrapper's stdout, making vLLM logs available through llama-swap's upstream log stream.
  --
  systemd-run
  --user
  --unit=vllm-qwen
  --collect
  --property=Restart=no
  --property=StandardOutput=journal
  --property=StandardError=journal
  --setenv=HF_HOME
  --setenv=VLLM_SERVER_DEV_MODE
  --
  ${vllm}
  serve
  ${model}
  --port 18000
  --enable-sleep-mode
```

Environment variables required by vLLM must be passed explicitly with `--setenv`.

Check the service status:

```bash
uid=$(id -u llama)

sudo -u llama env \
  XDG_RUNTIME_DIR="/run/user/${uid}" \
  DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${uid}/bus" \
  systemctl --user status vllm-qwen.service
```

Follow the vLLM logs:

```bash
uid=$(id -u llama)

sudo -u llama env \
  XDG_RUNTIME_DIR="/run/user/${uid}" \
  DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${uid}/bus" \
  journalctl --user \
    -u vllm-qwen.service \
    -f \
    -o cat
```

## How it works

### serve subcommand

1. **Health check**: Sends a GET request to `${vllm-url}${health-path}` (default `/health`). If the response is HTTP 200, the daemon is reachable, so we proceed to step 2. Note that vLLM answers `/health` with 200 even while the engine is asleep, so reachability alone does not mean the engine is awake.
2. **Sleep check**: Reads the engine's sleep state. It first tries `vllm:engine_sleep_state` from `${vllm-url}/metrics`, whose `sleep_state` label reports what the engine actually did: `awake`, `weights_offloaded` (level 1) or `discard_all` (level 2). If `/metrics` is unavailable or does not expose the metric, it falls back to `${vllm-url}/is_sleeping`, which reports only asleep/awake. If the state cannot be determined at all, the wrapper treats it as a level‑2 sleep. If the engine is awake, nothing more is needed; proceed to step 4.
3. **Wake up**: Wakes the engine according to the level it reported. A level‑2 sleep discarded the weights, so the wrapper runs vLLM's documented restore sequence — `wake_up?tags=weights`, `collective_rpc` with `reload_weights`, and `wake_up?tags=kv_cache` — without a preceding plain `/wake_up`, which would restore from a backup that does not exist. Other levels wake with a single `/wake_up`. If the wake‑up succeeds (HTTP 200 or 204), wait for the daemon to become healthy again using the full `--wait-timeout` (a level‑2 reload reads the weights from disk), then proceed to step 4. If the server rejects the tagged request with a 4xx, it predates v0.8.3 and lacks the `tags` parameter; the wrapper falls back to a plain `/wake_up` followed by the weight reload. A 5xx is treated as a real failure and does not trigger the fallback.
4. **Start daemon**: If the daemon is unreachable, use the command specified after `--` (argv-based, launched via `exec.Command`). The wrapper starts the command as a child process, then waits for the daemon to become healthy by polling the health path.
5. **Reverse proxy**: Once the daemon is healthy, start an HTTP server listening on `${PORT}` (or the address provided to `--listen`) that proxies all requests to the vLLM upstream URL. The proxy preserves streaming responses by setting `X-Accel-Buffering: no`.

### sleep subcommand

1. Sends a POST request to `${vllm-url}/sleep?level=<level>` where `<level>` is the sleep level (default 1). vLLM reads the level from the query string only; a JSON body is ignored.
2. If `--stop-pid` is provided, sends SIGTERM to the specified `vllm-wrapper serve` process after the sleep request succeeds.
3. Exits with status 0, leaving the vLLM daemon running in sleep mode.

## Notes

- The wrapper uses standard library only (no external dependencies).
- It is designed to be simple and robust.
- For production use, ensure the vLLM daemon is properly managed (e.g., restarted if it crashes) outside of this wrapper.
- The wrapper does not handle TLS certificates; if your vLLM server uses HTTPS, provide the appropriate URL and ensure the system's root CAs are configured.
- On SIGTERM, the wrapper sends a sleep request to vLLM (using the configured sleep level) before shutting down, then exits cleanly without killing the vLLM daemon.
- The `/sleep`, `/wake_up` and `/is_sleeping` endpoints are only enabled when vLLM runs with `VLLM_SERVER_DEV_MODE=1`. Level 2 additionally requires `--enable-sleep-mode`.
- `--sleep-level` chooses the level used when *sleeping*. On wake the wrapper uses the level the engine reports, so the two subcommands cannot disagree about how to restore. A mismatch is logged as a warning.
- When `/metrics` is unavailable the wrapper falls back to `/is_sleeping`, which cannot distinguish levels. In that case a sleep is assumed to be level 2, so the full restore sequence runs; that is a no-op on an awake engine and is the safe choice for a level-2 sleep.
- Level-2 wakes use the `wake_up` `tags` parameter added in vLLM v0.8.3 (PR #15500). Older releases reject the tagged request; the wrapper detects the 4xx and falls back to a plain `/wake_up`, then reloads the weights from disk.
- The reverse proxy uses Go's default HTTP transport (`http.DefaultTransport`), so there is no fixed limit on how long it waits for response headers from vLLM. Long-running generations are bounded only by the client's own timeout.

## Building

```bash
go build -o vllm-wrapper ./cmd/vllm-wrapper
```

## Running Tests

```bash
go test ./...
```
