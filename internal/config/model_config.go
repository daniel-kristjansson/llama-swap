package config

import (
	"errors"
	"fmt"
	"runtime"
	"strconv"
	"strings"
)

const (
	MODEL_CONFIG_DEFAULT_TTL   = -1
	MODEL_CONFIG_DEFAULT_PROXY = "http://localhost:${PORT}"
	comfyUIConcurrencyLimit    = 999

	// ComfyUIModelID identifies the model used by the /comfyui endpoint.
	ComfyUIModelID = "comfyui_auto"
)

var validModalities = map[string]struct{}{
	"text":  {},
	"audio": {},
	"image": {},
	"video": {},
}

// ModelCapConfig defines what modalities and features a model supports.
// Used in /v1/models to inform clients. An empty block (all zero values) is
// treated as not configured.
type ModelCapConfig struct {
	In       []string `yaml:"in" json:"in,omitempty"`
	Out      []string `yaml:"out" json:"out,omitempty"`
	Tools    bool     `yaml:"tools" json:"tools,omitempty"`
	Reranker bool     `yaml:"reranker" json:"reranker,omitempty"`
	Context  int      `yaml:"context" json:"context,omitempty"`

	// DisableAuto turns off automatic capability discovery for this model.
	// It is a switch, not a capability, so Empty ignores it and it is never
	// rendered into /v1/models.
	DisableAuto bool `yaml:"disableAuto" json:"-"`
}

// Empty returns true when all capability fields are at their zero values.
// DisableAuto is deliberately excluded: a block that only turns discovery off
// still advertises nothing, so it must render as if it were absent.
func (c ModelCapConfig) Empty() bool {
	return len(c.In) == 0 && len(c.Out) == 0 && !c.Tools && !c.Reranker && c.Context == 0
}

// Merge returns c with every field left at its zero value filled in from
// auto. Configured values always win, field by field, so a model that sets
// only capabilities.tools still picks up a discovered context length.
//
// A field explicitly set to its zero value in the config is indistinguishable
// from an omitted one, so `tools: false` cannot override a discovered true.
// Use capabilities.disableAuto to suppress discovery for the model instead.
//
// DisableAuto is carried over from c unchanged; auto never sets it.
func (c ModelCapConfig) Merge(auto ModelCapConfig) ModelCapConfig {
	merged := c
	if len(merged.In) == 0 {
		merged.In = auto.In
	}
	if len(merged.Out) == 0 {
		merged.Out = auto.Out
	}
	if !merged.Tools {
		merged.Tools = auto.Tools
	}
	if !merged.Reranker {
		merged.Reranker = auto.Reranker
	}
	if merged.Context == 0 {
		merged.Context = auto.Context
	}
	return merged
}

// Validate checks that all modality values are recognized and context is
// non-negative. Returns an error if any value is invalid.
func (c ModelCapConfig) Validate() error {
	for _, m := range c.In {
		if _, ok := validModalities[m]; !ok {
			return fmt.Errorf("capabilities.in: invalid modality %q, must be one of: text, audio, image, video", m)
		}
	}
	for _, m := range c.Out {
		if _, ok := validModalities[m]; !ok {
			return fmt.Errorf("capabilities.out: invalid modality %q, must be one of: text, audio, image, video", m)
		}
	}
	if c.Context < 0 {
		return errors.New("capabilities.context: must be >= 0")
	}
	return nil
}

// TimeoutsConfig holds timeout settings for proxy connections
// 0 = no timeout
type TimeoutsConfig struct {
	Connect        int `yaml:"connect"`
	KeepAlive      int `yaml:"keepalive"`
	ResponseHeader int `yaml:"responseHeader"`
	TLSHandshake   int `yaml:"tlsHandshake"`
	ExpectContinue int `yaml:"expectContinue"`
	IdleConn       int `yaml:"idleConn"`
}

// CompatConfig holds compatibility settings for upstream applications.
type CompatConfig struct {
	// IgnoreWebsockets prevents websocket connections from participating in
	// model lifecycle activity such as swapping, concurrency, and TTL tracking.
	IgnoreWebsockets bool `yaml:"ignoreWebsockets"`
}

type ModelConfig struct {
	Cmd           string   `yaml:"cmd"`
	CmdStop       string   `yaml:"cmdStop"`
	Proxy         string   `yaml:"proxy"`
	Aliases       []string `yaml:"aliases"`
	Env           []string `yaml:"env"`
	CheckEndpoint string   `yaml:"checkEndpoint"`
	UnloadAfter   int      `yaml:"ttl"`
	UnloadTimeout int      `yaml:"unloadTimeout"`
	Unlisted      bool     `yaml:"unlisted"`
	UseModelName  string   `yaml:"useModelName"`

	// #179 for /v1/models
	Name        string `yaml:"name"`
	Description string `yaml:"description"`

	// Limit concurrency of HTTP requests to process
	ConcurrencyLimit int `yaml:"concurrencyLimit"`

	// Model filters see issue #174
	Filters ModelFilters `yaml:"filters"`

	// Macros: see #264
	// Model level macros take precedence over the global macros
	Macros MacroList `yaml:"macros"`

	// Metadata: see #264
	// Arbitrary metadata that can be exposed through the API
	Metadata map[string]any `yaml:"metadata"`

	// override global setting
	SendLoadingState *bool `yaml:"sendLoadingState"`

	// Timeout settings for proxy connections
	Timeouts TimeoutsConfig `yaml:"timeouts"`

	// DisableKeepAlives opens a new upstream connection for every request
	// instead of reusing pooled ones. Works around upstreams that advertise
	// keep-alive and then close the connection (#1205).
	DisableKeepAlives bool `yaml:"disableKeepAlives"`

	// Compatibility settings for upstream applications.
	Compat CompatConfig `yaml:"compat"`

	// Capabilities defines what modalities and features the model supports.
	Capabilities ModelCapConfig `yaml:"capabilities"`

	// Copy of HealthCheckTimeout from global config
	HealthCheckTimeout int `yaml:"healthCheckTimeout"`

	// MinAvailableMemory, when > 0, gates model startup on system memory:
	// the start command is not executed until the OS reports at least this
	// much available memory (MemAvailable on Linux), preventing a model
	// load from OOM-killing unrelated processes (e.g. a desktop session)
	// when a previously "stopped" model is still holding memory.
	MinAvailableMemory string `yaml:"minAvailableMemory"`

	// WaitAvailableMemory bounds how long startup blocks for the above
	// memory to free up, in seconds. 0 (default) falls back to
	// HealthCheckTimeout, then to 300s.
	WaitAvailableMemory int `yaml:"waitAvailableMemory"`

	// parsed at config load
	minAvailableMemoryBytes uint64
}

// MinAvailableMemoryBytes returns the parsed startup memory gate threshold
// in bytes. 0 means the gate is disabled.
func (m *ModelConfig) MinAvailableMemoryBytes() uint64 {
	return m.minAvailableMemoryBytes
}

// parseMinAvailableMemory parses values like "20G", "512M", "1.5G" (binary
// suffixes K/M/G/T) or plain byte counts. Empty string means disabled.
func parseMinAvailableMemory(s string) (uint64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, nil
	}
	var mult uint64 = 1
	if r := s[len(s)-1]; r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' {
		switch strings.ToUpper(s[len(s)-1:]) {
		case "K":
			mult = 1 << 10
		case "M":
			mult = 1 << 20
		case "G":
			mult = 1 << 30
		case "T":
			mult = 1 << 40
		default:
			return 0, fmt.Errorf("minAvailableMemory: unsupported suffix %q (use K, M, G or T)", s[len(s)-1:])
		}
		s = s[:len(s)-1]
	}
	n, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil || n < 0 {
		return 0, fmt.Errorf("minAvailableMemory: invalid value %q", s)
	}
	return uint64(n * float64(mult)), nil
}

func (m *ModelConfig) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type rawModelConfig ModelConfig
	defaults := rawModelConfig{
		Cmd:              "",
		CmdStop:          "",
		Proxy:            MODEL_CONFIG_DEFAULT_PROXY,
		Aliases:          []string{},
		Env:              []string{},
		CheckEndpoint:    "/health",
		UnloadAfter:      MODEL_CONFIG_DEFAULT_TTL, // use GlobalTTL
		UnloadTimeout:    0,                        // use global UnloadTimeout
		Unlisted:         false,
		UseModelName:     "",
		ConcurrencyLimit: 0,
		Name:             "",
		Description:      "",

		// matches http.DefaultTransport
		Timeouts: TimeoutsConfig{
			Connect:        30,
			KeepAlive:      30,
			ResponseHeader: 0,
			TLSHandshake:   10,
			ExpectContinue: 1,
			IdleConn:       90,
		},
	}

	// the default cmdStop to taskkill /f /t /pid ${PID}
	if runtime.GOOS == "windows" {
		defaults.CmdStop = "taskkill /f /t /pid ${PID}"
	}

	if err := unmarshal(&defaults); err != nil {
		return err
	}

	bytes, err := parseMinAvailableMemory(defaults.MinAvailableMemory)
	if err != nil {
		return err
	}
	defaults.minAvailableMemoryBytes = bytes

	*m = ModelConfig(defaults)
	return nil
}

func (m *ModelConfig) SanitizedCommand() ([]string, error) {
	return SanitizeCommand(m.Cmd)
}

// ModelFilters embeds Filters and adds legacy support for strip_params field
// See issue #174
type ModelFilters struct {
	Filters `yaml:",inline"`
}

func (m *ModelFilters) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type rawModelFilters ModelFilters
	defaults := rawModelFilters{}

	if err := unmarshal(&defaults); err != nil {
		return err
	}

	// Try to unmarshal with the old field name for backwards compatibility
	if defaults.StripParams == "" {
		var legacy struct {
			StripParams string `yaml:"strip_params"`
		}
		if legacyErr := unmarshal(&legacy); legacyErr != nil {
			return errors.New("failed to unmarshal legacy filters.strip_params: " + legacyErr.Error())
		}
		defaults.StripParams = legacy.StripParams
	}

	*m = ModelFilters(defaults)
	return nil
}

// SanitizedStripParams wraps Filters.SanitizedStripParams for backwards compatibility
// Returns ([]string, error) to match existing API
func (f ModelFilters) SanitizedStripParams() ([]string, error) {
	return f.Filters.SanitizedStripParams(), nil
}
