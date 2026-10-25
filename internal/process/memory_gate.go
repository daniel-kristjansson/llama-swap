package process

import (
	"context"
	"fmt"
	"time"

	"github.com/shirou/gopsutil/v4/mem"
)

const (
	memoryGatePollInterval = 2 * time.Second
	memoryGateLogInterval  = 10 * time.Second
	memoryGateDefaultWait  = 300
)

// waitAvailableMemory enforces the model's minAvailableMemory start gate:
// before the start command runs, it blocks until the OS reports at least the
// configured amount of available memory. This protects against starting a
// model load on top of memory a predecessor has not actually released yet
// (e.g. a vLLM engine stuck in a level-1 sleep holding its weights in host
// RAM), which would otherwise push the whole system into a global OOM.
//
// The gate is a no-op when minAvailableMemory is not set. The wait budget is
// waitAvailableMemory seconds, falling back to healthCheckTimeout, then to
// 300s; on expiry the start fails with an explicit error instead of risking
// an OOM kill of unrelated processes.
func (p *ProcessCommand) waitAvailableMemory(ctx context.Context) error {
	threshold := p.config.MinAvailableMemoryBytes()
	if threshold == 0 {
		return nil
	}

	waitSecs := p.config.WaitAvailableMemory
	if waitSecs <= 0 {
		waitSecs = p.config.HealthCheckTimeout
	}
	if waitSecs <= 0 {
		waitSecs = memoryGateDefaultWait
	}

	deadline := time.Now().Add(time.Duration(waitSecs) * time.Second)
	var lastLog time.Time

	for {
		vm, err := mem.VirtualMemoryWithContext(ctx)
		if err != nil {
			return fmt.Errorf("minAvailableMemory: reading system memory: %w", err)
		}
		if vm.Available >= threshold {
			return nil
		}

		if lastLog.IsZero() || time.Since(lastLog) >= memoryGateLogInterval {
			lastLog = time.Now()
			p.proxyLogger.Warnf("<%s> deferring start: %.1f GiB available, minAvailableMemory is %.1f GiB (will wait up to %d s)",
				p.id, float64(vm.Available)/(1<<30), float64(threshold)/(1<<30), waitSecs)
		}

		if time.Now().After(deadline) {
			return fmt.Errorf("minAvailableMemory: only %.1f GiB available of the required %.1f GiB after waiting %d s; refusing to start to avoid exhausting system memory",
				float64(vm.Available)/(1<<30), float64(threshold)/(1<<30), waitSecs)
		}

		select {
		case <-ctx.Done():
			return ErrStartAborted
		case <-time.After(memoryGatePollInterval):
		}
	}
}
