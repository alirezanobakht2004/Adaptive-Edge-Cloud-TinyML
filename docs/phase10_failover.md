# Phase 10 / M10 - Failover

Status: CLOSED. This file is retained as the original Phase-10 implementation plan. Completion status and measured controlled-hardware evidence are canonicalized in `docs/phase10_m10_completion.md`. Phase 11 may proceed without reopening this plan.

## Baseline audit

Baseline commit: a7b5394, clean and synchronized with origin/main.
Python baseline: 309 passed, five TFLite deprecation warnings (88.40 seconds).
Firmware: 0.2.0-r1. Production MQTT service has no independent application release
number: implementation is pinned to a7b5394, protocol inference-r1-v1, full-cloud
model gesture-full-cloud-v1.0.0. The unrelated FastAPI app still reports 0.1.0/M7.
Meta policy remains meta-policy-v1.0.0; its six features and model bytes are frozen.

Current production exchange uses a literal 3000000us deadline for both RTT probes
and CLOUD responses. Wi-Fi connection waits up to 20 seconds; MQTT uses PubSubClient
defaults. The worker skips windows after connectivity/probe failure and returns a
failed result after cloud failure. Its two-entry queue can fill during blocking
exchange/reconnect waits. Logging has no requested/effective distinction or fallback
reason. No second inference is currently implemented, and none will be introduced.

## Planned transition and timing contract

Cached local result ready -> unchanged learned LOCAL/CLOUD request -> actual
connectivity gate -> correlated CLOUD success or cached LOCAL fallback. Requested
LOCAL needs no network. Pending CLOUD requests must be polled without blocking the
next window. The cache keeps class/UQ, window identity and a one-inference count.

config/r1_failover_v1.json preserves the existing 3000ms production deadline.
It is an engineering budget, not an optimized latency or measured failover value.
The existing RTT probe remains network-state acquisition, not a new health service.
Only a last successfully observed RTT may enter the learned policy; its age is logged.
If no RTT has ever been measured, safety returns the cache without inventing a policy
input/action. Such pre-policy fallback is identified separately in the log.

Isolated desktop state-transition tests precede production edits. Hardware E2E,
actual Wi-Fi loss, timeout/recovery and sampling evidence remain pending.
