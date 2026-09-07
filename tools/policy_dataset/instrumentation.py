"""Opt-in timing adapters. State snapshots and post-action outcomes stay separate."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import time

from .schema import example_record, validate_record, _number


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def probability_metrics(probabilities):
    values = list(probabilities)
    if len(values) != 5:
        raise ValueError("Expected five class probabilities")
    for value in values:
        _number(value, "probability", 1)
        if value is None:
            raise ValueError("Null probability")
    if not math.isclose(sum(values), 1, abs_tol=1e-6):
        raise ValueError("Probabilities must sum to one")
    ordered = sorted(values, reverse=True)
    return {"confidence": ordered[0], "entropy": -sum(p * math.log(p) for p in values if p > 0),
            "margin": ordered[0] - ordered[1]}, values.index(max(values))


@dataclass
class PolicySample:
    record: dict

    @classmethod
    def create(cls, *, sample_id, device_id, window_id, run_id, session_id, scope="host"):
        record = example_record()
        record["metadata"].update(sample_id=sample_id, device_id=device_id, timestamp=utc_now(),
                                  run_id=run_id, session_id=session_id, record_kind="observation",
                                  measurement_scope=scope)
        record["state"]["application"]["window_id"] = window_id
        record["provenance"]["notes"] = "Unobserved state remains null."
        return cls(record)

    def snapshot_state(self, state, source):
        """Call before selecting an action. Do not use post-action metrics as policy inputs."""
        candidate = deepcopy(self.record)
        candidate["state"] = deepcopy(state)
        candidate["provenance"]["state_source"] = source
        validate_record(candidate)
        self.record = candidate

    @contextmanager
    def measure(self, field, clock=time.perf_counter_ns):
        if field not in self.record["measurements"] or not field.endswith("latency_ms"):
            raise ValueError("Unknown duration field")
        started = clock()
        try:
            yield
        except Exception as exc:
            self.record["outcome"].update(status="failed", error=type(exc).__name__)
            raise
        finally:
            self.record["measurements"][field] = (clock() - started) / 1_000_000
            self.record["provenance"]["clock_domains"][field] = "host monotonic"

    def local(self, preprocess, infer, value):
        """Instrument supplied host callbacks; this does not execute ESP32 firmware."""
        self.record["action"] = 0
        with self.measure("total_latency_ms"):
            with self.measure("preprocessing_latency_ms"):
                prepared = preprocess(value)
            with self.measure("local_inference_latency_ms"):
                probabilities = infer(prepared)
            metrics, predicted = probability_metrics(probabilities)
            self.record["outcome"].update(status="success", predicted_class=predicted,
                                          confidence=metrics["confidence"])
        # These are outcomes, not a pre-decision state snapshot.
        self.record["measurements"].update(request_bytes=0, response_bytes=0,
                                            local_confidence=metrics["confidence"],
                                            local_entropy=metrics["entropy"], local_margin=metrics["margin"])
        return probabilities

    def prefix(self, infer, value, action):
        if action not in (1, 2) or isinstance(action, bool):
            raise ValueError("Only policy actions 1 and 2 have mapped split prefixes")
        self.record["action"] = action
        with self.measure("prefix_latency_ms"):
            embedding = list(infer(value))
        if len(embedding) != {1: 64, 2: 48}[action]:
            raise ValueError("Wrong embedding dimension")
        for value in embedding:
            _number(value, "embedding")
            if value is None:
                raise ValueError("Null embedding")
        self.record["measurements"]["embedding_size_bytes"] = 4 * len(embedding)
        return embedding

    def cloud(self, infer, publish, value):
        """Opt-in server adapter: preserve the supplied inference and publish functions."""
        measures = self.record["measurements"]
        measures["request_receive_time"] = utc_now()
        self.record["provenance"]["clock_domains"]["request_receive_time"] = "server UTC"
        with self.measure("cloud_inference_latency_ms"):
            result = infer(value)
        self.record["provenance"]["clock_domains"]["cloud_inference_latency_ms"] = "server monotonic"
        publish(result)
        measures["response_publish_time"] = utc_now()
        self.record["provenance"]["clock_domains"]["response_publish_time"] = "server UTC; publish returned, not delivery ACK"
        return result


class RoundTrip:
    """One correlated request; duration uses only the client's monotonic clock."""

    def __init__(self, sample, request_id, payload, *, clock=time.perf_counter_ns):
        if not isinstance(payload, bytes) or not isinstance(request_id, str) or not request_id:
            raise ValueError("Request needs a nonempty ID and exact serialized bytes")
        self.sample, self.request_id, self.clock = sample, request_id, clock
        self.started = clock()
        self.finished = False
        sample.record["measurements"].update(mqtt_send_timestamp=utc_now(), request_bytes=len(payload))
        sample.record["provenance"]["clock_domains"]["round_trip_latency_ms"] = "client monotonic"
        sample.record["provenance"]["clock_domains"]["mqtt_send_timestamp"] = "client UTC, before publish"

    def finish(self, response, payload):
        if not isinstance(payload, bytes):
            raise ValueError("Response payload must be serialized bytes")
        if self.finished:
            raise ValueError("Request already finished")
        if response.get("request_id") != self.request_id:
            raise ValueError("Response request_id mismatch")
        action = self.sample.record["action"]
        if action not in (1, 2) or response.get("split") != action:
            raise ValueError("Policy action is not this MQTT split; ALL_CLOUD is not Split3")
        from ml.dataset.loader import CLASS_TO_ID
        predicted = CLASS_TO_ID.get(response.get("predicted_class"))
        if predicted is None:
            raise ValueError("Invalid returned class")
        confidence = response.get("confidence")
        _number(confidence, "confidence", 1)
        latency = response.get("server_latency_ms")
        _number(latency, "server_latency_ms")
        if confidence is None or latency is None:
            raise ValueError("Missing response metrics")
        if response.get("model_version") != f"gesture-cloud-tail-split{action}-v1.0.0":
            raise ValueError("Wrong cloud model version")
        self.sample.record["measurements"].update(
            server_response_timestamp=utc_now(), round_trip_latency_ms=(self.clock() - self.started) / 1e6,
            response_bytes=len(payload), cloud_inference_latency_ms=latency)
        self.sample.record["provenance"]["clock_domains"]["server_response_timestamp"] = "client UTC on response receipt"
        self.sample.record["outcome"].update(status="success", predicted_class=predicted, confidence=confidence)
        self.finished = True

    def timeout(self):
        if self.finished:
            raise ValueError("Request already finished")
        self.sample.record["measurements"]["round_trip_latency_ms"] = (self.clock() - self.started) / 1e6
        self.sample.record["outcome"].update(status="timeout", error="response timeout")
        self.finished = True
