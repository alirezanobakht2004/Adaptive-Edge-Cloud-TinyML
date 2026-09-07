"""Run PlatformIO with real MQTT inference and explicit legacy transport fixtures.

No filters means the full firmware suite. Use --filter 'test_phase7_*' to repeat
only the milestone checks. No production server or firmware behavior is changed.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import paho.mqtt.client as mqtt

from server.app.inference import SplitCloudInference
from server.app.mqtt import ServerState, on_message


def main() -> None:
    # ESP32 boot bytes may not be UTF-8; keep Windows console output loss-tolerant.
    sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--filter", action="append", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    pio = shutil.which("pio") or str(Path.home() / ".platformio/penv/Scripts/platformio.exe")
    current_suite = ""
    subscribed = threading.Event()
    state = ServerState(SplitCloudInference())
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="phase7-validation-server")

    def connected(client, userdata, flags, reason, properties):
        if not reason.is_failure:
            client.subscribe("gesture/+/inference/request")

    def handle(client, userdata, message):
        # This legacy test explicitly requires an unavailable server.
        if current_suite == "test_phase6_server_timeout":
            print("HARNESS: server-unavailable fixture for timeout test", flush=True)
            return
        on_message(client, state, message)

    client.on_connect = connected
    client.on_subscribe = lambda *unused: subscribed.set()
    client.on_message = handle
    client.connect("127.0.0.1", 1883)
    client.loop_start()
    try:
        if not subscribed.wait(10):
            raise RuntimeError("MQTT broker subscription failed")
        command = [pio, "test", "-d", "firmware", "-e", "esp32-s3-n8r2",
                   "--upload-port", args.port, "--test-port", args.port, "-v"]
        for pattern in args.filter:
            command.extend(["-f", pattern])
        print("RUN:", subprocess.list2cmdline(command), flush=True)
        with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1,
                              env={**os.environ, "PYTHONUNBUFFERED": "1"},
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                if line.startswith("Processing "):
                    current_suite = line.split()[1]
                if current_suite == "test_phase6_mqtt_receive" and "MQTT_SUBSCRIBE=PASS" in line:
                    # Existing receive-only transport test needs its documented test payload.
                    info = client.publish("gesture/esp32-01/inference/response",
                                          '{"transport_test":"phase6_mqtt_receive_pass"}')
                    info.wait_for_publish(timeout=2)
            result = process.wait()
        raise SystemExit(result)
    finally:
        client.disconnect()
        client.loop_stop()


if __name__ == "__main__":
    main()
