"""Capture an actual production serial run with the R1 MQTT service active."""
import argparse
import json
from pathlib import Path
import time

import serial
from server.app.r1_mqtt import R1Service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM10')
    parser.add_argument('--seconds', type=float, default=30)
    args = parser.parse_args()
    evidence = Path('docs/evidence')
    events = evidence / 'phase9_r1_production_server_events.jsonl'
    if events.exists():
        events.unlink()
    with R1Service(event_path=events) as service:
        with serial.Serial(args.port, 115200, timeout=.2) as device:
            device.dtr = False
            device.rts = True
            time.sleep(.15)
            device.rts = False
            started = time.monotonic()
            chunks = []
            while time.monotonic() - started < args.seconds:
                chunks.append(device.read(4096))
        text = b''.join(chunks).decode('utf-8', errors='replace').replace('\r', '').replace('\x00', '')
        (evidence / 'phase9_r1_production_serial.log').write_text(text, encoding='utf-8')
        decisions = []
        for line in text.splitlines():
            if 'R1_DECISION ' in line:
                decisions.append(json.loads(line.split('R1_DECISION ', 1)[1]))
        counts = {action: sum(row['selected_action'] == action for row in decisions) for action in ('LOCAL', 'CLOUD')}
        report = {'scope': 'Short actual production IMU run; not a general performance benchmark',
                  'capture_seconds': args.seconds, 'decisions': len(decisions), 'actions': counts,
                  'successful_decisions': sum(row['success'] for row in decisions),
                  'second_inferences': sum(row['second_inference_count'] for row in decisions),
                  'controlled': False, 'server_errors': service.errors,
                  'queue_full_count': text.count('R1_QUEUE_FULL'),
                  'state_unavailable_count': text.count('R1_STATE_UNAVAILABLE'),
                  'sampling_lines': [line for line in text.splitlines() if line.startswith('W=')],
                  'policy_states': [row['policy_state'] for row in decisions],
                  'status': 'PASS' if decisions and all(row['success'] for row in decisions) and not service.errors else 'FAIL'}
        (evidence / 'phase9_r1_production_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: value for key, value in report.items() if key not in ('sampling_lines', 'policy_states')}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
