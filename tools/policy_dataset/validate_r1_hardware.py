"""Run isolated production-core E2E or retained split suites against the R1 server."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from server.app.r1_mqtt import R1Service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='test_phase9_r1_e2e')
    parser.add_argument('--port', default='COM10')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    evidence = root / 'docs/evidence'
    stem = args.suite.replace('*', 'all').replace('?', 'any')
    events = evidence / f'{stem}_server_events.jsonl'
    if events.exists():
        events.unlink()
    command = [str(Path.home() / '.platformio/penv/Scripts/pio.exe'), 'test', '-e',
               'esp32-s3-n8r2', '-f', args.suite, '--upload-port', args.port, '--test-port', args.port, '-v']
    with R1Service(event_path=events) as service:
        process = subprocess.Popen(command, cwd=root / 'firmware', stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            lines.append(line.rstrip())
        code = process.wait()
        (evidence / f'{stem}_hardware.log').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        cloud = [e for e in service.events if e['kind'] == 'CLOUD']
        checks = {'platformio_pass': code == 0, 'server_errors_empty': not service.errors}
        if args.suite == 'test_phase9_r1_e2e':
            checks.update(one_cloud_request=len(cloud) == 1,
                          correlated_cloud=bool(cloud) and cloud[0]['request']['request_id'] ==
                          cloud[0]['response']['request_id'] == 'r1-controlled-cloud',
                          exactly_ten_features=bool(cloud) and len(cloud[0]['request']['features']) == 10,
                          no_local_request=all(e['request']['request_id'] != 'r1-controlled-local' for e in cloud),
                          version_logged=bool(cloud) and cloud[0]['response']['policy_version'] == 'meta-policy-v1.0.0')
        report = {'suite': args.suite, 'command': command, 'checks': checks, 'errors': service.errors,
                  'controlled': True, 'events_file': events.name,
                  'status': 'PASS' if all(checks.values()) else 'FAIL'}
        (evidence / f'{stem}_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
