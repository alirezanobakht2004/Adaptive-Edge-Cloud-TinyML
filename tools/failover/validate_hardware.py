"""Controlled failover on ESP32, with actual inference-service stop and recovery."""
import json
from pathlib import Path
import subprocess
import time

import paho.mqtt.client as mqtt
from server.app.r1_mqtt import R1Service


def main():
    root=Path(__file__).resolve().parents[2]
    evidence=root/'docs/evidence/phase10'
    evidence.mkdir(parents=True,exist_ok=True)
    observed=[]
    observer=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,client_id='m10-independent-observer')
    observer.on_connect=lambda client,*args:client.subscribe('gesture/esp32-r1/inference/#')
    observer.on_message=lambda client,userdata,message:observed.append({'topic':message.topic,'payload':json.loads(message.payload),'at_ns':time.monotonic_ns()})
    observer.connect('127.0.0.1',1883);observer.loop_start()
    service=R1Service()
    service.__enter__();running=True;transitions=[];lines=[]
    command=[str(Path.home()/'.platformio/penv/Scripts/pio.exe'),'test','-e','esp32-s3-n8r2',
             '-f','test_phase10_failover','--upload-port','COM10','--test-port','COM10','-v']
    try:
        process=subprocess.Popen(command,cwd=root/'firmware',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            print(line,end='',flush=True);lines.append(line.rstrip())
            if line.strip()=='M10_SERVER_STOP_REQUEST':
                service.__exit__(None,None,None);running=False
                transitions.append({'server':'STOPPED','at_ns':time.monotonic_ns()})
            elif line.strip()=='M10_SERVER_START_REQUEST':
                service.ready.clear();service.__enter__();running=True
                transitions.append({'server':'RESTORED','at_ns':time.monotonic_ns()})
        code=process.wait()
    finally:
        if running:service.__exit__(None,None,None)
        observer.disconnect();observer.loop_stop()
    (evidence/'isolated_hardware.log').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    records=[json.loads(line.split('R1_DECISION ',1)[1]) for line in lines if line.startswith('R1_DECISION ')]
    by_id={row['request_id']:row for row in records}
    requests=[e for e in observed if e['topic'].endswith('/request')]
    responses=[e for e in observed if e['topic'].endswith('/response')]
    timeout=[e for e in requests if e['payload']['request_id']=='m10-server-timeout']
    stopped=next((x['at_ns'] for x in transitions if x['server']=='STOPPED'),None)
    restored=next((x['at_ns'] for x in transitions if x['server']=='RESTORED'),None)
    checks={'platformio_pass':code==0,'seven_windows':len(records)==7,
            'one_local_inference_each':bool(records) and all(r['local_inference_count']==1 and r['second_inference_count']==0 for r in records),
            'no_dropped_decision':len(records)==7 and all(r['success'] for r in records),
            'wifi_reason':by_id.get('m10-wifi-unavailable',{}).get('failover_reason')=='WIFI_UNAVAILABLE',
            'mqtt_reason':by_id.get('m10-mqtt-unavailable',{}).get('failover_reason')=='MQTT_UNAVAILABLE',
            'timeout_reason':by_id.get('m10-server-timeout',{}).get('failover_reason')=='CLOUD_RESPONSE_TIMEOUT',
            'broker_received_during_service_stop':bool(timeout) and stopped is not None and restored is not None and stopped<timeout[0]['at_ns']<restored,
            'no_timeout_response':not any(e['payload']['request_id']=='m10-server-timeout' for e in responses),
            'ten_feature_payloads':bool(requests) and all(len(e['payload']['features'])==10 for e in requests),
            'next_window_works':by_id.get('m10-next-local',{}).get('success') is True,
            'cloud_recovery':by_id.get('m10-cloud-recovered',{}).get('effective_action')=='CLOUD'}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','controlled':True,'checks':checks,
            'records':records,'service_transitions':transitions,'observed_mqtt':observed,
            'timeout_ms':3000,'service_errors':service.errors,'command':command}
    (evidence/'isolated_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2))
    return 0 if all(checks.values()) else 1


if __name__=='__main__':raise SystemExit(main())
