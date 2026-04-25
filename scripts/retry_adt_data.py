"""Retry just the data operations (models + twins + relationships) — RBAC already assigned."""
import json
from pathlib import Path
from azure.identity import InteractiveBrowserCredential
import requests

TENANT_ID = "4803f9ef-12cd-46f4-ad6c-c5245df0714f"
ADT_HOST = "https://methanol-apc-adt.api.eus.digitaltwins.azure.net"
DTDL_PATH = Path(__file__).parent.parent / "methanol_apc_env" / "dtdl" / "methanol_plant_models.json"

cred = InteractiveBrowserCredential(tenant_id=TENANT_ID)
token = cred.get_token("https://digitaltwins.azure.net/.default")
H = {"Authorization": f"Bearer {token.token}", "Content-Type": "application/json"}
print("Authenticated!")

# 1. Quick access test
print("\n1. Testing ADT access...")
r = requests.get(f"{ADT_HOST}/models?api-version=2023-10-31", headers=H)
print(f"   GET /models: {r.status_code}")
if r.status_code == 403:
    print("   RBAC still not propagated. Please wait a few more minutes and retry.")
    print("   Or go to Azure Portal → methanol-apc-adt → Access Control (IAM) → Add role assignment")
    print("   → Azure Digital Twins Data Owner → assign to your user")
    exit(1)

# 2. Upload DTDL models
print("\n2. Uploading DTDL models...")
with open(DTDL_PATH) as f:
    models = json.load(f)
print(f"   Found {len(models)} models")
r = requests.post(f"{ADT_HOST}/models?api-version=2023-10-31", headers=H, json=models)
if r.status_code in (200, 201):
    print(f"   ✅ All {len(models)} models uploaded!")
elif r.status_code == 409:
    print("   ✅ Models already exist!")
else:
    print(f"   Status {r.status_code}: {r.text[:500]}")

# 3. Create twins
print("\n3. Creating twins...")
twins = [
    ("methanol-plant-001", "dtmi:methanol:plant;1", {"plantName": "Methanol Synthesis Plant", "plantStatus": "running", "totalMethanolProduced": 0.0, "cumulativeProfit": 0.0, "stepNumber": 0}),
    ("syngas-feed-001", "dtmi:methanol:syngas_feed;1", {"reformerOutletTemp": 850.0, "reformerPressure": 25.0, "feedRateH2": 5.0, "feedRateCO": 2.5, "h2CoRatio": 2.0, "steamToCarbon": 3.0, "fuelGasFlow": 5.0, "controllingAgent": "ReformerAgent"}),
    ("compressor-001", "dtmi:methanol:compressor;1", {"power": 65.0, "inletPressure": 25.0, "outletPressure": 80.0, "compressionRatio": 3.2, "controllingAgent": "SynthesisAgent"}),
    ("reactor-001", "dtmi:methanol:reactor;1", {"temperature": 250.0, "pressure": 80.0, "catalystHealth": 1.0, "reactionRate": 0.15, "selectivity": 0.95, "bed1Temp": 245.0, "bed2Temp": 250.0, "bed3Temp": 255.0, "bed4Temp": 260.0, "singlePassConversion": 0.05, "carbonEfficiency": 0.85, "currentControlAction": "{}", "controllingAgent": "SynthesisAgent", "agentConfidence": 0.0, "emergencyShutdown": False}),
    ("quench-zone-001", "dtmi:methanol:quench_zone;1", {"inletTemp": 260.0, "outletTemp": 240.0, "quenchGasFlow": 1.0, "bedIndex": 1}),
    ("quench-zone-002", "dtmi:methanol:quench_zone;1", {"inletTemp": 265.0, "outletTemp": 242.0, "quenchGasFlow": 1.2, "bedIndex": 2}),
    ("quench-zone-003", "dtmi:methanol:quench_zone;1", {"inletTemp": 270.0, "outletTemp": 245.0, "quenchGasFlow": 1.5, "bedIndex": 3}),
    ("separator-001", "dtmi:methanol:separator;1", {"temperature": 40.0, "pressure": 30.0, "liquidLevel": 50.0, "condensationEfficiency": 0.96}),
    ("distillation-001", "dtmi:methanol:distillation;1", {"refluxRatio": 3.0, "reboilerDuty": 50.0, "overheadTemp": 64.7, "bottomsTemp": 100.0, "columnPressure": 1.0, "productPurity": 0.9985, "productFlowRate": 120.0, "controllingAgent": "PurificationAgent"}),
    ("cooling-tower-001", "dtmi:methanol:cooling_tower;1", {"coolingWaterFlow": 40.0, "supplyTemp": 25.0, "returnTemp": 45.0, "fanSpeed": 60.0, "heatDuty": 500.0}),
    ("recycle-loop-001", "dtmi:methanol:recycle_loop;1", {"recycleRatio": 3.5, "purgeRate": 0.5, "purgeValvePosition": 5.0, "inertFraction": 0.1, "flareValve": 0.0}),
    ("agent-reformer", "dtmi:methanol:agent_controller;1", {"agentId": "reformer-001", "agentRole": "reformer", "currentAction": "{}", "confidence": 0.0, "stepReward": 0.0, "cumulativeReward": 0.0, "modelName": "Qwen2.5-7B-Instruct", "isActive": True}),
    ("agent-synthesis", "dtmi:methanol:agent_controller;1", {"agentId": "synthesis-001", "agentRole": "synthesis", "currentAction": "{}", "confidence": 0.0, "stepReward": 0.0, "cumulativeReward": 0.0, "modelName": "Qwen2.5-7B-Instruct", "isActive": True}),
    ("agent-purification", "dtmi:methanol:agent_controller;1", {"agentId": "purification-001", "agentRole": "purification", "currentAction": "{}", "confidence": 0.0, "stepReward": 0.0, "cumulativeReward": 0.0, "modelName": "Qwen2.5-7B-Instruct", "isActive": True}),
    ("agent-supervisory", "dtmi:methanol:agent_controller;1", {"agentId": "supervisory-001", "agentRole": "supervisory", "currentAction": "{}", "confidence": 0.0, "stepReward": 0.0, "cumulativeReward": 0.0, "modelName": "Qwen2.5-7B-Instruct", "isActive": True}),
]
ok = 0
for tid, model, props in twins:
    body = {"$metadata": {"$model": model}, **props}
    r = requests.put(f"{ADT_HOST}/digitaltwins/{tid}?api-version=2023-10-31", headers=H, json=body)
    s = "✅" if r.status_code in (200, 201) else ("⚠️ exists" if r.status_code == 412 else f"❌ {r.status_code}")
    name = model.split(":")[2].split(";")[0]
    print(f"   {s} {tid} ({name})")
    if r.status_code in (200, 201): ok += 1
    elif r.status_code not in (412,): print(f"      {r.text[:200]}")
print(f"   Created {ok}/{len(twins)}")

# 4. Create relationships
print("\n4. Creating relationships...")
rels = [
    ("methanol-plant-001", "contains-syngas", "containsSyngasFeed", "syngas-feed-001"),
    ("methanol-plant-001", "contains-compressor", "containsCompressor", "compressor-001"),
    ("methanol-plant-001", "contains-reactor", "containsReactor", "reactor-001"),
    ("methanol-plant-001", "contains-separator", "containsSeparator", "separator-001"),
    ("methanol-plant-001", "contains-distillation", "containsDistillation", "distillation-001"),
    ("methanol-plant-001", "contains-cooling", "containsCoolingTower", "cooling-tower-001"),
    ("methanol-plant-001", "contains-recycle", "containsRecycleLoop", "recycle-loop-001"),
    ("syngas-feed-001", "feeds-compressor", "feedsTo", "compressor-001"),
    ("compressor-001", "feeds-reactor", "feedsTo", "reactor-001"),
    ("reactor-001", "feeds-separator", "feedsTo", "separator-001"),
    ("reactor-001", "has-quench-1", "hasQuenchZone", "quench-zone-001"),
    ("reactor-001", "has-quench-2", "hasQuenchZone", "quench-zone-002"),
    ("reactor-001", "has-quench-3", "hasQuenchZone", "quench-zone-003"),
    ("separator-001", "liquid-to-distill", "liquidTo", "distillation-001"),
    ("separator-001", "gas-to-recycle", "gasTo", "recycle-loop-001"),
    ("recycle-loop-001", "recycle-to-compressor", "recyclesTo", "compressor-001"),
    ("cooling-tower-001", "cools-reactor", "cools", "reactor-001"),
    ("reactor-001", "cooled-by-tower", "cooledBy", "cooling-tower-001"),
    ("agent-reformer", "controls-syngas", "controls", "syngas-feed-001"),
    ("agent-synthesis", "controls-reactor", "controls", "reactor-001"),
    ("agent-synthesis", "controls-compressor", "controls", "compressor-001"),
    ("agent-synthesis", "controls-cooling", "controls", "cooling-tower-001"),
    ("agent-synthesis", "controls-recycle", "controls", "recycle-loop-001"),
    ("agent-purification", "controls-distill", "controls", "distillation-001"),
    ("agent-supervisory", "controls-plant", "controls", "methanol-plant-001"),
]
rok = 0
for src, rid, rname, tgt in rels:
    body = {"$relationshipId": rid, "$sourceId": src, "$relationshipName": rname, "$targetId": tgt}
    r = requests.put(f"{ADT_HOST}/digitaltwins/{src}/relationships/{rid}?api-version=2023-10-31", headers=H, json=body)
    s = "✅" if r.status_code in (200, 201) else ("⚠️" if r.status_code == 409 else f"❌ {r.status_code}")
    print(f"   {s} {src} --[{rname}]--> {tgt}")
    if r.status_code in (200, 201): rok += 1
    elif r.status_code not in (409,): print(f"      {r.text[:150]}")
print(f"   Created {rok}/{len(rels)}")

# 5. Verify
print("\n5. Query all twins...")
r = requests.post(f"{ADT_HOST}/query?api-version=2023-10-31", headers=H, json={"query": "SELECT * FROM digitaltwins"})
if r.status_code == 200:
    twins_found = r.json().get("value", [])
    print(f"   ✅ {len(twins_found)} twins in ADT")
    for t in twins_found:
        m = t.get("$metadata", {}).get("$model", "?")
        print(f"      - {t.get('$dtId')} ({m.split(':')[2].split(';')[0] if ':' in m else m})")
else:
    print(f"   ❌ {r.status_code}: {r.text[:200]}")

print(f"\nADT URL: {ADT_HOST}")
print(f"Explorer: https://explorer.digitaltwins.azure.net/?adt={ADT_HOST}")
