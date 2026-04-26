"""Provision Azure Digital Twins: assign RBAC, upload DTDL models, create twins + relationships."""
import json
import os
import time
import sys
from pathlib import Path

from azure.identity import InteractiveBrowserCredential
import requests

# ── Config ──
SUB_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
TENANT_ID = os.environ["AZURE_TENANT_ID"]
RG_NAME = os.environ["AZURE_RESOURCE_GROUP"]
ADT_NAME = os.environ["AZURE_ADT_NAME"]
ADT_HOST = os.environ["AZURE_DIGITAL_TWINS_URL"].rstrip("/")

DTDL_PATH = Path(__file__).parent.parent / "methanol_apc_env" / "dtdl" / "methanol_plant_models.json"

# Azure Digital Twins Data Owner role ID (varies by subscription)
ADT_DATA_OWNER_ROLE = "bcd981a7-7f74-457b-83e1-cceb9e632ffe"

print(f"ADT Host: {ADT_HOST}")
print(f"DTDL Path: {DTDL_PATH}")

# ── Authenticate ──
cred = InteractiveBrowserCredential(tenant_id=TENANT_ID)

# Get management token (for RBAC)
mgmt_token = cred.get_token("https://management.azure.com/.default")
mgmt_headers = {"Authorization": f"Bearer {mgmt_token.token}", "Content-Type": "application/json"}

# Get ADT data plane token
adt_token = cred.get_token("https://digitaltwins.azure.net/.default")
adt_headers = {"Authorization": f"Bearer {adt_token.token}", "Content-Type": "application/json"}

print("Authenticated!\n")

# ── Step 0: Get current user's object ID ──
print("0. Getting current user info...")
graph_token = cred.get_token("https://graph.microsoft.com/.default")
r = requests.get(
    "https://graph.microsoft.com/v1.0/me",
    headers={"Authorization": f"Bearer {graph_token.token}"}
)
if r.status_code == 200:
    user_id = r.json()["id"]
    user_name = r.json().get("displayName", "unknown")
    print(f"   User: {user_name} ({user_id})")
else:
    print(f"   Failed to get user info: {r.status_code} {r.text[:200]}")
    print("   Skipping RBAC assignment - you may need to assign manually")
    user_id = None

# ── Step 1: Assign ADT Data Owner role ──
if user_id:
    print("\n1. Assigning Azure Digital Twins Data Owner role...")
    import uuid
    role_assignment_id = str(uuid.uuid4())
    scope = f"/subscriptions/{SUB_ID}/resourceGroups/{RG_NAME}/providers/Microsoft.DigitalTwins/digitalTwinsInstances/{ADT_NAME}"
    r = requests.put(
        f"https://management.azure.com{scope}/providers/Microsoft.Authorization/roleAssignments/{role_assignment_id}?api-version=2022-04-01",
        headers=mgmt_headers,
        json={
            "properties": {
                "roleDefinitionId": f"/subscriptions/{SUB_ID}/providers/Microsoft.Authorization/roleDefinitions/{ADT_DATA_OWNER_ROLE}",
                "principalId": user_id
            }
        }
    )
    if r.status_code in (200, 201):
        print("   ✅ Role assigned!")
    elif r.status_code == 409:
        print("   ✅ Role already assigned!")
    else:
        print(f"   ⚠️  Status {r.status_code}: {r.text[:300]}")
        print("   Continuing anyway - role may already exist...")

    # Wait for role propagation
    print("   Waiting 60s for RBAC propagation (Azure can take up to 5 min)...")
    time.sleep(60)

    # Re-acquire ADT token after RBAC
    adt_token = cred.get_token("https://digitaltwins.azure.net/.default")
    adt_headers = {"Authorization": f"Bearer {adt_token.token}", "Content-Type": "application/json"}

# ── Step 2: Upload DTDL Models ──
print("\n2. Uploading DTDL models...")
with open(DTDL_PATH) as f:
    models = json.load(f)

print(f"   Found {len(models)} models")

# Upload all models at once (batch upload handles dependencies)
r = requests.post(
    f"{ADT_HOST}/models?api-version=2023-10-31",
    headers=adt_headers,
    json=models
)
if r.status_code in (200, 201):
    print(f"   ✅ All {len(models)} models uploaded!")
elif r.status_code == 409:
    print("   ✅ Models already exist!")
else:
    print(f"   Status {r.status_code}: {r.text[:500]}")
    if r.status_code == 403:
        print("\n   RBAC not propagated yet. Wait 30s and re-run, or assign role manually:")
        print(f"   az role assignment create --assignee {user_id} --role 'Azure Digital Twins Data Owner' --scope {scope}")

# ── Step 3: Create Twin Instances ──
print("\n3. Creating twin instances...")

twins = [
    {
        "id": "methanol-plant-001",
        "model": "dtmi:methanol:plant;1",
        "properties": {
            "plantName": "Methanol Synthesis Plant - OpenEnv Demo",
            "plantStatus": "running",
            "totalMethanolProduced": 0.0,
            "cumulativeProfit": 0.0,
            "stepNumber": 0
        }
    },
    {
        "id": "syngas-feed-001",
        "model": "dtmi:methanol:syngas_feed;1",
        "properties": {
            "reformerOutletTemp": 850.0,
            "reformerPressure": 25.0,
            "feedRateH2": 5.0,
            "feedRateCO": 2.5,
            "h2CoRatio": 2.0,
            "steamToCarbon": 3.0,
            "fuelGasFlow": 5.0,
            "controllingAgent": "ReformerAgent"
        }
    },
    {
        "id": "compressor-001",
        "model": "dtmi:methanol:compressor;1",
        "properties": {
            "power": 65.0,
            "inletPressure": 25.0,
            "outletPressure": 80.0,
            "compressionRatio": 3.2,
            "controllingAgent": "SynthesisAgent"
        }
    },
    {
        "id": "reactor-001",
        "model": "dtmi:methanol:reactor;1",
        "properties": {
            "temperature": 250.0,
            "pressure": 80.0,
            "catalystHealth": 1.0,
            "reactionRate": 0.15,
            "selectivity": 0.95,
            "bed1Temp": 245.0,
            "bed2Temp": 250.0,
            "bed3Temp": 255.0,
            "bed4Temp": 260.0,
            "singlePassConversion": 0.05,
            "carbonEfficiency": 0.85,
            "currentControlAction": "{}",
            "controllingAgent": "SynthesisAgent",
            "agentConfidence": 0.0,
            "emergencyShutdown": False
        }
    },
    {
        "id": "quench-zone-001",
        "model": "dtmi:methanol:quench_zone;1",
        "properties": {
            "inletTemp": 260.0,
            "outletTemp": 240.0,
            "quenchGasFlow": 1.0,
            "bedIndex": 1
        }
    },
    {
        "id": "quench-zone-002",
        "model": "dtmi:methanol:quench_zone;1",
        "properties": {
            "inletTemp": 265.0,
            "outletTemp": 242.0,
            "quenchGasFlow": 1.2,
            "bedIndex": 2
        }
    },
    {
        "id": "quench-zone-003",
        "model": "dtmi:methanol:quench_zone;1",
        "properties": {
            "inletTemp": 270.0,
            "outletTemp": 245.0,
            "quenchGasFlow": 1.5,
            "bedIndex": 3
        }
    },
    {
        "id": "separator-001",
        "model": "dtmi:methanol:separator;1",
        "properties": {
            "temperature": 40.0,
            "pressure": 30.0,
            "liquidLevel": 50.0,
            "condensationEfficiency": 0.96
        }
    },
    {
        "id": "distillation-001",
        "model": "dtmi:methanol:distillation;1",
        "properties": {
            "refluxRatio": 3.0,
            "reboilerDuty": 50.0,
            "overheadTemp": 64.7,
            "bottomsTemp": 100.0,
            "columnPressure": 1.0,
            "productPurity": 0.9985,
            "productFlowRate": 120.0,
            "controllingAgent": "PurificationAgent"
        }
    },
    {
        "id": "cooling-tower-001",
        "model": "dtmi:methanol:cooling_tower;1",
        "properties": {
            "coolingWaterFlow": 40.0,
            "supplyTemp": 25.0,
            "returnTemp": 45.0,
            "fanSpeed": 60.0,
            "heatDuty": 500.0
        }
    },
    {
        "id": "recycle-loop-001",
        "model": "dtmi:methanol:recycle_loop;1",
        "properties": {
            "recycleRatio": 3.5,
            "purgeRate": 0.5,
            "purgeValvePosition": 5.0,
            "inertFraction": 0.1,
            "flareValve": 0.0
        }
    },
    {
        "id": "agent-reformer",
        "model": "dtmi:methanol:agent_controller;1",
        "properties": {
            "agentId": "reformer-001",
            "agentRole": "reformer",
            "currentAction": "{}",
            "confidence": 0.0,
            "stepReward": 0.0,
            "cumulativeReward": 0.0,
            "modelName": "Qwen2.5-7B-Instruct",
            "isActive": True
        }
    },
    {
        "id": "agent-synthesis",
        "model": "dtmi:methanol:agent_controller;1",
        "properties": {
            "agentId": "synthesis-001",
            "agentRole": "synthesis",
            "currentAction": "{}",
            "confidence": 0.0,
            "stepReward": 0.0,
            "cumulativeReward": 0.0,
            "modelName": "Qwen2.5-7B-Instruct",
            "isActive": True
        }
    },
    {
        "id": "agent-purification",
        "model": "dtmi:methanol:agent_controller;1",
        "properties": {
            "agentId": "purification-001",
            "agentRole": "purification",
            "currentAction": "{}",
            "confidence": 0.0,
            "stepReward": 0.0,
            "cumulativeReward": 0.0,
            "modelName": "Qwen2.5-7B-Instruct",
            "isActive": True
        }
    },
    {
        "id": "agent-supervisory",
        "model": "dtmi:methanol:agent_controller;1",
        "properties": {
            "agentId": "supervisory-001",
            "agentRole": "supervisory",
            "currentAction": "{}",
            "confidence": 0.0,
            "stepReward": 0.0,
            "cumulativeReward": 0.0,
            "modelName": "Qwen2.5-7B-Instruct",
            "isActive": True
        }
    }
]

created = 0
for twin in twins:
    twin_id = twin["id"]
    body = {
        "$metadata": {"$model": twin["model"]},
        **twin["properties"]
    }
    r = requests.put(
        f"{ADT_HOST}/digitaltwins/{twin_id}?api-version=2023-10-31",
        headers=adt_headers,
        json=body
    )
    status = "✅" if r.status_code in (200, 201) else ("⚠️ exists" if r.status_code == 412 else f"❌ {r.status_code}")
    print(f"   {status} {twin_id} ({twin['model'].split(':')[2].split(';')[0]})")
    if r.status_code in (200, 201):
        created += 1
    elif r.status_code not in (412,):
        print(f"      Error: {r.text[:200]}")

print(f"   Created {created}/{len(twins)} twins")

# ── Step 4: Create Relationships ──
print("\n4. Creating relationships...")

relationships = [
    # Plant contains everything
    ("methanol-plant-001", "contains-syngas", "contains", "syngas-feed-001"),
    ("methanol-plant-001", "contains-compressor", "contains", "compressor-001"),
    ("methanol-plant-001", "contains-reactor", "contains", "reactor-001"),
    ("methanol-plant-001", "contains-separator", "contains", "separator-001"),
    ("methanol-plant-001", "contains-distillation", "contains", "distillation-001"),
    ("methanol-plant-001", "contains-cooling", "contains", "cooling-tower-001"),
    ("methanol-plant-001", "contains-recycle", "contains", "recycle-loop-001"),
    # Process flow
    ("syngas-feed-001", "feeds-compressor", "feedsTo", "compressor-001"),
    ("compressor-001", "feeds-reactor", "feedsTo", "reactor-001"),
    ("reactor-001", "feeds-separator", "feedsTo", "separator-001"),
    ("reactor-001", "has-quench-1", "hasQuenchZone", "quench-zone-001"),
    ("reactor-001", "has-quench-2", "hasQuenchZone", "quench-zone-002"),
    ("reactor-001", "has-quench-3", "hasQuenchZone", "quench-zone-003"),
    ("separator-001", "liquid-to-distill", "liquidTo", "distillation-001"),
    ("separator-001", "gas-to-recycle", "gasTo", "recycle-loop-001"),
    ("recycle-loop-001", "recycle-to-compressor", "recyclesTo", "compressor-001"),
    # Cooling
    ("cooling-tower-001", "cools-reactor", "cools", "reactor-001"),
    ("reactor-001", "cooled-by-tower", "cooledBy", "cooling-tower-001"),
    # Agent control
    ("agent-reformer", "controls-syngas", "controls", "syngas-feed-001"),
    ("agent-synthesis", "controls-reactor", "controls", "reactor-001"),
    ("agent-synthesis", "controls-compressor", "controls", "compressor-001"),
    ("agent-synthesis", "controls-cooling", "controls", "cooling-tower-001"),
    ("agent-synthesis", "controls-recycle", "controls", "recycle-loop-001"),
    ("agent-purification", "controls-distill", "controls", "distillation-001"),
    ("agent-supervisory", "controls-plant", "controls", "methanol-plant-001"),
]

rel_created = 0
for source_id, rel_id, rel_name, target_id in relationships:
    body = {
        "$relationshipId": rel_id,
        "$sourceId": source_id,
        "$relationshipName": rel_name,
        "$targetId": target_id
    }
    r = requests.put(
        f"{ADT_HOST}/digitaltwins/{source_id}/relationships/{rel_id}?api-version=2023-10-31",
        headers=adt_headers,
        json=body
    )
    status = "✅" if r.status_code in (200, 201) else ("⚠️ exists" if r.status_code == 409 else f"❌ {r.status_code}")
    print(f"   {status} {source_id} --[{rel_name}]--> {target_id}")
    if r.status_code in (200, 201):
        rel_created += 1
    elif r.status_code not in (409,):
        print(f"      Error: {r.text[:200]}")

print(f"   Created {rel_created}/{len(relationships)} relationships")

# ── Step 5: Verify ──
print("\n5. Verification...")

# Query all twins
r = requests.post(
    f"{ADT_HOST}/query?api-version=2023-10-31",
    headers=adt_headers,
    json={"query": "SELECT * FROM digitaltwins"}
)
if r.status_code == 200:
    result = r.json()
    twin_count = len(result.get("value", []))
    print(f"   ✅ Total twins in ADT: {twin_count}")
    for t in result.get("value", []):
        model = t.get("$metadata", {}).get("$model", "?")
        print(f"      - {t.get('$dtId', '?')} ({model.split(':')[2].split(';')[0] if ':' in model else model})")
else:
    print(f"   ❌ Query failed: {r.status_code} {r.text[:300]}")

print(f"\n{'='*60}")
print(f"ADT URL: {ADT_HOST}")
print(f"Set env var:")
print(f'  $env:AZURE_DIGITAL_TWINS_URL = "{ADT_HOST}"')
print(f"\nExplorer: https://explorer.digitaltwins.azure.net/?adt={ADT_HOST}")
print(f"{'='*60}")
