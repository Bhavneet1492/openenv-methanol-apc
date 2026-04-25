"""List Azure subscriptions and create Digital Twins resources."""
from azure.identity import InteractiveBrowserCredential
import requests
import json

cred = InteractiveBrowserCredential()
token = cred.get_token("https://management.azure.com/.default")
headers = {"Authorization": f"Bearer {token.token}", "Content-Type": "application/json"}

# 1. List subscriptions
r = requests.get("https://management.azure.com/subscriptions?api-version=2022-12-01", headers=headers)
subs = r.json().get("value", [])
for s in subs:
    print(f"Sub: {s['displayName']} | ID: {s['subscriptionId']} | State: {s['state']}")

if not subs:
    print("No subscriptions found!")
    exit(1)

# Use first active subscription
sub_id = subs[0]["subscriptionId"]
print(f"\nUsing subscription: {sub_id}")

# 2. Create resource group
RG_NAME = "methanol-apc-rg"
LOCATION = "eastus"
rg_url = f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{RG_NAME}?api-version=2024-07-01"
rg_body = {"location": LOCATION}
r = requests.put(rg_url, headers=headers, json=rg_body)
print(f"Resource Group: {r.status_code} {r.json().get('properties', {}).get('provisioningState', r.text[:100])}")

# 3. Create Azure Digital Twins instance
ADT_NAME = "methanol-apc-adt"
adt_url = f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{RG_NAME}/providers/Microsoft.DigitalTwins/digitalTwinsInstances/{ADT_NAME}?api-version=2023-01-31"
adt_body = {"location": LOCATION}
r = requests.put(adt_url, headers=headers, json=adt_body)
print(f"Digital Twins: {r.status_code}")
adt_data = r.json()
if "hostName" in adt_data:
    print(f"  URL: https://{adt_data['hostName']}")
else:
    print(f"  Response: {json.dumps(adt_data, indent=2)[:500]}")

# 4. Create IoT Hub (free tier)
IOT_NAME = "methanol-apc-iothub"
iot_url = f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{RG_NAME}/providers/Microsoft.Devices/IotHubs/{IOT_NAME}?api-version=2023-06-30"
iot_body = {
    "location": LOCATION,
    "sku": {"name": "F1", "capacity": 1},
    "properties": {}
}
r = requests.put(iot_url, headers=headers, json=iot_body)
print(f"IoT Hub: {r.status_code}")
if r.status_code in (200, 201):
    print(f"  Name: {IOT_NAME}")
else:
    print(f"  Response: {r.text[:300]}")

# 5. Create Storage Account
STORAGE_NAME = "methanolacpstorage"  # must be globally unique, lowercase
storage_url = f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{RG_NAME}/providers/Microsoft.Storage/storageAccounts/{STORAGE_NAME}?api-version=2023-05-01"
storage_body = {
    "location": LOCATION,
    "sku": {"name": "Standard_LRS"},
    "kind": "StorageV2",
    "properties": {"allowBlobPublicAccess": True}
}
r = requests.put(storage_url, headers=headers, json=storage_body)
print(f"Storage: {r.status_code}")
if r.status_code in (200, 202):
    print(f"  Name: {STORAGE_NAME}")
else:
    print(f"  Response: {r.text[:300]}")

print("\nDone! Resources creating in Azure.")
