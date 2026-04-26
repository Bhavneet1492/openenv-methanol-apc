"""Create Azure Digital Twins resources using known subscription."""
import os

from azure.identity import InteractiveBrowserCredential
import requests
import json

SUB_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
TENANT_ID = os.environ["AZURE_TENANT_ID"]
RG_NAME = os.environ["AZURE_RESOURCE_GROUP"]
LOCATION = os.environ["AZURE_LOCATION"]
ADT_NAME = os.environ["AZURE_ADT_NAME"]
IOT_NAME = os.environ["AZURE_IOT_HUB_NAME"]
STORAGE_NAME = os.environ["AZURE_STORAGE_ACCOUNT"]

print(f"Using subscription: {SUB_ID}")
print(f"Tenant: {TENANT_ID}")

cred = InteractiveBrowserCredential(tenant_id=TENANT_ID)
token = cred.get_token("https://management.azure.com/.default")
headers = {"Authorization": f"Bearer {token.token}", "Content-Type": "application/json"}
print("Authenticated!\n")

# 1. Create Resource Group
print("1. Creating Resource Group...")
r = requests.put(
    f"https://management.azure.com/subscriptions/{SUB_ID}/resourceGroups/{RG_NAME}?api-version=2024-07-01",
    headers=headers, json={"location": LOCATION}
)
print(f"   Status: {r.status_code} - {r.json().get('properties', {}).get('provisioningState', r.text[:200])}")

# 2. Create Azure Digital Twins
print("\n2. Creating Azure Digital Twins instance...")
r = requests.put(
    f"https://management.azure.com/subscriptions/{SUB_ID}/resourceGroups/{RG_NAME}/providers/Microsoft.DigitalTwins/digitalTwinsInstances/{ADT_NAME}?api-version=2023-01-31",
    headers=headers, json={"location": LOCATION}
)
print(f"   Status: {r.status_code}")
adt = r.json()
adt_host_name = adt.get("hostName") or adt.get("properties", {}).get("hostName")
if adt_host_name:
    print(f"   URL: https://{adt_host_name}")
else:
    print(f"   Response: {json.dumps(adt, indent=2)[:500]}")

# 3. Create Storage Account
print("\n3. Creating Storage Account...")
r = requests.put(
    f"https://management.azure.com/subscriptions/{SUB_ID}/resourceGroups/{RG_NAME}/providers/Microsoft.Storage/storageAccounts/{STORAGE_NAME}?api-version=2023-05-01",
    headers=headers,
    json={"location": LOCATION, "sku": {"name": "Standard_LRS"}, "kind": "StorageV2"}
)
print(f"   Status: {r.status_code}")
if r.status_code in (200, 202):
    print(f"   Name: {STORAGE_NAME}")
else:
    print(f"   Response: {r.text[:300]}")

print("\nDone! Resources are being provisioned.")
if adt_host_name:
    print(f"\nSet this env var:")
    print(f'  $env:AZURE_DIGITAL_TWINS_URL = "https://{adt_host_name}"')
else:
    print("\nGet your ADT URL with: az dt show --dt-name <name> --query hostName -o tsv")
