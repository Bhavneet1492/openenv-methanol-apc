"""Register Microsoft.DigitalTwins provider and create ADT instance."""
from azure.identity import InteractiveBrowserCredential
import requests
import time

SUB_ID = "a87ef111-a233-4bec-a754-58b02f39cc2b"
TENANT_ID = "4803f9ef-12cd-46f4-ad6c-c5245df0714f"
RG_NAME = "methanol-apc-rg"
LOCATION = "eastus"
ADT_NAME = "methanol-apc-adt"

cred = InteractiveBrowserCredential(tenant_id=TENANT_ID)
token = cred.get_token("https://management.azure.com/.default")
headers = {"Authorization": f"Bearer {token.token}", "Content-Type": "application/json"}
print("Authenticated!")

# 1. Register Microsoft.DigitalTwins provider
print("\n1. Registering Microsoft.DigitalTwins provider...")
r = requests.post(
    f"https://management.azure.com/subscriptions/{SUB_ID}/providers/Microsoft.DigitalTwins/register?api-version=2021-04-01",
    headers=headers
)
print(f"   Status: {r.status_code}")
state = r.json().get("registrationState", "?")
print(f"   Registration state: {state}")

# Wait for registration
if state != "Registered":
    print("   Waiting for registration (can take 1-2 minutes)...")
    for i in range(12):
        time.sleep(10)
        r2 = requests.get(
            f"https://management.azure.com/subscriptions/{SUB_ID}/providers/Microsoft.DigitalTwins?api-version=2021-04-01",
            headers=headers
        )
        state = r2.json().get("registrationState", "?")
        print(f"   [{i*10}s] State: {state}")
        if state == "Registered":
            break

# 2. Create ADT instance
print(f"\n2. Creating Azure Digital Twins: {ADT_NAME}...")
r = requests.put(
    f"https://management.azure.com/subscriptions/{SUB_ID}/resourceGroups/{RG_NAME}/providers/Microsoft.DigitalTwins/digitalTwinsInstances/{ADT_NAME}?api-version=2023-01-31",
    headers=headers, json={"location": LOCATION}
)
print(f"   Status: {r.status_code}")
adt = r.json()
if r.status_code in (200, 201):
    host = adt.get("properties", {}).get("hostName", "")
    print(f"   Hostname: {host}")
    print(f"\n   Set env var:")
    print(f'   $env:AZURE_DIGITAL_TWINS_URL = "https://{host}"')
else:
    print(f"   Response: {r.text[:500]}")
