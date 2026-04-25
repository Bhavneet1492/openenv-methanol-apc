"""Find Azure tenants and subscriptions."""
from azure.identity import InteractiveBrowserCredential
import requests

cred = InteractiveBrowserCredential()
token = cred.get_token("https://management.azure.com/.default")
headers = {"Authorization": f"Bearer {token.token}"}

# List tenants
r = requests.get("https://management.azure.com/tenants?api-version=2022-12-01", headers=headers)
tenants = r.json().get("value", [])
print(f"Found {len(tenants)} tenant(s):")
for t in tenants:
    tid = t.get("tenantId", "?")
    name = t.get("displayName", "?")
    print(f"  Tenant: {name} | ID: {tid}")

    # Try listing subs in this tenant
    try:
        cred2 = InteractiveBrowserCredential(tenant_id=tid)
        token2 = cred2.get_token("https://management.azure.com/.default")
        headers2 = {"Authorization": f"Bearer {token2.token}"}
        r2 = requests.get("https://management.azure.com/subscriptions?api-version=2022-12-01", headers=headers2)
        subs = r2.json().get("value", [])
        for s in subs:
            print(f"    Sub: {s['displayName']} | ID: {s['subscriptionId']} | State: {s['state']}")
        if not subs:
            print("    (no subscriptions)")
    except Exception as e:
        print(f"    Error: {e}")
