import urllib.request
import os

dest = os.path.join(os.path.expanduser("~"), "Downloads", "AzureCLI.msi")
print(f"Downloading Azure CLI to {dest}...")
urllib.request.urlretrieve("https://aka.ms/installazurecliwindowsx64", dest)
print(f"Done! Now run: msiexec /i {dest} /quiet")
