"""Fix encoding corruption in process-flow.svg"""
import os

path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "process-flow.svg")

# Read as raw bytes
with open(path, "rb") as f:
    raw = f.read()

# The file was written as UTF-8 but then re-read as Latin-1 and saved again,
# causing double-encoding. Fix by decoding as Latin-1 then re-encoding as UTF-8.
try:
    text = raw.decode("utf-8")
except:
    text = raw.decode("latin-1")

# Replace all corrupted multi-byte sequences with ASCII equivalents
replacements = {
    "â€"": "-",
    "â€"": "-", 
    "â™»": "",
    "â†'": "to",
    "â‚‚": "2",
    "â‚ƒ": "3",
    "â‚„": "4",
    "Â°": " ",
    "Â·": "/",
    "â‰ˆ": "~",
    "â˜…": "*",
    "â•": "=",
}

for old, new in replacements.items():
    text = text.replace(old, new)

# Write back as clean UTF-8 without BOM
with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print(f"Fixed. File size: {os.path.getsize(path)} bytes")
