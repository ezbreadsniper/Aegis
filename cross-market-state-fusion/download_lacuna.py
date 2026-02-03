import requests
import os

URL_BASE = "https://huggingface.co/HumanPlane/LACUNA/resolve/main/"
FILES_TO_TRY = [
    "model.safetensors", 
    "model.pt", 
    "pytorch_model.bin",
    "lacuna_v5.safetensors", 
    "config.json" # To verify access
]
DEST_DIR = "."

def download_file(filename):
    url = f"{URL_BASE}{filename}?download=true"
    print(f"⬇️ Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(os.path.join(DEST_DIR, filename), 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Saved to {filename}")
            return True
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    return False

# Main download loop
success = False
for f in FILES_TO_TRY:
    if download_file(f):
        success = True
        break

if not success:
    print("⚠️  Could not find any weights files.")
else:
    print("🚀 Weights ready.")
