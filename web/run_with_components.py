import os
import glob
from pathlib import Path

print("Starting Chrome/Selenium blocker...")

# Get base directory
base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
print(f"Searching in: {base_dir}")

# Specific targets to look for
targets = [
    "**/data/agents*/*/forge/data/assets/crx/*.crx",  # Chrome extensions
    "**/chromedriver*.exe",  # ChromeDriver executables
    "**/drivers/**/*.exe",    # Any driver executables
]

# Find and rename files
for pattern in targets:
    for file_path in glob.glob(os.path.join(base_dir, pattern), recursive=True):
        if os.path.exists(file_path):
            print(f"Found target: {file_path}")
            try:
                # Rename with .disabled extension
                disabled_path = file_path + ".disabled"
                if not os.path.exists(disabled_path):
                    os.rename(file_path, disabled_path)
                    print(f"Disabled: {file_path}")
                else:
                    print(f"Already disabled: {file_path}")
            except Exception as e:
                print(f"Error disabling {file_path}: {e}")

# Create a custom .env file with settings to disable browser
env_path = os.path.join(base_dir, ".env.nobrowser")
with open(env_path, "w") as f:
    f.write("""
# Disable browser-related features
SELENIUM_WEB_BROWSER=none
USE_WEB_BROWSER=false
HEADLESS_BROWSER=true
BROWSE_CHUNK_MAX_LENGTH=0
BROWSE_SPACY_LANGUAGE_MODEL=none
SELENIUM_TIMEOUT=1
MEMORY_BACKEND=simple
""")

print(f"Created {env_path} with browser-disabling settings")
print("\nTo use AutoGPT without any browser capability, run with:")
print(f"python -m autogpt --env-file {env_path}")
print("Or copy these settings to your main .env file")