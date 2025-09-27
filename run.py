from app import create_app
from dotenv import load_dotenv
import os
import nltk

# --- NLTK resource check ---
nltk_resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "cmudict",
    "averaged_perceptron_tagger_eng",
]

for resource in nltk_resources:
    try:
        nltk.data.find(f"tokenizers/{resource}") if "punkt" in resource else nltk.data.find(f"corpora/{resource}")
    except LookupError:
        print(f"[NLTK] Resource '{resource}' not found. Downloading...")
        nltk.download(resource)
    else:
        print(f"[NLTK] Resource '{resource}' is already available.")

# Load environment variables
load_dotenv()

app = create_app()

if __name__ == '__main__':
    # Get configuration from environment
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    print(f"Starting PredscanAI on http://{host}:{port}")
    print("Default admin login: admin@predscan.ai / admin123")
    
    app.run(debug=debug, host=host, port=port)