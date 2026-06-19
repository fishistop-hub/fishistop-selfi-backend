import os
import json
import base64
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import numpy as np
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

# ── Google Drive Setup ─────────────────────────────────────────────────────
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    creds_json = os.environ.get('GOOGLE_CREDENTIALS')
    if not creds_json:
        raise Exception("GOOGLE_CREDENTIALS not set")
    creds_dict = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=SCOPES
    )
    return build('drive', 'v3', credentials=creds)

def list_photos(folder_id):
    service = get_drive_service()
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType contains 'image/'",
        fields="files(id, name, mimeType)",
        pageSize=1000
    ).execute()
    return results.get('files', [])

def download_photo(file_id):
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf.read()

# ── Face Matching ──────────────────────────────────────────────────────────
def match_face(selfie_data, event_photos):
    matches = []

    # Save selfie to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        f.write(selfie_data)
        selfie_path = f.name

    for photo in event_photos:
        try:
            photo_bytes = download_photo(photo['id'])
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                f.write(photo_bytes)
                photo_path = f.name

            result = DeepFace.verify(
                selfie_path,
                photo_path,
                model_name='VGG-Face',
                enforce_detection=False
            )

            if result['verified']:
                # Return public Drive link
                matches.append({
                    'id': photo['id'],
                    'name': photo['name'],
                    'url': f"https://drive.google.com/uc?id={photo['id']}&export=view"
                })

            os.unlink(photo_path)

        except Exception as e:
            print(f"Error processing photo {photo['name']}: {e}")
            continue

    os.unlink(selfie_path)
    return matches

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/match', methods=['POST'])
def match():
    try:
        data = request.json
        selfie_b64 = data.get('selfie', '')
        event_id = data.get('eventId', '')

        if not selfie_b64 or not event_id:
            return jsonify({'error': 'Missing selfie or eventId'}), 400

        # Decode selfie from base64
        selfie_b64 = selfie_b64.split(',')[-1]  # remove data:image/jpeg;base64,
        selfie_bytes = base64.b64decode(selfie_b64)

        # Get photos from Drive folder
        photos = list_photos(event_id)
        if not photos:
            return jsonify({'matches': [], 'message': 'No photos in this event folder'})

        # Match faces
        matched = match_face(selfie_bytes, photos)

        return jsonify({
            'matches': [m['url'] for m in matched],
            'count': len(matched)
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
