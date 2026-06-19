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
import face_recognition
from PIL import Image

app = Flask(__name__)
CORS(app)

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
    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf.read()

def get_face_encoding(image_bytes):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        f.write(image_bytes)
        path = f.name
    img = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(img)
    os.unlink(path)
    return encodings[0] if encodings else None

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

        selfie_b64 = selfie_b64.split(',')[-1]
        selfie_bytes = base64.b64decode(selfie_b64)

        selfie_encoding = get_face_encoding(selfie_bytes)
        if selfie_encoding is None:
            return jsonify({'error': 'No face detected in selfie'}), 400

        photos = list_photos(event_id)
        if not photos:
            return jsonify({'matches': [], 'message': 'No photos found'})

        matches = []
        for photo in photos:
            try:
                photo_bytes = download_photo(photo['id'])
                photo_encoding = get_face_encoding(photo_bytes)
                if photo_encoding is None:
                    continue
                result = face_recognition.compare_faces([selfie_encoding], photo_encoding, tolerance=0.5)
                if result[0]:
                    matches.append(f"https://drive.google.com/uc?id={photo['id']}&export=view")
            except Exception as e:
                print(f"Error: {e}")
                continue

        return jsonify({'matches': matches, 'count': len(matches)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
