import os
import json
import base64
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import cv2
import insightface
from insightface.app import FaceAnalysis

app = Flask(__name__)
CORS(app)

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Load face model once at startup
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

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
        fields="files(id, name)",
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

def get_embedding(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    faces = face_app.get(img)
    if not faces:
        return None
    return faces[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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

        selfie_embedding = get_embedding(selfie_bytes)
        if selfie_embedding is None:
            return jsonify({'error': 'No face detected in selfie'}), 400

        photos = list_photos(event_id)
        if not photos:
            return jsonify({'matches': [], 'message': 'No photos found'})

        matches = []
        for photo in photos:
            try:
                photo_bytes = download_photo(photo['id'])
                photo_embedding = get_embedding(photo_bytes)
                if photo_embedding is None:
                    continue
                sim = cosine_similarity(selfie_embedding, photo_embedding)
                if sim > 0.4:
                    matches.append(f"https://drive.google.com/uc?id={photo['id']}&export=view")
            except Exception as e:
                print(f"Error processing photo: {e}")
                continue

        return jsonify({'matches': matches, 'count': len(matches)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
