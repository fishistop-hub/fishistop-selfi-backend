import os
import io
import json
import base64
import boto3
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

app = Flask(__name__)
CORS(app, origins=["https://photo-app-sandy-ten.vercel.app", "http://localhost:5173"])

rekognition = boto3.client(
    "rekognition",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
)

def get_drive_service():
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not creds_json:
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON env var not set")
    creds_dict = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)

def list_images_in_folder(folder_id):
    service = get_drive_service()
    query = f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false"
    results = []
    page_token = None
    while True:
        resp = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name)",
            pageSize=100,
            pageToken=page_token,
        ).execute()
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results

def download_image_bytes(file_id):
    service = get_drive_service()
    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()

def compare_faces(selfie_bytes, photo_bytes, threshold=80.0):
    try:
        resp = rekognition.compare_faces(
            SourceImage={"Bytes": selfie_bytes},
            TargetImage={"Bytes": photo_bytes},
            SimilarityThreshold=threshold,
        )
        return len(resp.get("FaceMatches", [])) > 0
    except Exception:
        return False

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "fishi-selfi-backend"})

@app.route("/find-photos", methods=["POST"])
def find_photos():
    data = request.get_json(force=True)
    selfie_b64 = data.get("selfie")
    folder_id = data.get("event_folder_id")

    if not selfie_b64 or not folder_id:
        return jsonify({"error": "selfie and event_folder_id are required"}), 400

    if "," in selfie_b64:
        selfie_b64 = selfie_b64.split(",", 1)[1]
    try:
        selfie_bytes = base64.b64decode(selfie_b64)
    except Exception:
        return jsonify({"error": "Invalid base64 for selfie"}), 400

    try:
        files = list_images_in_folder(folder_id)
    except Exception as e:
        return jsonify({"error": f"Drive error: {str(e)}"}), 500

    if not files:
        return jsonify({"matches": [], "total_scanned": 0})

    matches = []
    for f in files:
        try:
            photo_bytes = download_image_bytes(f["id"])
        except Exception:
            continue
        if compare_faces(selfie_bytes, photo_bytes):
            matches.append({
                "file_id": f["id"],
                "name": f["name"],
                "url": f"https://drive.google.com/uc?export=view&id={f['id']}",
                "download_url": f"https://drive.google.com/uc?export=download&id={f['id']}",
            })

    return jsonify({"matches": matches, "total_scanned": len(files)})

@app.route("/events", methods=["GET"])
def list_events():
    root_id = request.args.get("root_folder_id")
    if not root_id:
        return jsonify({"error": "root_folder_id required"}), 400
    try:
        service = get_drive_service()
        query = f"'{root_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        resp = service.files().list(q=query, fields="files(id, name)", pageSize=50).execute()
        folders = resp.get("files", [])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"events": folders})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
