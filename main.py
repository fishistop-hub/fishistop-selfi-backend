import os
import io
import json
import base64
import boto3
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

app = Flask(__name__)
CORS(app, origins=["https://photo-app-sandy-ten.vercel.app", "http://localhost:5173"])

# ── AWS clients ──────────────────────────────────────────────────────────────
rekognition = boto3.client(
    "rekognition",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
)

# ── Google Drive ─────────────────────────────────────────────────────────────
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


# ── Rekognition Collection helpers ───────────────────────────────────────────
def get_collection_id(folder_id):
    """Each Drive folder gets its own Rekognition collection."""
    return f"fishi-{folder_id}"


def collection_exists(collection_id):
    try:
        rekognition.describe_collection(CollectionId=collection_id)
        return True
    except rekognition.exceptions.ResourceNotFoundException:
        return False


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "fishi-selfi-backend"})


@app.route("/index-event", methods=["POST", "GET"])
def index_event():
    """
    Index all photos in a Drive folder into a Rekognition collection.
    Call this ONCE per event after uploading photos to Drive.
    
    Body: { "event_folder_id": "..." }
    """
    data = request.get_json(force=True)
    folder_id = data.get("event_folder_id")

    if not folder_id:
        return jsonify({"error": "event_folder_id is required"}), 400

    collection_id = get_collection_id(folder_id)

    # Create collection if it doesn't exist
    if not collection_exists(collection_id):
        rekognition.create_collection(CollectionId=collection_id)

    # List all photos in Drive folder
    try:
        files = list_images_in_folder(folder_id)
    except Exception as e:
        return jsonify({"error": f"Drive error: {str(e)}"}), 500

    indexed = 0
    failed = 0
    for f in files:
        try:
            photo_bytes = download_image_bytes(f["id"])
            rekognition.index_faces(
                CollectionId=collection_id,
                Image={"Bytes": photo_bytes},
                ExternalImageId=f["id"],  # store Drive file ID
                DetectionAttributes=[],
                MaxFaces=10,
            )
            indexed += 1
        except Exception:
            failed += 1
            continue

    return jsonify({
        "collection_id": collection_id,
        "total_photos": len(files),
        "indexed": indexed,
        "failed": failed,
        "status": "ready" if indexed > 0 else "no_faces_found"
    })


@app.route("/find-photos", methods=["POST"])
def find_photos():
    """
    Search for a face in the event collection.
    Fast — uses Rekognition collection search, no photo downloading.
    
    Body: { "selfie": "<base64>", "event_folder_id": "..." }
    """
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

    collection_id = get_collection_id(folder_id)

    # Check if collection exists
    if not collection_exists(collection_id):
        return jsonify({
            "error": "Event not indexed yet. Please run /index-event first.",
            "matches": []
        }), 400

    # Search collection with selfie
    try:
        response = rekognition.search_faces_by_image(
            CollectionId=collection_id,
            Image={"Bytes": selfie_bytes},
            MaxFaces=100,
            FaceMatchThreshold=70.0,
        )
    except rekognition.exceptions.InvalidParameterException:
        return jsonify({"matches": [], "total_scanned": 0, "message": "No face detected in selfie"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    face_matches = response.get("FaceMatches", [])

    # Build results using Drive file IDs stored in ExternalImageId
    matches = []
    seen_ids = set()
    for match in face_matches:
        file_id = match["Face"]["ExternalImageId"]
        if file_id not in seen_ids:
            seen_ids.add(file_id)
            matches.append({
                "file_id": file_id,
                "similarity": round(match["Face"]["Confidence"], 1),
                "url": f"https://drive.google.com/uc?export=view&id={file_id}",
                "download_url": f"https://drive.google.com/uc?export=download&id={file_id}",
            })

    return jsonify({
        "matches": matches,
        "total_scanned": len(face_matches),
    })


@app.route("/collection-status", methods=["GET"])
def collection_status():
    """Check if an event folder has been indexed."""
    folder_id = request.args.get("event_folder_id")
    if not folder_id:
        return jsonify({"error": "event_folder_id required"}), 400

    collection_id = get_collection_id(folder_id)
    exists = collection_exists(collection_id)

    if exists:
        info = rekognition.describe_collection(CollectionId=collection_id)
        return jsonify({
            "indexed": True,
            "face_count": info.get("FaceCount", 0),
            "collection_id": collection_id,
        })
    else:
        return jsonify({"indexed": False, "collection_id": collection_id})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
