from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import shutil
from pathlib import Path
import numpy as np
import cv2
import face_recognition

app = FastAPI()

def np_to_py(obj):
    if isinstance(obj, dict):
        return {k: np_to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [np_to_py(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj



@app.post("/bedom-bild/")
async def assess_image(file: UploadFile = File(...)):
    output_folder = Path("/output")
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / file.filename

    # Spara uppladdad bild till output-mappen
    with output_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Öppna bilden med Pillow för analys
    try:
        with Image.open(output_file) as img:
            width, height = img.size
            format = img.format
            img_rgb = np.array(img.convert("RGB"))
    except Exception as e:
        return JSONResponse(content={
            "filename": file.filename,
            "status": "error",
            "error": f"Failed to open image: {str(e)}"
        }, status_code=400)

    # Enkel kvalitetskontroll
    MIN_WIDTH = 600
    MIN_HEIGHT = 400
    min_size_ok = width >= MIN_WIDTH and height >= MIN_HEIGHT

    assessment = {
        "filename": file.filename,
        "format": format,
        "width": width,
        "height": height,
        "size_bytes": output_file.stat().st_size,
        "min_size_ok": min_size_ok
    }

    # Face recognition: Hitta ansikten
    face_locations = face_recognition.face_locations(img_rgb)
    num_faces = len(face_locations)
    assessment["num_faces"] = num_faces
    assessment["face_detected"] = num_faces == 1

    if num_faces == 1:
        top, right, bottom, left = face_locations[0]
        face_width = right - left
        face_height = bottom - top
        face_center_x = (left + right) // 2
        face_center_y = (top + bottom) // 2

        # Kontrollera att ansiktet är centrerat
        center_margin_x = width * 0.20  # 20% från mitten godkänns
        center_margin_y = height * 0.20
        assessment["face_centered"] = (
            abs(face_center_x - width // 2) < center_margin_x and
            abs(face_center_y - height // 2) < center_margin_y
        )

        # Kontrollera att ansiktet är tillräckligt stort
        min_face_height = height * 0.3
        assessment["face_large_enough"] = face_height >= min_face_height

        # Hämta facial landmarks för mer avancerad analys
        face_landmarks = face_recognition.face_landmarks(img_rgb, [face_locations[0]])
        if face_landmarks:
            left_eye = face_landmarks[0].get("left_eye", [])
            right_eye = face_landmarks[0].get("right_eye", [])
            # Ögonen synliga om vi hittat båda
            assessment["eyes_detected"] = len(left_eye) > 0 and len(right_eye) > 0
        else:
            assessment["eyes_detected"] = False

        # (Valfritt) Kontrollera "pose"/rotation - exempel (begränsat med face_recognition):
        # Om ögonhöjdsskillnad är för stor tyder det på lutat huvud
        if face_landmarks and "left_eye" in face_landmarks[0] and "right_eye" in face_landmarks[0]:
            left_eye_y = np.mean([pt[1] for pt in face_landmarks[0]["left_eye"]])
            right_eye_y = np.mean([pt[1] for pt in face_landmarks[0]["right_eye"]])
            eye_level_diff = abs(left_eye_y - right_eye_y)
            assessment["head_tilt_ok"] = eye_level_diff < (face_height * 0.07)  # max 7% lutning
        else:
            assessment["head_tilt_ok"] = None
    else:
        assessment.update({
            "face_centered": False,
            "face_large_enough": False,
            "eyes_detected": False,
            "head_tilt_ok": False
        })

    # (Valfritt) Kontrollera bakgrund - här mäts std-avvikelse i hörnen:
    def background_std_area(img_rgb, x, y, w, h):
        area = img_rgb[y:y+h, x:x+w, :]
        return np.std(area)

    margin = 50
    areas = [
        background_std_area(img_rgb, 0, 0, margin, margin),
        background_std_area(img_rgb, width-margin, 0, margin, margin),
        background_std_area(img_rgb, 0, height-margin, margin, margin),
        background_std_area(img_rgb, width-margin, height-margin, margin, margin),
    ]
    assessment["background_uniformity"] = max(areas) < 25  # justera threshold efter behov

    # Slutgiltigt resultat
    required = [
        assessment["min_size_ok"],
        assessment["face_detected"],
        assessment["face_centered"],
        assessment["face_large_enough"],
        assessment["eyes_detected"],
        assessment["head_tilt_ok"] is None or assessment["head_tilt_ok"],
        assessment["background_uniformity"],
    ]
    assessment["result"] = "OK" if all(required) else "not_suitable"

    return JSONResponse(content=np_to_py(assessment))

