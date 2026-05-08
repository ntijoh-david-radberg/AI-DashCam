from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import easyocr
import re
import time
import os

app = Flask(__name__)

MODEL_PATH = "best.pt"

latest_plate = "Ingen skylt hittad"
latest_confidence = 0
latest_time = "Aldrig"
latest_image_path = "static/latest_plate.jpg"

os.makedirs("static", exist_ok=True)

model = YOLO(MODEL_PATH)
reader = easyocr.Reader(["en"], gpu=False)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

time.sleep(2)


def clean_plate_text(text):
    text = text.upper()
    text = text.replace(" ", "")
    text = re.sub(r"[^A-Z0-9]", "", text)

    match = re.search(r"[A-Z]{3}[0-9]{2}[A-Z0-9]", text)

    if match:
        return match.group(0)

    return None


def read_plate_text(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    results = reader.readtext(gray)

    for result in results:
        text = result[1]
        cleaned = clean_plate_text(text)

        if cleaned:
            return cleaned

    return None


def generate_frames():
    global latest_plate, latest_confidence, latest_time

    while True:
        frame = picam2.capture_array()

        results = model(frame, conf=0.5, verbose=False)

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                plate_img = frame[y1:y2, x1:x2]

                if plate_img.size == 0:
                    continue

                reg_number = read_plate_text(plate_img)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = "REG SKYLT"

                if reg_number:
                    latest_plate = reg_number
                    latest_confidence = round(confidence * 100, 1)
                    latest_time = time.strftime("%H:%M:%S")

                    cv2.imwrite(latest_image_path, cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR))

                    label = f"{reg_number} ({latest_confidence}%)"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/plate_data")
def plate_data():
    biluppgifter_url = ""

    if latest_plate != "Ingen skylt hittad":
        biluppgifter_url = f"https://biluppgifter.se/fordon/{latest_plate}"

    return jsonify({
        "plate": latest_plate,
        "confidence": latest_confidence,
        "time": latest_time,
        "image": "/static/latest_plate.jpg",
        "biluppgifter_url": biluppgifter_url
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)