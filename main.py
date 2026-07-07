import sqlite3
import csv
import os
from datetime import datetime
import time
from ultralytics import YOLO
import cv2
from playsound import playsound
import threading


model = YOLO("runs/detect/train/weights/best.pt")


os.makedirs("screenshots", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("database", exist_ok=True)


conn = sqlite3.connect("database/wildlife.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS detections(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    time TEXT,
    animal TEXT,
    confidence REAL,
    screenshot TEXT
)
""")
conn.commit()


csv_file = "logs/detections.csv"

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Time", "Animal", "Confidence", "Screenshot"])


animals = [
    "lion",
    "tiger",
    "cheetah",
    "fox",
    "hyena",
    "wolf"
]


dangerous_animals = [
    "lion",
    "tiger",
    "cheetah",
    "hyena",
    "wolf"
]


alarm_playing = False
last_alarm_time = 0
cooldown = 10

def play_alarm():
    global alarm_playing
    alarm_playing = True
    playsound("assets/alarm.mp3")
    alarm_playing = False


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.80)

    for result in results:
            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls].lower().strip()

                print(f"Detected: {label} ({conf:.2f})")

                if label not in animals:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (0, 255, 0)

                if label in dangerous_animals:
                    color = (0, 0, 255)

                    cv2.putText(
                        frame,
                        f"WARNING: {label.upper()}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                    current_time = time.time()

                    if not alarm_playing and current_time - last_alarm_time > cooldown:

                        last_alarm_time = current_time

                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        filename = f"screenshots/{label}_{timestamp}.jpg"

                        cv2.imwrite(filename, frame)

                        with open(csv_file, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                datetime.now().strftime("%Y-%m-%d"),
                                datetime.now().strftime("%H:%M:%S"),
                                label,
                                round(conf, 2),
                                filename
                            ])

                        cursor.execute("""
                        INSERT INTO detections
                        (date,time,animal,confidence,screenshot)
                        VALUES (?,?,?,?,?)
                        """, (
                            datetime.now().strftime("%Y-%m-%d"),
                            datetime.now().strftime("%H:%M:%S"),
                            label,
                            round(conf, 2),
                            filename
                        ))

                        conn.commit()

                        threading.Thread(
                            target=play_alarm,
                            daemon=True
                        ).start()

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

    cv2.imshow("Wild Animal Detection", frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
conn.close()