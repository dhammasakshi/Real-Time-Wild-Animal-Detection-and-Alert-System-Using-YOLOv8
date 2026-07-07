#  Real-Time Wild Animal Detection and Alert System Using YOLOv8

A real-time computer vision application that detects wild animals using a custom-trained YOLOv8 model and OpenCV. The system processes live webcam footage, identifies animals, displays bounding boxes with confidence scores, triggers an alarm for detections, captures screenshots, and stores detection records in CSV and SQLite databases.

---

## Features

- Real-time animal detection using a webcam
- Custom-trained YOLOv8 object detection model
- Detects the following animals:
  - Lion
  - Tiger
  - Bear
  - Cheetah
  - Fox
  - Hyena
  - Wolf
- Displays bounding boxes and confidence scores
- Plays an alarm when an animal is detected
- Automatically captures screenshots
- Stores detection records in CSV and SQLite database

---

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- SQLite
- NumPy

---

## Project Structure

```
Wild-Animal-Detection-System/
│
├── assets/
│   └── alarm.mp3
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
```

---

## Run the Project

```bash
python main.py
```

---

## Output

- Detects wild animals in real time
- Draws bounding boxes with confidence scores
- Plays an alarm when an animal is detected
- Automatically captures screenshots
- Logs detections into CSV and SQLite database

---

## Future Enhancements

- Improve detection accuracy with a larger and more diverse dataset
- Add support for additional wild animal species
- Develop a Streamlit-based web interface
- Optimize the model for deployment on edge devices such as Raspberry Pi

---

## Author

**Dhammasakshi Salve**

Machine Learning Enthusiast 

