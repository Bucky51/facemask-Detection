
# 😷 Face Mask Detection

## 📖 Overview
This project is a **Face Mask Detection System** built using Python, OpenCV, and deep learning. It can detect whether a person is wearing a mask or not in real-time using a webcam. The system leverages a pre-trained deep learning model to classify faces with or without masks.

---

## 📂 Project Structure
```
facemask-Detection/
│
├── dataset/                 # Contains mask and no-mask image dataset
├── face_detector/           # Pre-trained face detection model (Caffe model)
├── mask_detector.model      # Trained Keras model for mask detection
├── detect_mask_video.py     # Script to run real-time mask detection
├── train_mask_detector.py   # Script to train the mask detection model
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Bucky51/facemask-Detection.git
cd facemask-Detection
```

### 2. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 3. Download Face Detection Model Files (if not already in `face_detector/`)
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

You can download them from the official OpenCV repository:  
[OpenCV Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

Place them in the `face_detector/` directory.

---

## 🚀 Usage

### Run Real-Time Mask Detection
```bash
python detect_mask_video.py
```
This will open your webcam and start detecting faces with or without masks in real-time.

### Optional: Train the Model
If you want to retrain the mask detection model:
```bash
python train_mask_detector.py
```
Ensure the dataset is properly structured before training.

---

## 🧠 Technologies Used
- Python 3.x
- OpenCV
- Keras / TensorFlow
- NumPy
- Imutils

---

## 📈 Future Enhancements
- Improve accuracy with a larger dataset
- Integrate with CCTV/video surveillance systems
- Deploy as a web application or mobile app
- Send alerts for mask violations

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments
- OpenCV for the face detection model
- Keras/TensorFlow for the deep learning framework
- Dataset contributions from online open-source resources

---

## 💻 Author
Developed by [Bucky51](https://github.com/Bucky51)
