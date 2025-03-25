
# Face Mask Detection using CNN and OpenCV

## 📌 Project Description
This project is a deep learning-based face mask detection system. It uses a Convolutional Neural Network (CNN) model built with TensorFlow and Keras to detect whether a person in an image or video is wearing a mask or not. The system also integrates OpenCV for real-time face detection and classification.


## 🛠 Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Numpy
- Haar Cascade Classifier

---

## 📂 Project Structure
```
├── face.py                      # Real-time detection script
├── model.py                     # Model training script
├── mymodel.h5                   # Trained CNN model
├── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
├── new.JPG                      # Sample image for testing
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🔧 Installation & Setup
1. **Clone the repository**
```
git clone https://github.com/Bucky51/facemask-Detection.git
cd facemask-Detection
```

2. **Install the dependencies**
```
pip install -r requirements.txt
```

3. **Download the Haar Cascade XML**
Ensure `haarcascade_frontalface_default.xml` is in the working directory.

4. **Run the model training (Optional)**
```
python model.py
```
It saves the trained model as `mymodel1.h5`.

5. **Run the face mask detection**
```
python face.py
```

---

## 🧠 Model Architecture
- 3 Convolutional layers (32 filters each)
- MaxPooling after each Conv layer
- Flatten layer
- Dense layer with 100 neurons (ReLU)
- Output layer (Sigmoid for binary classification)

---

## ✅ Features
- Real-time face mask detection
- Trained on custom dataset
- Simple and easy-to-use
- Highly customizable

---

## 💻 Requirements
See `requirements.txt`.  
Install the following major libraries:
```
TensorFlow==2.5.0
Keras==2.4.3
OpenCV-python==4.4.0.46
numpy==1.19.4
```

---

## 📌 Sample Usage
```python
# Inside face.py
model = load_model('mymodel.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```
Run the script to perform detection on the webcam feed or images.

---

## 🚀 Future Improvements
- Improve model accuracy with more dataset
- Add multi-class detection (incorrect mask wearing)
- Deploy as a web app or mobile app

---

## 🙌 Acknowledgements
- TensorFlow and Keras team
- OpenCV community
- Dataset contributors

---

## 📜 License
This project is for educational purposes only.
