#  ImageClassification1

A deep learning–based Flask web application that classifies images of **sports celebrities** using a trained model and OpenCV. The app allows users to upload an image and receive instant classification results with label prediction.

---

#🚀 Features

- 🖼 Upload image interface
-  Pre-trained deep learning model for celebrity recognition
-  Flask backend API for prediction
-  Face detection using OpenCV Haar cascades
-  Real-time inference with output displayed on web page

  ## 🗂️ Project Structure

  imageclassification/
│
├── model/
│ └── imageclass.ipynb # Jupyter notebook for training
│ └── trained_model.pkl/.h5 # Model file
│ └── opencv/haarcascades/ # Haar cascade XMLs for face detection
│
├── server/
│ └── server.py # Flask backend server
│ └── templates/index.html # Frontend UI
│ └── static/ # CSS / image assets
│ └── opencv/haarcascades/ # Haar cascades used by server
│
├── requirements.txt # Required Python packages
└── README.md # Project documentation
