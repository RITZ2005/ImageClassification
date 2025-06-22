import joblib
import json
import numpy as np
import base64
import cv2
import os
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image_base64_data, file_path=None):
    try:
        print("\n=== Starting Image Classification ===")
        print(f"Processing image: {file_path}")
        imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
        print(f"Found {len(imgs)} faces")
        
        if not imgs:
            print("No faces detected")
            return None

        result = []
        for i, img in enumerate(imgs):
            print(f"\nProcessing face {i+1}")
            print("Resizing image...")
            scalled_raw_img = cv2.resize(img, (32, 32))
            
            print("Applying wavelet transform...")
            img_har = w2d(img, 'db1', 5)
            scalled_img_har = cv2.resize(img_har, (32, 32))
            
            print("Combining features...")
            combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

            len_image_array = 32*32*3 + 32*32
            print(f"Final feature vector length: {len_image_array}")

            final = combined_img.reshape(1,len_image_array).astype(float)
            print("Making prediction...")
            prediction = __model.predict(final)[0]
            probabilities = __model.predict_proba(final)[0]
            
            print(f"Raw prediction: {prediction}")
            print(f"Class probabilities: {probabilities}")
            print(f"Predicted class name: {class_number_to_name(prediction)}")
            
            result.append({
                'class': class_number_to_name(prediction),
                'class_probability': np.around(probabilities*100,2).tolist(),
                'class_dictionary': __class_name_to_number
            })

        print("=== Classification Complete ===\n")
        return result
    except Exception as e:
        print(f"Error in classify_image: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return None

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    try:
        with open(os.path.join("server", "artifacts", "class_dictionary.json"), "r") as f:
            __class_name_to_number = json.load(f)
            __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

        global __model
        if __model is None:
            with open(os.path.join("server", "artifacts", "saved_model.pkl"), 'rb') as f:
                __model = joblib.load(f)
        print("loading saved artifacts...done")
    except Exception as e:
        print(f"Error loading artifacts: {str(e)}")
        raise

def get_cv2_image_from_base64_string(b64str):
    try:
        encoded_data = b64str.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {str(e)}")
        return None

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    try:
        print("\n=== Starting Face Detection ===")
        print("Loading cascade classifiers...")
        face_cascade = cv2.CascadeClassifier(os.path.join('server', 'opencv', 'haarcascades', 'haarcascade_frontalface_default.xml'))
        eye_cascade = cv2.CascadeClassifier(os.path.join('server', 'opencv', 'haarcascades', 'haarcascade_eye.xml'))

        if image_path:
            print(f"Reading image from path: {image_path}")
            img = cv2.imread(image_path)
        else:
            print("Reading image from base64")
            img = get_cv2_image_from_base64_string(image_base64_data)

        if img is None:
            print("Failed to read image")
            return []

        print("Converting to grayscale...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Detecting faces...")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(f"Found {len(faces)} faces")

        cropped_faces = []
        for i, (x,y,w,h) in enumerate(faces):
            print(f"\nProcessing face {i+1}")
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            print("Detecting eyes...")
            eyes = eye_cascade.detectMultiScale(roi_gray)
            print(f"Found {len(eyes)} eyes in face")
            if len(eyes) >= 2:
                print("Face has 2 or more eyes, keeping it")
                cropped_faces.append(roi_color)
            else:
                print("Face doesn't have enough eyes, skipping")

        print(f"=== Face Detection Complete: Found {len(cropped_faces)} valid faces ===\n")
        return cropped_faces
    except Exception as e:
        print(f"Error in get_cropped_image_if_2_eyes: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return []

def get_b64_test_image_for_virat():
    with open("server/b64.txt") as f:
        return f.read()

if __name__ == '__main__':
    load_saved_artifacts()

   # print(classify_image(get_b64_test_image_for_virat(), None))
    print(classify_image(None, os.path.join('model', 'test_images', 'sharapova1.jpg')))

   