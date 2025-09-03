import cv2
import numpy as np
import os
import face_recognition
import time

# Function to find encodings of images
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:  # Ensure face encoding exists
            encode_list.append(encodes[0])
    return encode_list

# Function to load training images and their encodings
def load_training_images(path):
    images = []
    class_names = []
    if not os.path.exists(path):
        os.makedirs(path)
    for img in os.listdir(path):
        image = cv2.imread(os.path.join(path, img))
        if image is not None:
            images.append(image)
            class_names.append(os.path.splitext(img)[0])
    return images, class_names

# Function to recognize faces and handle unknowns
def recognize_faces(known_encodes, class_names, frame):
    scale = 0.25
    box_multiplier = 1 / scale

    current_image = cv2.resize(frame, (0, 0), None, scale, scale)
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(current_image, model='cnn')
    face_encodes = face_recognition.face_encodings(current_image, face_locations)

    for encode_face, face_location in zip(face_encodes, face_locations):
        matches = face_recognition.compare_faces(known_encodes, encode_face, tolerance=0.6)
        face_dis = face_recognition.face_distance(known_encodes, encode_face)
        match_index = np.argmin(face_dis) if face_dis.size > 0 else None

        if match_index is not None and matches[match_index]:
            name = class_names[match_index].upper()
        else:
            name = 'Unknown'
            print("Unknown Face Detected")
            save_unknown_face(frame, face_location)  # Save unknown face

        y1, x2, y2, x1 = face_location
        y1, x2, y2, x1 = int(y1 * box_multiplier), int(x2 * box_multiplier), int(y2 * box_multiplier), int(x1 * box_multiplier)

        color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), color, cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

# Function to save unknown faces
def save_unknown_face(frame, face_location):
    if not os.path.exists('faces/unknowns'):
        os.makedirs('faces/unknowns')

    y1, x2, y2, x1 = face_location
    unknown_face = frame[y1:y2, x1:x2]
    filename = os.path.join('faces/unknowns', f"unknown_{int(time.time())}.jpg")
    cv2.imwrite(filename, unknown_face)
    print(f"Unknown face saved as {filename}")

# Function to test face recognition
def test_face_recognition():
    path = 'faces/known'
    images, class_names = load_training_images(path)
    known_encodes = find_encodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        recognize_faces(known_encodes, class_names, frame)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            print("Exiting face recognition...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Start face recognition
test_face_recognition() 