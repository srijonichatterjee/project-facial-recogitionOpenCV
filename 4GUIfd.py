import cv2
import os
import numpy as np
import face_recognition
import tkinter as tk
from tkinter import Label, Button

# Load or create necessary directories
if not os.path.exists('faces'):
    os.makedirs('faces')

# Function to find encodings of images
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:  # Ensure encoding exists
            encode_list.append(encodes[0])
    return encode_list

# Function to load training images and encodings
def load_training_images(path):
    images, class_names = [], []
    for img in os.listdir(path):
        image = cv2.imread(os.path.join(path, img))
        if image is not None:
            images.append(image)
            class_names.append(os.path.splitext(img)[0])
    return images, class_names

# Function to capture and store new face
def capture_face():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    name = input("Enter name: ")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Capture Face - Press 'c' to Save, 'p' to Exit", frame)

        key = cv2.waitKey(1)
        if key == ord('c'):
            filename = os.path.join('faces', name + '.jpg')
            cv2.imwrite(filename, frame)
            print(f"Image Saved: {filename}")
        elif key == ord('p'):
            print("Exiting capture mode...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to recognize faces in real-time
def recognize_faces():
    path = 'faces'
    images, class_names = load_training_images(path)
    known_encodes = find_encodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        # Resize and convert image to RGB
        small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodes = face_recognition.face_encodings(rgb_frame, face_locations)

        for encode_face, face_location in zip(face_encodes, face_locations):
            matches = face_recognition.compare_faces(known_encodes, encode_face, tolerance=0.6)
            face_dis = face_recognition.face_distance(known_encodes, encode_face)
            match_index = np.argmin(face_dis) if face_dis.size > 0 else None

            if match_index is not None and matches[match_index]:
                name = class_names[match_index].upper()
            else:
                name = 'Unknown'

            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y2 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Face Recognition - Press 'p' to Exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            print("Exiting face recognition...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to properly close the application
def exit_app():
    print("Exiting Application...")
    root.quit()  # Stops the Tkinter loop
    root.destroy()  # Closes the Tkinter window
    cv2.destroyAllWindows()  # Ensures all OpenCV windows are closed
    os._exit(0)

# Creating GUI with Tkinter
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("400x300")

Label(root, text="Face Recognition System", font=("Arial", 16)).pack(pady=20)
Button(root, text="Capture Face", font=("Arial", 12), command=capture_face).pack(pady=10)
Button(root, text="Recognize Faces", font=("Arial", 12), command=recognize_faces).pack(pady=10)
Button(root, text="Exit", font=("Arial", 12), bg="red", fg="white", command=exit_app).pack(pady=20)

root.mainloop()