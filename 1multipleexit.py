import cv2
import numpy as np
import os
import face_recognition

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
path=r'C:\Users\KIIT\Desktop\srijoni project RT\faces\known'
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

# Function to recognize faces in the video feed
def recognize_faces(known_encodes, class_names, frame):
    scale = 0.25
    box_multiplier = 1 / scale

    current_image = cv2.resize(frame, (0, 0), None, scale, scale)
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

    # Find the face locations and encodings for the current frame
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

        y1, x2, y2, x1 = face_location
        y1, x2, y2, x1 = int(y1 * box_multiplier), int(x2 * box_multiplier), int(y2 * box_multiplier), int(x1 * box_multiplier)

        # Draw rectangle around detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

# Function to test face recognition
def test_face_recognition():
    # Load training images and find their encodings
    path = r'faces\known'
    images, class_names = load_training_images(path)
    known_encodes = find_encodings(images)
    print('Encoding Complete')

    # Create the VideoCapture object
    cap = cv2.VideoCapture(0)

    while True:
        # Read each frame
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        # Recognize faces in the frame
        recognize_faces(known_encodes, class_names, frame)

        # Show the output
        cv2.imshow("Face Recognition", frame)

        # Exit loop when 'p' is pressed
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print("Exiting face recognition...")
            break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()

# Get the person's name
name = input("Enter name: ")

# Create directory if not exists
if not os.path.exists('faces'):
    os.makedirs('faces')

# Create the VideoCapture object
cap = cv2.VideoCapture(0)

while True:
    # Read each frame
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    # Show the output
    cv2.imshow("Capture Face", frame)

    # If 'c' key is pressed then click picture
    if cv2.waitKey(1) == ord('c'):
        filename = os.path.join('faces', name + '.jpg')
        cv2.imwrite(filename, frame)
        print("Image Saved -", filename)

    # Exit loop when 'p' is pressed
    if cv2.waitKey(1) & 0xFF == ord('p'):
        print("Exiting capture mode...")
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()

# Test face recognition
test_face_recognition()