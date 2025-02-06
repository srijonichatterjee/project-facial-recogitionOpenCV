import cv2
import os

# Path to the image for fallback
image_path = r"C:\Users\KIIT\Desktop\rt faces\agnis.jpg"
person_name = "Agniswar Chatterjee"  # Replace with the actual name of the person in the image

# Function to check if the camera is working
def check_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not detected.")
        return False
    else:
        print("Camera detected successfully.")
        cap.release()
        return True

# Main program
if check_camera():
    # Camera is working, capture live feed
    print("Accessing camera feed...")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        # Load Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Live Camera Feed", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    # Camera not working, fallback to image processing
    if not os.path.exists(image_path):
        print("Error: Image file does not exist.")
        exit()

    print("Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        exit()
    else:
        print("Image loaded successfully.")

    print("Loading Haar Cascade...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Haar cascade file not loaded.")
        exit()

    print("Converting to grayscale...")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Detecting faces...")
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display the name on the image
        cv2.putText(image, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()