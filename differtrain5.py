import cv2
import os

# Directory containing images for testing
image_dir = r"C:\Users\KIIT\Desktop\rt faces"

# Function to extract person/animal name from the file name
def extract_name_from_filename(filename):
    # Assuming the file name is formatted as "<name>.jpg"
    name, _ = os.path.splitext(filename)
    return name.replace("_", " ").title()  # Replace underscores with spaces and capitalize words

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

        # Load Haar Cascade for human face detection
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
    if not os.path.exists(image_dir):
        print("Error: Image directory does not exist.")
        exit()

    # Load Haar Cascades for human and animal face detection
    human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

    # Process each image in the directory
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)

        # Skip non-image files
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Extract person/animal name from the file name
        name = extract_name_from_filename(image_file)

        print(f"Processing image: {image_file} (Name: {name})")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_file}.")
            continue

        print("Converting to grayscale...")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        print("Detecting faces...")

        # Detect human faces
        human_faces = human_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Detect cat faces
        cat_faces = cat_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(human_faces) > 0:
            print(f"Detected {len(human_faces)} human face(s) in {image_file}")
            for (x, y, w, h) in human_faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image, f"Human: {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        elif len(cat_faces) > 0:
            print(f"Detected {len(cat_faces)} cat face(s) in {image_file}")
            for (x, y, w, h) in cat_faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, f"Cat: {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        else:
            print(f"No faces detected in {image_file}")
            cv2.putText(image, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the image
        cv2.imshow(f"Detected Faces - {name}", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
