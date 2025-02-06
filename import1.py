import cv2

# Take input of the person's name
name = input("Enter name: ")

# Create the VideoCapture object
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read each frame
    success, frame = cap.read()
    
    if not success:
        print("Error: Could not read frame.")
        break

    # Display the frame with a meaningful title
    cv2.imshow(f"Live Feed - {name}", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
