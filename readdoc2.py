import cv2

# Take input of the person name
name = input("Enter name:  ")

# Path to the saved image on the desktop
image_path = r"C:\Users\KIIT\Desktop\rt faces\human.jpg"

# Read the image from the specified path
print(f"Reading image from: {image_path}")
frame = cv2.imread(image_path)

# Check if the image was successfully loaded
if frame is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully.")
    # Show the output
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
