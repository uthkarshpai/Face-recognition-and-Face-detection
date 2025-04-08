import cv2
import face_recognition
import os
import numpy as np

# Path to known faces
images_path = r"path\to\your\images\"

# Store face encodings and names
known_face_encodings = []
known_face_names = []

# Load known images
for file in os.listdir(images_path):
    if file.endswith((".jpg", ".png", ".jpeg")):
        file_path = os.path.join(images_path, file)

        # Load image
        image = face_recognition.load_image_file(file_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:  # If a face is found
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(file)[0])  # Use filename as label
        else:
            print(f"Skipping {file}: No face detected.")

print("Loaded known faces:", known_face_names)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame from BGR to RGB (required by face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        # Use the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances.size else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display name
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Face Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
