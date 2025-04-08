import cv2
import os

def create_dataset(person_name, save_dir='dataset', num_images=50):
    # Create directory if it does not exist
    person_dir = os.path.join(save_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            img_path = os.path.join(person_dir, f'{count}.jpg')
            cv2.imwrite(img_path, face_img)
            count += 1
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        cv2.imshow('Face Capture', frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Dataset collection complete. Images saved in {person_dir}")

if __name__ == "__main__":
    name = input("Enter your name: ")
    create_dataset(name)
