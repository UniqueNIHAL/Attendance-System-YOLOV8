import cv2
import os

def capture_images(student_name, num_images=5):
    cap = cv2.VideoCapture(0)
    os.makedirs(f'dataset/{student_name}', exist_ok=True)
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        cv2.imshow('Capture Images - Press "Space" to capture', frame)

        # Capture image on spacebar press
        if cv2.waitKey(1) & 0xFF == ord(' '):
            img_path = f'dataset/{student_name}/{student_name}_{count}.jpg'
            cv2.imwrite(img_path, frame)
            print(f"Image saved: {img_path}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()

    if count < num_images:
        print(f"Only {count}/{num_images} images captured. Please try again.")

# Run the function
student_name = input("Enter student's name: ")
capture_images(student_name)