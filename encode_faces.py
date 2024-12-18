import face_recognition
import os
import pickle

def encode_faces(dataset_path):
    known_encodings = []
    known_names = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('jpg', 'png')):
                img_path = os.path.join(root, file)
                name = os.path.basename(root)

                # Load image and generate encodings
                try:
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
                        print(f"Encoding added for {name}: {img_path}")
                    else:
                        print(f"Face not found in {img_path}")

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    # Save encodings to file
    data = {'encodings': known_encodings, 'names': known_names}
    with open('face_encodings.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"Encodings saved to face_encodings.pkl with {len(known_encodings)} entries.")

# Update dataset path and run the function
dataset_path = 'dataset/'
encode_faces(dataset_path)
