import cv2
import os
import face_recognition
import numpy as np
import glob


class SimpleFacerec:
    def __init__(self):
        self.faces = []
        self.names = []

        # Resize frame for a faster speed
        self.frame_resize_ratio = 0.25

    def load_encoding_images(self, face_image_path):
        """
        Load encoding images from path
        :param face_image_path:
        :return:
        """
        # Load Images
        face_image_path = glob.glob(os.path.join(face_image_path, "*.*"))

        print("{} encoding images found.".format(len(face_image_path)))

        # Store image encoding and names
        for i in face_image_path:
            initial_image = cv2.imread(i)
            rgb_img = cv2.cvtColor(initial_image, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(i)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            image_encoding_endpoint = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            self.faces.append(image_encoding_endpoint)
            self.names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resize_ratio, fy=self.frame_resize_ratio)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.faces, face_encoding)
            name = ""

            # # If a match was found in faces, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.names[best_match_index]
            else:
                name = "Unknown Face Detected!"
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resize_ratio
        return face_locations.astype(int), face_names
