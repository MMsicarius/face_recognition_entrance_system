import cv2
import os
import face_recognition
import numpy as np
import pandas as pd


class SimpleFacerec:
    def __init__(self):
        self.faces = []
        self.names = []
        self.access = []
        self.access_level = 0

        # Resize frame for a faster speed
        self.frame_resize_ratio = 0.25
        self.credentials = pd.read_csv("access_credentials.csv", index_col="Index")
        self.access_log = pd.read_csv("access_log.csv")
        # TODO credentials
        print(self.credentials.loc[self.credentials["Index"] == 1])

    def change_access_level(self):
        print("Please specify the access level of the entrance:")
        x = input()
        if x.isnumeric():
            self.access_level = x
        else:
            "Error! Invalid access value!"

    def load_encoding_images(self):
        for index, row in self.credentials.iterrows():
            self.names.append(row["Name"])
            initial_image = cv2.imread(("faces\\" + row["Image"]))
            rgb_img = cv2.cvtColor(initial_image, cv2.COLOR_BGR2RGB)
            image_encoding_endpoint = face_recognition.face_encodings(rgb_img)[0]
            self.faces.append(image_encoding_endpoint)
            self.access.append(row["Access Level"])
        print("Encoding faces loaded")

    def detect_known_faces(self, frame):
        # access states 0 = denied, 1 = too low access level, 2 = granted
        access_state = 0
        frame_resized = cv2.resize(frame, (0, 0), fx=self.frame_resize_ratio, fy=self.frame_resize_ratio)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        resized_frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(resized_frame_rgb)
        face_encodings = face_recognition.face_encodings(resized_frame_rgb, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.faces, face_encoding)

            """ uses the pre-existing face encoding and compares with the face in frame to get a euclidean distance
            ref -> https://face-recognition.readthedocs.io/en/latest/face_recognition.html"""
            face_distances = face_recognition.face_distance(self.faces, face_encoding)
            face_best_matching = np.argmin(face_distances)
            if matches[face_best_matching] and \
                    self.credentials.loc[f"{face_best_matching}", "Access Level"] >= self.access_level:
                name = self.names[face_best_matching]
                access_state = 2
            elif matches[face_best_matching] and \
                    self.credentials.loc[f"{face_best_matching}", "Access Level"] >= self.access_level:
                name = self.names[face_best_matching]
                access_state = 1
            else:
                name = "Unknown Face Detected!"
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resize_ratio
        return face_locations.astype(int), face_names, access_state
