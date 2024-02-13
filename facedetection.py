import face_recognition
import cv2
import os
import numpy as np

from names import names

class FaceDetection():
    def __init__(self):
        #self.camera = cv2.VideoCapture(0)
        self.CAMERA_ON = True
        self.known_face = []
        self.known_name = []
        self.encode_face()


    def encode_face(self):
        for im in os.listdir(names["knownImage"]):
            face = face_recognition.load_image_file(os.path.join(names["knownImage"],im))
            face_encode = face_recognition.face_encodings(face)[0]
            self.known_face.append(face_encode)
            self.known_name.append(im)


    def run(self):
        while self.CAMERA_ON:
            ret, frame = self.camera.read()
            small_frame = cv2.resize(frame, (0,0), fx=0.25,fy=0.25)
            small_frame = small_frame[:,:,::-1]

            face_location = face_recognition.face_locations(small_frame)
            face_encoding = face_recognition.face_encodings(small_frame,face_location)

            for face,loc in zip(face_encoding,face_location):
                name = "Unknown"
                matches = face_recognition.compare_faces(self.known_face,face)
                face_similarity = face_recognition.face_distance(self.known_face,face)
                best = np.argmin(face_similarity)
                confidence = 1 - face_similarity[best]

                if matches[best] and (confidence > 0.5):
                    name = self.known_name[best]

                for top,right,bottom,left in np.expand_dims(loc,0):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
                    cv2.rectangle(frame,(left,bottom),(right-60,bottom+15),(0,0,255),-1)
                    cv2.putText(frame,name, (left+3,bottom+10),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255))

                    if confidence > 0.5:
                        cv2.putText(frame,str(confidence), (left+3,bottom+20),cv2.FONT_HERSHEY_DUPLEX,0.3,(255,255,255))



            cv2.imshow("Frame",frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def single_run(self,frame):
        small_frame = cv2.resize(frame, (0,0), fx=0.25,fy=0.25)
        small_frame = small_frame[:,:,::-1]

        face_location = face_recognition.face_locations(small_frame)
        face_encoding = face_recognition.face_encodings(small_frame,face_location)

        for face,loc in zip(face_encoding,face_location):
            name = "Unknown"
            matches = face_recognition.compare_faces(self.known_face,face)
            face_similarity = face_recognition.face_distance(self.known_face,face)
            best = np.argmin(face_similarity)
            confidence = 1 - face_similarity[best]

            if matches[best] and (confidence > 0.5):
                name = self.known_name[best]

            for top,right,bottom,left in np.expand_dims(loc,0):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
                cv2.rectangle(frame,(left,bottom),(right-60,bottom+15),(0,0,255),-1)
                cv2.putText(frame,name, (left+3,bottom+10),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255))

                if confidence > 0.5:
                    cv2.putText(frame,str(confidence), (left+3,bottom+20),cv2.FONT_HERSHEY_DUPLEX,0.3,(255,255,255))

        return frame







if __name__ == "__main__":
    cam = FaceDetection()
    cam.run()
