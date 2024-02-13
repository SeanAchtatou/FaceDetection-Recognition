import cv2
import time



class Camera():
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.CAMERA_ON = True

    def run(self):
        fps = 1/300
        start = time.time()
        while self.CAMERA_ON:
            ret, frame = self.camera.read()
            yield frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



if __name__ == "__main__":
    cam = Camera()
    cam.run()

