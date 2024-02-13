import tkinter
import customtkinter as ctk
import numpy
import cv2
import os
import camera, facedetection
import threading, multiprocessing as mp


from PIL import Image
from names import names


IMAGE_WIDTH, IMAGE_HEIGHT = 640,480
image_location = names["baseImageLOC"]


class PyGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode(names["appearance"])

        self.title(names["title"])
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{700}x{650}")
        self.minsize(700,650)
        self.maxsize(700,650)
        self.baseImage = Image.open(image_location)
        self.threadLaunch = threading.Thread(target=self.startCAM)
        self.threadD = threading.Thread(target=self.startDetect)
        self.detect = facedetection.FaceDetection()
        self.cam = camera.Camera()
        self.det_status = False


        self.image = ctk.CTkImage(light_image=self.baseImage, size=(IMAGE_WIDTH , IMAGE_HEIGHT))
        self.label = ctk.CTkLabel(master=self,
                                  image=self.image,
                                  text="")
        self.label.place(x=30,y=10)


        self.buttonLaunch = ctk.CTkButton(master=self,
                                          text=names["START"],
                                          command=self.threadLaunch.start,
                                          height=60,
                                          width=300,
                                          fg_color=("black"),
                                          hover_color="green")
        self.buttonLaunch.place(x=30,y=500)

        self.buttonStop = ctk.CTkButton(master=self,
                                        text=names["STOP"],
                                        command=self.stopCAM,
                                        height=60,
                                        width=300,
                                        fg_color=("black"),
                                        hover_color="red",
                                        state="disabled")
        self.buttonStop.place(x=370,y=500)

        self.detectLaunch = ctk.CTkButton(master=self,
                                          text=names["STARTD"],
                                          command=self.threadD.start,
                                          height=60,
                                          width=300,
                                          fg_color=("black"),
                                          hover_color="green",
                                          state="disabled")
        self.detectLaunch.place(x=30,y=570)

        self.detectStop = ctk.CTkButton(master=self,
                                        text=names["STOPD"],
                                        command=self.stopDetect,
                                        height=60,
                                        width=300,
                                        fg_color=("black"),
                                        hover_color="red",
                                        state="disabled")
        self.detectStop.place(x=370,y=570)


    def run(self):
        self.mainloop()

    def startCAM(self):
        self.buttonLaunch.configure(state="disabled")
        self.buttonStop.configure(state="normal")
        self.detectLaunch.configure(state="normal")
        self.cam.CAMERA_ON = True
        for i in self.cam.run():
            if (self.cam.CAMERA_ON) and (not self.det_status):
                self.image.configure(light_image=Image.fromarray(i[:, :, ::-1]))

    def stopCAM(self):
        self.buttonLaunch.configure(state="normal")
        self.buttonStop.configure(state="disabled")
        self.detectLaunch.configure(state="disabled")
        self.detectStop.configure(state="disabled")
        self.cam.CAMERA_ON = False
        self.det_status = False
        self.image.configure(light_image=self.baseImage)
        self.threadLaunch = threading.Thread(target=self.startCAM)
        self.buttonLaunch.configure(command=self.threadLaunch.start)
        self.threadD = threading.Thread(target=self.startDetect)
        self.detectLaunch.configure(command=self.threadD.start)

    def startDetect(self):
        self.detectStop.configure(state="normal")
        self.detectLaunch.configure(state="disabled")
        self.det_status = True

        for i in self.cam.run():
            if (self.cam.CAMERA_ON) and (self.det_status):
                f_image = self.detect.single_run(i)
                self.image.configure(light_image=Image.fromarray(f_image[:, :, ::-1]))


    def stopDetect(self):
        self.detectStop.configure(state="disabled")
        self.detectLaunch.configure(state="normal")
        self.threadD = threading.Thread(target=self.startDetect)
        self.detectLaunch.configure(command=self.threadD.start)
        self.det_status = False









if __name__ == "__main__":
    app = PyGUI()
    app.run()
