import os
import time
import tkinter as tk
from functools import partial
from tkinter import *
from sift import comparison
import PIL.Image
import PIL.ImageTk
import cv2

is_login_possible = False


def run_tests(self):
    img1 = 'extra_frame.jpg'
    img2 = 'last_frame.jpg'
    start_time = time.time()
    result = comparison(img1, img2)
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')
    if result:
        self.newWindow = tk.Toplevel(self.master)
        self.app = App(self.newWindow, 'Take image to login')



class App:
    def __init__(self, window, window_title):
        self.photo = None
        self.window = window
        self.window.title(window_title)
        self.video_source = 0
        self.ok = False

        self.vid = VideoCapture(self.video_source)
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()
        self.btn_snapshot = tk.Button(window, text="Make photo", command=self.make_photo)
        self.btn_snapshot.pack(side=tk.LEFT)
        self.delay = 10
        self.update_frame()

    def make_photo(self):
        ret, frame = self.vid.get_frame()
        if ret:
            try:
                if os.path.isfile('extra_frame.jpg'):
                    os.remove('extra_frame.jpg')
                os.rename('last_frame.jpg', 'extra_frame.jpg')
            except FileNotFoundError:
                print("File not exist")
            cv2.imwrite("last_frame.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.vid.destroy()
            self.window.destroy()
            cv2.destroyAllWindows()
            run_tests(self.window)

    def update_frame(self):
        ret, frame = self.vid.get_frame()
        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update_frame)


class VideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            raise ValueError("Unable to open camera", video_source)

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        dimensions = {
            '480p': (640, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
        }
        res = dimensions['480p']
        self.vid.set(3, res[0])
        self.vid.set(4, res[1])
        self.width, self.height = res

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                return ret, None
        else:
            return None

    def destroy(self):
        self.vid.release()
        cv2.destroyAllWindows()


class RegisterWindow:

    @staticmethod
    def validate_login(self, username, password):
        print("username entered :", username.get())
        print("password entered :", password.get())
        self.newWindow = tk.Toplevel(self.master)
        self.app = App(self.newWindow, 'Take image to login')
        return

    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        master.title('Create account')

        Label(master, text="User Name").grid(row=0, column=0)
        username = StringVar()
        Entry(master, textvariable=username).grid(row=0, column=1)

        Label(master, text="Password").grid(row=1, column=0)
        password = StringVar()
        Entry(master, textvariable=password, show='*').grid(row=1, column=1)

        validate_login = partial(self.validate_login, master, username, password)

        # login button
        Button(master, text="Login", command=validate_login).grid(row=4, column=0)

    def close_windows(self):
        self.master.destroy()


class MainWindow:
    def __init__(self, master):
        self.app = None
        self.newWindow = None
        self.master = master
        self.frame = tk.Frame(self.master, width=300, height=300)
        self.frame.size()
        self.button1 = tk.Button(self.frame, text='Login', width=125, command=self.login_window)
        self.button1.pack()
        self.button2 = tk.Button(self.frame, text='Register', width=125, command=self.register_window)
        self.button2.pack()
        self.frame.pack()

    def login_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = App(self.newWindow, 'Take image to login')

    def register_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = RegisterWindow(self.newWindow)


def main():
    root = tk.Tk()
    root.geometry("300x300")
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
