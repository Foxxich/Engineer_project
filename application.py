import os
import time
import tkinter as tk
from functools import partial
from tkinter import *
import PIL.ImageTk
import cv2
from PIL import Image
from PIL import ImageTk
from sift import comparison


class LoggedWindow:
    def on_closing(self):
        self.master.destroy()
        self.main_window.destroy()

    def __init__(self, master, main_window, is_successful):
        self.master = master
        self.main_window = main_window
        self.frame = tk.Frame(self.master, width=300, height=300)
        master.title('You are logged')

        if is_successful:
            img = Image.open('logged.png')
        else:
            img = Image.open('error.png')
        self.tk_image = ImageTk.PhotoImage(img)
        Label(self.master, image=self.tk_image).place(x=0, y=0, relwidth=1, relheight=1)

        master.protocol("WM_DELETE_WINDOW", self.on_closing)


def run_tests(self, main_window):
    img1 = 'previous_image.jpg'
    img2 = 'new_image.jpg'
    start_time = time.time()
    result = comparison(img1, img2)
    end_time = time.time()
    print("Total time: ", round((end_time - start_time)), ' Seconds')
    if result:
        self.newWindow = tk.Toplevel(self.master)
        self.app = LoggedWindow(self.newWindow, main_window, True)
    else:
        self.newWindow = tk.Toplevel(self.master)
        self.app = LoggedWindow(self.newWindow, main_window, False)


def show_logged(self, main_window):
    self.master.withdraw()
    self.newWindow = tk.Toplevel(self.master)
    self.app = LoggedWindow(self.newWindow, main_window, True)


class App:
    def __init__(self, window, window_title, main_window, testing):
        self.app = None
        self.newWindow = None
        self.photo = None
        self.window = window
        self.testing = testing
        self.main_window = main_window
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
                if os.path.isfile('previous_image.jpg'):
                    os.remove('previous_image.jpg')
                os.rename('new_image.jpg', 'previous_image.jpg')
            except FileNotFoundError:
                print("File not exist")
            cv2.imwrite("new_image.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.vid.destroy()
            self.window.destroy()
            cv2.destroyAllWindows()
            if self.testing:
                run_tests(self.window, self.main_window)
            else:
                show_logged(self.window, self.main_window)

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
        self.out = None
        self.vid = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            raise ValueError("Unable to open camera", video_source)

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        dimensions = {
            '480p': (640, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
        }
        chosen_dimension = dimensions['480p']
        self.vid.set(3, chosen_dimension[0])
        self.vid.set(4, chosen_dimension[1])
        self.width, self.height = chosen_dimension

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

    def validate_login(self, username):
        print("username entered :", username.get())
        self.newWindow = tk.Toplevel(self.main_window)
        self.app = App(self.newWindow, 'Take image to login', self.main_window, False)
        self.master.destroy()
        return

    def __init__(self, master, main_window):
        self.app = None
        self.newWindow = None
        self.master = master
        self.main_window = main_window
        self.frame = tk.Frame(self.master)
        master.title('Create account')

        Label(master, text="User Name").grid(row=0, column=0)
        username = StringVar()
        Entry(master, textvariable=username).grid(row=0, column=1)

        validate_login = partial(self.validate_login, username)

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
        self.master.withdraw()
        self.newWindow = tk.Toplevel(self.master)
        self.app = App(self.newWindow, 'Take image to login', self.master, True)

    def register_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = RegisterWindow(self.newWindow, self.master)


def main():
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()

    # TODO: add https://stackoverflow.com/questions/14910858/how-to-specify-where-a-tkinter-window-opens
