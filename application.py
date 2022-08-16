import tkinter as tk

# program to capture single image from webcam in python

# importing OpenCV library
from tkinter import *

from cv2 import *

# camera
# from cv2 import VideoCapture, imshow, imwrite, waitKey, destroyWindow
#
# cam_port = 0
# cam = VideoCapture(cam_port)
#
# # reading the input using the camera
# result, image = cam.read()
#
# # If image will detected without any error,
# # show result
# if result:
#
#     # showing result, it take frame name and image
#     # output
#     imshow("GeeksForGeeks", image)
#
#     # saving image in local storage
#     imwrite("GeeksForGeeks.png", image)
#
#     # If keyboard interrupt occurs, destroy image
#     # window
#     waitKey(0)
#     destroyWindow("GeeksForGeeks")
#
# # If captured image is corrupted, moving to else part
# else:
#     print("No image detected. Please! try again")
def open_camera():
    pass


def login():
    open_camera()

def register():
    master = Tk()

    # this will create a label widget
    l1 = Label(master, text="U need to provide your image to login!!")

    # grid method to arrange labels in respective
    # rows and columns as specified
    l1.grid(row=0, column=0, sticky=W, pady=2)

    # entry widgets, used to take entry from user
    e1 = Entry(master)

    # this will arrange entry widgets
    e1.grid(row=0, column=1, pady=2)

    # checkbutton widget
    c1 = Checkbutton(master, text="Preserve")
    c1.grid(row=2, column=0, sticky=W, columnspan=2)

    # button widget
    b1 = Button(master, text="Zoom in")
    b2 = Button(master, text="Zoom out")

    # arranging button widgets
    b1.grid(row=2, column=2, sticky=E)
    b2.grid(row=2, column=3, sticky=E)


window = tk.Tk()

window.columnconfigure([0, 1, 2], minsize=111, weight=1)
window.rowconfigure([0, 1, 2], minsize=111, weight=1)

btn_decrease = tk.Button(master=window, text="login", command=login)
btn_decrease.grid(row=1, column=0, sticky=E, pady = 3)

btn_increase = tk.Button(master=window, text="register", command=register)
btn_increase.grid(row=1, column=2, sticky=W, pady = 3)

window.mainloop()