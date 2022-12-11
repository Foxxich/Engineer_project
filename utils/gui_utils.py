from tkinter import messagebox


def center_window(root, width=300, height=200):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))


def close_algorithm_window(parameters, main_window, window, results, algorithm):
    if algorithm == 'sift' and len(parameters) == 2:
        results.set_sift(parameters)
        window.destroy()
        main_window.deiconify()
    elif algorithm == 'vgg' and len(parameters) == 2:
        results.set_vgg(parameters)
        window.destroy()
        main_window.deiconify()
    elif algorithm == 'pca' and len(parameters) == 1:
        results.set_pca(parameters)
        window.destroy()
        main_window.deiconify()
    elif algorithm == 'cnn' and len(parameters) == 5:
        results.set_cnn(parameters)
        window.destroy()
        main_window.deiconify()
    else:
        messagebox.showerror(title=None, message='Choose all the parameters!')
