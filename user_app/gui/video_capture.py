import cv2


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
