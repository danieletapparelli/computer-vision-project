import cv2

class Tracker():
    def __init__(self, frame, bounding_box):

        #instantiate the typology of tracker
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.tracker.init(frame, bounding_box)
      

    def track(self, frame):

        #track and return updated bounding box
        (success, box) = self.tracker.update(frame)
        if success:
            (x,y,w,h)  = [int(v) for v in box]
            return (x,y,w,h)