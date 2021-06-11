import cv2

class TrackerL():
    def __init__(self, frame, bounding_box):
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.tracker.init(frame, bounding_box)
  
        

    def track(self, frame):
        (success, box) = self.tracker.update(frame)

        if success:
            (x,y,w,h)  = [int(v) for v in box]
            print('TRACKED:',(x,y,w,h) )
            return (x,y,w,h)