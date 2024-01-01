from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os
from collections import defaultdict
from datetime import datetime
from pandas import DataFrame

class Detection_model():
    def __init__ (self, model_path:str):
        self.model = YOLO(model_path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu') 

    def detect(self, frame):
        '''
        return  list of x, y, w, h, score, class_ids and 
                dictionary of class names {0: 'empty', 1: 'occupied'}
        '''
        results = self.model(frame, verbose=False)[0]
        return results.boxes.data.tolist(), results.names # names: {0: 'empty', 1: 'occupied'}

class ParkingManager():
    def __init__(self, detector:object, 
                 tracker:object, 
                 video_path=None,
                 threshold:float= 0.3,
                 resize:tuple=None,
                 save_video:bool=False,
                 save_table:bool=False):
        
        self.detector = detector
        self.tracker = tracker
        self.cap = cv2.VideoCapture(video_path)
        assert self.cap.isOpened(), "Error reading video file"
        self.threshold = threshold
        self.resize = resize
        self.save_video = save_video
        self.save_table = save_table

    def export_video(self, size:tuple=None, fps:int=30):
        assert size, "Required output video size"
        output_path = './outputs/OUTPUT.mp4'
        return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    def plot_boxes(self, frame, bbox, class_name=None, track_id=None):
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        if track_id:
            w, h = w-x, h-y
            text = f"{track_id}"
            color = (0,0,255)
        else:
            text = class_name
            color = (0,255,0)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
        text_w, text_h = text_size
        cv2.rectangle(frame, (x, y), (x+text_w+2, y-text_h-3), color, -1)
        cv2.putText(frame, text, (x+1, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        cv2.rectangle(frame, (x, y), (w, h), color, 1)
    
    def run(self):

        if self.save_table:
            parking_df = DataFrame(columns=['ID', 'Parked_at', 'Left_at', 'Duration'])

        if self.save_video:
            if self.resize:
                cap_out = self.export_video(size=self.resize)
            else:
                width  = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
                height = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
                cap_out = self.export_video(size=(width, height))

        track_history = defaultdict(lambda:[])

        while 1:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.resize:
                frame = cv2.resize(frame, (640,640))

            results, class_names = self.detector.detect(frame)
            
            occupiedDetected = []
            for result in results:
                x, y, w, h, score, class_ids = result
                class_name = class_names[int(class_ids)]

                if score > self.threshold:
                    if class_ids==1:
                        occupiedDetected.append(([x, y, w, h], score, class_name))
                    else:
                        self.plot_boxes(frame, [x, y, w, h], class_name)
                
            tracks = self.tracker.update_tracks(occupiedDetected, frame=frame)

            if track_history:
                removed_IDs = [track_history.pop(id) for id in track_history if id not in [track.track_id for track in tracks]]
                if self.save_table:
                    for id in removed_IDs:
                        parked_at = id['Parked_at']
                        left_at = datetime.now()
                        duration = left_at - parked_at
                        parking_df.loc[len(parking_df)] = [id, parked_at, left_at, duration]
                                 
            for track in tracks:
                if not track.is_confirmed():
                    continue
                ltrb = track.to_ltrb(orig=True)
                track_id = track.track_id
                # store tracking id
                if track_id in track_history:
                    track = track_history[track_id]
                    track.append({'Parked_at': datetime.now()})
                # plotting
                self.plot_boxes(frame, ltrb, track_id=track_id)

            cv2.imshow("Parking System", frame)

            if self.save_video:
                cap_out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if self.save_video:
            cap_out.release()
        self.cap.release()
        cv2.destroyAllWindows()

        if self.save_table:
            parking_df.to_csv('./outputs/OUTPUT_TABLE.csv', index=False)

if __name__ == "__main__":
    import timeit

    model_path = "./detection_model/train3/weights/best.pt"
    video_path = "./videos/test.mp4"
    
    usage = ParkingManager( detector=Detection_model(model_path=model_path),
                            tracker=DeepSort(),
                            video_path=video_path,
                            threshold=0.3,
                            resize=(640,640),
                            save_video=True,
                            save_table=True )
    #usage.run()

    print("Benchmark", timeit.timeit(lambda: usage.run(), setup="pass",number=1))