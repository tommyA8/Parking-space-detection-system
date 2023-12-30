from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import datetime
import pandas as pd
from collections import defaultdict

# Load a model
model_path = "./yolov8_custom_model/runs/detect/train/weights/best.pt"
model = YOLO(model_path)  # load a custom model
model.to('cuda' if torch.cuda.is_available() else 'cpu') 

# Load a video
video_name = 'test'#'BLK-HDPTZ12 Security Camera Parkng Lot Surveillance Video'
video_path = f'./videos/{video_name}.mp4'
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# export a video
output_path = f'./outputs/OUTPUT_{video_name}.mp4'
cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (640, 640))

# Create an empty DataFrame
parking_df = pd.DataFrame(columns=['ID', 'Parked_at', 'Left_at', 'Duration'])

def plot_occupied_boxes(box):
    x, y, X, Y = box
    x, y, X, Y = int(x), int(y), int(X), int(Y)
    w = X-x
    h = Y-y
    text = f"{track_id}" #f"{track_id}{str(results.names[class_ids])}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
    text_w, text_h = text_size
    cv2.rectangle(frame, (x, y), (x+text_w+2, y-text_h-3), (0,0,255), -1)
    cv2.putText(frame, text, (x+1, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    cv2.rectangle(frame, (x, y), (w, h), (0,0,255), 1)

def calculate_parked_duration(Parked_at):
    parked_datetime = datetime.datetime.fromtimestamp(Parked_at)
    current_datetime = datetime.datetime.now()
    duration = current_datetime - parked_datetime
    return duration, current_datetime

# Function to record parking information in the DataFrame
def record_parking_info(parked_at, left_at, id):
    duration = left_at - parked_at
    # Append a new row with parking information to the DataFrame
    parking_df.loc[len(parking_df)] = [id, parked_at, left_at, duration]

# detection confidence
threshold = 0.3

track_history = defaultdict(lambda:[])
tracker = DeepSort()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640,640))

    results = model(frame, verbose=False)[0]

    detectionByFrame = []
    for result in results.boxes.data.tolist():

        x, y, w, h, score, class_ids = result
        x, y, w, h = int(x), int(y), int(w), int(h)
        class_ids = int(class_ids)

        if score > threshold: # names: {0: 'empty', 1: 'occupied'}
            if class_ids == 1: # detect space-occupied
                detectionByFrame.append(([x, y, w, h], score, str(results.names[class_ids])))
            else:
                #plot_empty_boxes()
                text_size, _ = cv2.getTextSize(str(results.names[class_ids]), cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
                text_w, text_h = text_size
                cv2.rectangle(frame, (x, y), (x+text_w+2, y-text_h-3), (0,255,0), -1)
                cv2.putText(frame, str(results.names[class_ids]), (x+1, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                cv2.rectangle(frame, (x, y), (w, h), (0,255,0), 1)

    # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
    tracks = tracker.update_tracks(detectionByFrame, frame=frame)

    if track_history:
        missing_id = [id for id in track_history if id not in [track.track_id for track in tracks]]
        # remove tracking id
        for id in missing_id:
            removed_value = track_history.pop(id)
            Parked_at = removed_value[0][-1]['Parked_at']
            Left_at = datetime.datetime.now()
            record_parking_info(Parked_at, Left_at, id)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb(orig=True)

        track = track_history[track_id]
        track.append([{'Parked_at': datetime.datetime.now()}])
        plot_occupied_boxes(ltrb)

    cv2.imshow("Parking space detection", frame)
    cap_out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cap_out.release()
cv2.destroyAllWindows()

# # show DataFrame (Output)
# df = pd.DataFrame(data)
# # print(df.info)
parking_df.to_csv(f'./outputs/OUTPUT_TABLE_{video_name}.csv', index=False)