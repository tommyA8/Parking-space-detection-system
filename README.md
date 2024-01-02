# **Parking space detection system**
A system for smart parking spaces using an object detection model that can count the duration a car has been parked

Approaches:
- Training YOLOv8 with [parking slot detecter](https://universe.roboflow.com/car-parking-space/parking-spot-detector-a84ql) dataset for Object detection
- Using [DeepSORT](https://github.com/levan92/deep_sort_realtime) for Object Tracking
- Developing a multi-object tracking system and counting the time

## **Training**
- 2 Classes: `['empty', 'occupied']`
- Hyperparameter: `Epochs=200` `optimizer=AdamW` `image size=640x640` and `batch size=64`
### Training result

- Validation Confusion Matrix Normalized

Empty-Class            |  Occupied-Class
:-------------------------:|:-------------------------:
Correct detection made or `True-Positive = 96 %` | Correct detection made or `True-Positive = 97 %`
Incorrect detection made or `False-Positive = 70 %`|Incorrect detection made or `False-Positive = 30 %`
A Ground-truth missed (not detected) or `False-Negative = 4 %`|A Ground-truth missed (not detected) or `False-Negative = 3 %`
<p align="center">
<img src="https://github.com/tommyA8/Parking-space-detection-system/blob/main/detection_model/val3/confusion_matrix_normalized.png?raw=true" width="700" height="500"/>

----

`Precision-Recall Currve` 
: Achieved a score of 97.6% at the IOU threshold of 0.5.
<p align="center">
<img src="https://github.com/tommyA8/Parking-space-detection-system/blob/main/detection_model/val3/PR_curve.png?raw=true" width="800" height="450"/>
<p align="center">

## **Sample Results**
- **green bounding boxes** show each tracked car along with its ID number.

- **Red bounding boxes** indicating empty spaces have not been tracked.

- Full Video Ouput 1 [source](https://github.com/tommyA8/Parking-space-detection-system/blob/main/outputs/OUTPUT_BLK-HDPTZ12%20Security%20Camera%20Parkng%20Lot%20Surveillance%20Video.mp4) | Full Video Ouput 2 [source](https://github.com/tommyA8/Parking-space-detection-system/blob/main/outputs/OUTPUT_BLK-HDPTZ12%20Security%20Camera%20Parkng%20Lot%20Surveillance%20Video.mp4)

VIDEO OUTPUT 1            |  VIDEO OUTPUT 2
:-------------------------:|:-------------------------:
[![source_img1](https://github.com/tommyA8/Parking-space-detection-system/blob/main/images/Parking_space_detection_1.jpeg?raw=true)](https://github.com/tommyA8/Parking-space-detection-system/blob/main/outputs/OUTPUT_BLK-HDPTZ12%20Security%20Camera%20Parkng%20Lot%20Surveillance%20Video.mp4)Figure 1  |  [![source_img2](https://github.com/tommyA8/Parking-space-detection-system/blob/main/images/testImg.jpeg?raw=true)](https://github.com/tommyA8/Parking-space-detection-system/blob/main/outputs/OUTPUT_BLK-HDPTZ12%20Security%20Camera%20Parkng%20Lot%20Surveillance%20Video.mp4)Figure 2

TABLE OUTPUT 1            |  TABLE OUTPUT 2
:-------------------------:|:-------------------------:
[![source_table1](https://github.com/tommyA8/Parking-space-detection-system/blob/main/images/table_output_orig_vdo.jpg?raw=true)](https://github.com/tommyA8/Parking-space-detection-system/blob/main/images/table_output_orig_vdo.jpg?raw=true)Figure 3 | [![source_table2](https://github.com/tommyA8/Parking-space-detection-system/blob/main/images/table_output_test_vdo.jpg?raw=true)](https://github.com/tommyA8/Parking-space-detection-system/blob/main/images/table_output_test_vdo.jpg?raw=true)Figure 4

## **Conclusion**
From OUTPUT VIDEO 1, it was observed that no vehicles exited, yet the system recorded that a vehicle had exited

Based on the evaluation of the trained Detection Model, satisfactory results have been observed. However, the system still encounters errors as you can see the OUTPUT VIDEO 1 that due to the Object Detection Model being trained on limited diverse data, such as variations in lighting, camera angles, obscured vehicles, weather conditions at the time, strong winds, or rainfall, among others.



# **Citation**
```python
@misc{ parking-spot-detector-a84ql_dataset,
    title = { parking spot detector Dataset },
    type = { Open Source Dataset },
    author = { car parking space },
    howpublished = { \url{ https://universe.roboflow.com/car-parking-space/parking-spot-detector-a84ql } },
    url = { https://universe.roboflow.com/car-parking-space/parking-spot-detector-a84ql },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2023 },
    month = { dec },
    note = { visited on 2023-12-27 },
}
```
