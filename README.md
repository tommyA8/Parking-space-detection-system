# Parking space detection system
A system for smart parking spaces using an object detection model that can count the duration a car has been parked

Approaches:
- Training YOLOv8s with [parking slot detecter](https://universe.roboflow.com/car-parking-space/parking-spot-detector-a84ql) dataset
- Developing a multi-object tracking system and counting the time

## Training
- 2 Classes: `['empty', 'occupied']`
- Hyperparameter: `Epochs = 50` `optimizer = AdamW` `image size = 640x640` and `batch size = 32`
### Training result
- Training result
<p align="left">
<img src="https://github.com/tommyA8/Parking-space-detection-system/blob/main/yolov8_custom_model/runs/detect/train/results.png?raw=true" width="800" height="450"/>
<p align="center">

- Validation Confusion Matrix Normalized
<p align="center">
<img src="https://github.com/tommyA8/Parking-space-detection-system/blob/main/yolov8_custom_model/runs/detect/val/confusion_matrix_normalized.png?raw=true" width="700" height="500"/>

## Final Result
**green bounding boxes** show each tracked car along with its ID number.

**Red bounding boxes** indicating empty spaces have not been tracked.

- video ouput 1 [source](https://github.com/tommyA8/Parking-space-detection-system/blob/main/outputs/OUTPUT_BLK-HDPTZ12%20Security%20Camera%20Parkng%20Lot%20Surveillance%20Video.mp4)

[![source_img](https://github.com/tommyA8/Parking-space-detection-system/blob/main/images/Parking_space_detection_1.jpeg?raw=true)](https://github.com/tommyA8/Parking-space-detection-system/blob/main/outputs/OUTPUT_BLK-HDPTZ12%20Security%20Camera%20Parkng%20Lot%20Surveillance%20Video.mp4)

- video ouput 2 [source](https://github.com/tommyA8/Parking-space-detection-system/blob/main/outputs/OUTPUT_BLK-HDPTZ12%20Security%20Camera%20Parkng%20Lot%20Surveillance%20Video.mp4)

[![source_img](https://github.com/tommyA8/Parking-space-detection-system/blob/main/images/testImg.jpeg?raw=true)](https://github.com/tommyA8/Parking-space-detection-system/blob/main/outputs/OUTPUT_BLK-HDPTZ12%20Security%20Camera%20Parkng%20Lot%20Surveillance%20Video.mp4)

# Citation
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
