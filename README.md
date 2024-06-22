# The project of Team9 : Parking Space Detection System🚗 #
## This git repository is for the Term Project final report for the "Embedded Systems and IoT Applications" course. ##

## Presentation Materials ##
### 🎬 Youtube Link :
https://www.youtube.com/watch?v=X4cBmc7w5Qk

### 📷 Datasets Source: 

  On Robowflow)
  
  https://universe.roboflow.com/swee-xiao-qi/parking-lot-availability/dataset/8
  https://universe.roboflow.com/smartparking-tijv5/smart_parking/dataset/2<br>

  Custom Data download)
  https://drive.google.com/file/d/1Kg26S8Iw1E9RF8NG9LkC9zqWM5_j7tZl/view?usp=sharing <br>


  

### 📕 PowerPoint Slides: 

  https://docs.google.com/presentation/d/1_BPLD-plQ3MDMkgVgukvJvForcohcgZq/edit?usp=sharing&ouid=117017025492141614372&rtpof=true&sd=true


### 🎥 Final Product Video: 
https://drive.google.com/file/d/1KNxn3nk9PoJdqOD3CRaO3LsSjmez3-lN/view?usp=sharing


## Table of Contents ##
1. Project Summary
2. Project Diagram
3. Prepare Data
4. Key Code Explanation
5. Project Installation and Usage Guide
6. Trial and Error
7. References

## Project Summary ##
💡 Project Motivation : 
Korea is currently experiencing many traffic issues related to cars. Especially in densely populated areas, it can be difficult for drivers to secure parking spaces. For example, at Dankook University, students and faculty members often struggle to find parking due to the lack of parking information.


📌 Project Goal : 
Using YOLOv8 to detect cars and empty parking spaces (in real-time) and provide parking information to users.

🎥  Final Output : 
A video providing the number of available parking spaces and occupied cars in a specific parking lot.

## Project Diagram

### Project Implementation Steps (Introduced in Presentation Recording)
<img src="https://github.com/jjw0513/Team9_Parking_detection/assets/151171066/156b63a2-0848-4e29-8e3b-b584d26da74c" width="400">

### Types of YOLOv8 Models
<img src="https://github.com/jjw0513/Team9_Parking_detection/assets/151171066/b351e520-2ab3-4f71-aac4-f725c6007f1a" width="400">

We chose YOLOv8s for the following reasons:
- Balanced Performance: YOLOv8s provides 44.9 mAP (mean Average Precision), offering high accuracy in object detection. It was an appropriate choice considering the scale of our project without being too heavy.
- Suitable Model Size: YOLOv8s has a size of 22.6 MB, making it a medium-sized model with reasonable memory usage. This was suitable for our team as we were limited to using only laptops.


### Data Flow and Model Structure)
<img width="500" alt="image" src="https://github.com/jjw0513/Team9_Parking_detection/assets/151171066/762186fc-6efc-43a7-afa1-0051f7ecce47">

#### - Input :
  Parking lot images are input. These images are used as the basic data for the system to detect parking spaces.

#### - Backbone : Pre-trained YOLOv8 model is loaded and used.
  Conv (Convolutional Layer): Features are extracted through multiple convolutional layers. These layers are used to extract low-level features from the images.

  C2f (C2f Block): This block is connected to specific convolution operations. Two C2f blocks are used to extract more complex features from the images.

  SPPF (Spatial Pyramid Pooling-Fast): Combines feature maps of various sizes to detect objects at different scales.

#### - Neck : 
  Located between the Backbone and Head layers.
  
  Concat (Concatenation): Combines features extracted from multiple layers. This helps integrate multi-scale information from the images.

  Upsample: Increases the size of the feature maps to match the original image size, restoring finer details.

  C2f (C2f Block): Used again for additional feature extraction.

  Conv (Convolutional Layer): Final convolution operation to extract the final features.

#### - Prediction :
  Detect: The model detects parking spaces and cars. Object detection occurs at this stage.


#### - Output :
  Detection results are displayed on the parking lot image. For example, empty parking spaces are labeled as "Empty" and occupied spaces as "Car," each marked with bounding boxes.



# Prepare Data
We used the following datasets appropriately.

#### 1. Open Datasets (Roboflow)
https://universe.roboflow.com/swee-xiao-qi/parking-lot-availability/dataset/8
https://universe.roboflow.com/smartparking-tijv5/smart_parking/dataset/2

#### 2. Custom Dataset (Dankook University Parking Lot)
<img width="400" alt="image" src="https://github.com/jjw0513/Team9_Parking_detection/assets/151171066/849129a0-023b-4a26-8778-102ef68edec8">

Approximately 200 captured photos.

Prepared training data using Roboflow's labeling program.


## Key Code Explanation
Let's explain the key code of the project

- **custom.yaml** :This YAML file is used as the configuration file for the object detection model. 
    It is mainly used to train or evaluate specific data in deep learning models like YOLO. It references the images directories of train, validation, and test present in the datasets.<br>


- **yolo_train.py** : Code to load and train a pre-trained YOLOv8 model for object detection.
    It sets the data path, fetches image and label files, and sets the ratio of training and validation data. It opens the yaml file to reference the class and data information. 
    It then loads the pre-trained model and performs training using model.train().


- **yolo_test** : Code to use the trained weights to perform predictions on test data.
    It loads the pre-trained YOLO model (final_w.pt) and performs predictions on the test data.


- **mp4_test.py** : Code to use the trained YOLO model to detect objects in a video file and save the results as a video file (for video testing). 
    It uses the cap library of cv2 to process and save video frames.

- **parking_app.py** : Code to implement the parking space detection system as an application.
    It uses the trained YOLO model to perform detection and save the results as a video.

    **ParkingPtsSelection()**: Loads Ultralytics' GUI program to specify the boundary coordinates of the parking spaces. The boundary coordinates are then saved as a JSON file and used for detection. 
    We used a frame from the target video to specify the coordinates and applied the coordinate information to the result video.

    **ParkingManagement()** : Loads the weights of the pre-trained model to perform object detection. It then applies the detection model to the application. The object uses the YOLO model to process object detection and parking applications.

    It reads each frame using 'cap', performs detection, and saves the results as a video file.<br>




- **parking_management.py** : Code to implement the parking space detection application.

    **ParkingPtsSelection class**: Provides a GUI window for users to select the boundary coordinates of the parking spaces.

    **ParkingManagement class**: Uses the JSON file and model weights to implement the application.

    **parking_regions_extraction**: Extracts parking space boundary coordinates from the JSON file.

    **process_data method**: Processes the detection results of the YOLO model to determine the occupancy status of the parking spaces. 
    It updates the occupancy status based on whether there is a vehicle within the boundary of each parking space. Finally, it displays the status of the parking spaces on the image.

    **display_frames method**: Displays the current occupancy information on the screen.


## Project Installation and Usage Guide
To use the project directly, follow these steps:

#### 1. Install libraries using requirements.txt

```bash
pip install -r requirements.txt
```

#### 2. Download the datasets directory

Download the directory attached at the top.

If there is a parking lot you want to detect, you can also add a photo and label information of that 
space.

Also, If you want to detect objects other than cars and parking spaces, modify the .yaml file.


#### 3. Perform training

Set the ratio of train and validation manually or use the datasets directory as it is.
```bash    
python yolo_train.py
```
If you want to check the training performance, you can create a Weights & Biases (wandb) account to view the graphs.

    
#### 4. Save the trained weights (final_w.pt)


#### 5. Conduct testing
Load the previously saved weight information.
```bash
python yolo_test.py
```

#### 6. Generate the coordinate information of the parking lot (as a .json file)
Prepare the video you want to detect and a frame from the video.
```bash
python parking_app.py
```

If you want to set coordinates on a parking lot photo, use the following object.
```bash
parking_selection = parking_management.ParkingPtsSelection()
```

#### 7. Apply the application to the target video.
Load the previously saved weight information.

If parking lot coordinates have been pre-specified, you can encode the video as follows after commenting out the coordinates.

```bash
#parking_selection = parking_management.ParkingPtsSelection()
```


## Trial and Error
We encountered 3 major difficulties.

### 1) Angle of the Target Video
The photos and videos of the parking lot taken from Dankook University buildings were not perfectly perpendicular to the ground.
This posed obstacles in detecting objects during the training process, especially with objects obscured by leaves or adjacent vehicles, and impacted the accurate visual processing of the application.

=> We resolved this by shooting as vertically as possible from the Humanities Building, which had no obstructions.

### 2) Training Data and Model Performance
When trained solely on the pre-prepared data from the Roboflow site, there was an issue of not accurately detecting objects in the Dankook University parking lot.

=> We directly captured and labeled approximately 200 data samples from Dankook University and applied data augmentation. As a result, the final model performance increased.

### 3) Application Implementation
We faced difficulties in outputting the detection result video using the trained model. There was a mismatch issue between the predefined parking space coordinates and the model's detection information for unknown reasons.
Example) A parking space remained marked as occupied even when the car had left (green line).

=> We output the center of the detected object as a yellow dot to verify the object status and confirmed there was no issue with the detection itself. We resolved this by making the application recognize only the Car class by removing the 'Empty' information from the class information.


## References
1) Datasets
- On Robowflow)

    https://universe.roboflow.com/swee-xiao-qi/parking-lot-availability/dataset/8

    https://universe.roboflow.com/smartparking-tijv5/smart_parking/dataset/2





2) Reference Code for Implementation
- On Ultralytics)

    https://docs.ultralytics.com/

- On Git-Hub) For Coordinate Selection and Applying Coordinate Information

    https://github.com/ultralytics/ultralytics.git

    https://github.com/olgarose/ParkingLot.git


