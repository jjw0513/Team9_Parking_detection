# #from ultralytics import solutions
import parking_management
import cv2
from ultralytics import solutions

#(application)주차 공간의 경계 좌표를 지정해주기 위한 thinker GUI 불러오기
parking_selection = parking_management.ParkingPtsSelection()

#학습된 YOLO 가중치 정보를 불러와 객체를 탐지하도록 Load
parking_management_obj = parking_management.ParkingManagement(model_path="/final_w.pt")


# Path to json file, that created with above point selection app
polygon_json_path = "label_json/bounding_boxes.json"

# Video capture
cap = cv2.VideoCapture("./datasets/school_video/0605_school.mp4")   #탐지 대상 비디오 불러오기
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer 최종 탐지 결과를 비디오로 저장
video_writer = cv2.VideoWriter("parking management_park.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize parking management object
management = parking_management.ParkingManagement(model_path="final_w.pt")

while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        break

    json_data = management.parking_regions_extraction(polygon_json_path)
    results = management.model.track(im0, persist=True, show=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        management.process_data(json_data, im0, boxes, clss)

    management.display_frames(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()