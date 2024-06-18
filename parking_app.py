# #from ultralytics import solutions
import parking_management
parking_selection = parking_management.ParkingPtsSelection()
parking_management_obj = parking_management.ParkingManagement(model_path="/final_w.pt")
import cv2
from ultralytics import solutions

# Path to json file, that created with above point selection app
polygon_json_path = "label_json/bounding_boxes.json"

# Video capture
cap = cv2.VideoCapture("./datasets/school_video/0605_school.mp4")   #비디오 일치시키기
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
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