import cv2
from ultralytics import YOLO
import yaml
import os

# 설정 파일 작성
# data = {
#     "test": './school_test/images/',
#     "names": {0: 'Car', 1: 'Empty'},
#     "nc": 2
# }

# with open('./custom.yaml', 'w') as f:
#     yaml.dump(data, f)
#
# # 설정 파일 확인
# with open('./custom.yaml', 'r') as f:
#     lines = yaml.safe_load(f)
#     print(lines)

# 학습된 모델 불러오기
model = YOLO('./runs/detect/train3/weights/best.pt')

# 비디오 파일 경로
video_path = './datasets/school_video/home.mp4'
output_path = './output_video3.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 비디오 작성 객체 생성
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임에 대해 예측 수행
    results = model.predict(source=frame, save=False)

    # 결과 프레임 가져오기
    annotated_frame = results[0].plot()

    # 비디오 파일에 프레임 저장
    out.write(annotated_frame)

# 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed and saved to", output_path)
