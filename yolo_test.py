import yaml
from ultralytics import YOLO
import cv2
import os

from ultralytics import YOLO

data = {
    #"train" : './train/images/',
    #    "val" : './valid/images/',
    #    "test" : './test/images/',
        "test" : './test/images/',
        "names" : {0 : 'Car', 1 : 'Empty'},
    "nc" : 2
}

with open('./custom.yaml', 'w') as f :
    yaml.dump(data, f)

# check written file
with open('./custom.yaml', 'r') as f :
    lines = yaml.safe_load(f)
    print(lines)


# 학습된 모델 불러오기
model = YOLO('final_w.pt')

# 테스트 데이터에 대해 예측 수행
results = model.predict(source='./datasets/test/images',save=True, epochs=10)  # 테스트 이미지 경로 지정
plots = results[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()

#plots = results[0].plot
# 결과 출력
print(results)
#cv2.imshow("plot", plots)
