import torch
import yaml
import os
from ultralytics import YOLO
import shutil
from sklearn.model_selection import train_test_split
print("CUDA available: ", torch.cuda.is_available())

# 데이터 경로 설정
train_images_path = 'datasets/train2/images/'
train_labels_path = 'datasets/train2/labels/'
valid_images_path = 'datasets/valid2/images/'
valid_labels_path = 'datasets/valid2/labels/'


# 유효성 검사 데이터를 저장할 디렉토리 생성
os.makedirs(valid_images_path, exist_ok=True)
os.makedirs(valid_labels_path, exist_ok=True)

# 이미지와 라벨 파일 목록 가져오기
image_files = [f for f in os.listdir(train_images_path) if f.endswith('.jpg') or f.endswith('.png')]
label_files = [f for f in os.listdir(train_labels_path) if f.endswith('.txt')]

# 이미지와 라벨이 일치하는지 확인하는 함수
def match_files(images, labels):
    image_stems = {os.path.splitext(f)[0] for f in images}
    label_stems = {os.path.splitext(f)[0] for f in labels}
    common_stems = image_stems.intersection(label_stems)
    matched_images = [f for f in images if os.path.splitext(f)[0] in common_stems]
    matched_labels = [f for f in labels if os.path.splitext(f)[0] in common_stems]
    return matched_images, matched_labels

# 이미지와 라벨이 일치하는지 확인
matched_images, matched_labels = match_files(image_files, label_files)

# train 데이터를 70%, validation 데이터를 30%로 나누기
train_images, valid_images, train_labels, valid_labels = train_test_split(
    matched_images, matched_labels, test_size=0.3, random_state=42
)



# validation 데이터를 분리된 디렉토리로 이동
for img_file in valid_images:
    shutil.move(os.path.join(train_images_path, img_file), os.path.join(valid_images_path, img_file))
for lbl_file in valid_labels:
    shutil.move(os.path.join(train_labels_path, lbl_file), os.path.join(valid_labels_path, lbl_file))


data = {
    "train" : './train2/images/',
        "val" : './valid2/images/',
        "test" : '/test/images/',
        "names" : {0 : 'Car', 1 : 'Empty'},
    "nc" : 2
}

with open('./custom.yaml', 'w') as f :
    yaml.dump(data, f)

# check written file
with open('./custom.yaml', 'r') as f :
    lines = yaml.safe_load(f)
    print(lines)


model = YOLO('yolov8s.pt')

model.train(data='./custom.yaml' , epochs=300, batch = 32)