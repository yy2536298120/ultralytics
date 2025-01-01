import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt')
    model.predict(source=r'D:\Proj\pythonProj\datasets\VisDrone\VisDrone2019-DET-test-dev\images',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show = True,
                  save=True,
                #   conf=0.2,
                #   iou=0.7,
                )