import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\Proj\pythonProj\ultralytics\ultralytics\cfg\models\11\yolo11.yaml')
    #model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=32,
                close_mosaic=10,
                device='0',
                optimizer='SGD', # using SGD
                project='runs/train',
                name='exp',
                )