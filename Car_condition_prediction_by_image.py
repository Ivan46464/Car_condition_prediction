from tensorboard.notebook import display

if __name__ == '__main__':
    import torch
   # print(torch.cuda.is_available())
    from PIL import Image as PILImage
    from ultralytics import YOLO
    #model = YOLO('yolov10n.pt')
    #model.train(data='data.yaml', epochs=120, imgsz=640, batch=16)
    model = YOLO('runs/detect/train9/weights/best.pt')
    results=model(source='test/images/car-scratch-repair_jpg.rf.a4c5e071e3c45fab9de44461ce1c3a71.jpg',save=True)
    import glob

    images = glob.glob('E:/Pycharm/Projects/Car_condition_prediction/Car_condition_predict/runs/detect/predict10/*.jpg')

    for image_path in images:
        img = PILImage.open(image_path)
        img.show()