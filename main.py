from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(
    data='data.yaml',
    epochs=30,
    imgsz=512,
    batch=32,
    patience=7,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=2,
    degrees=0.0,
    translate=0.05,
    scale=0.2,
    shear=0.5,
    perspective=0.0002,
    flipud=0.0,
    fliplr=0.4,
    mosaic=0.7,
    mixup=0.0,
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.3,
    device=0,
    workers=4,
    save=True,
    save_period=5,
    val=True,
    project='result',
    name='dva',
)