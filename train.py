from ultralytics import YOLO

# teacher_model = YOLO("/home/sbkwon/works/ultralytics_sp/runs/train/yolo11x-pose-widefd3/weights/best.pt")

# student_model = YOLO("/home/sbkwon/works/ultralytics_sp/yolo11s-pose_origin.pt")

# student_model.train(
#     data="/home/sbkwon/works/ultralytics_sp/ultralytics/cfg/datasets/dataset_ym.yaml",
#     cfg='/home/sbkwon/works/ultralytics_sp/ultralytics/cfg/configuration_ym.yaml',
#     teacher=teacher_model.model, # None if you don't wanna use knowledge distillation
#     distillation_loss="mgd",
#     epochs=100,
#     batch=16,
#     imgsz=640,
#     project="runs/train",
#     name="yolo11s-pose-widefd3-kd-0919-mgd",
#     device="5",  
#     workers=8,
# )

teacher_model_2 = YOLO("/home/sbkwon/works/ultralytics_sp/runs/train/yolo11x-pose-widefd3/weights/best.pt")

student_model_2 = YOLO("/home/sbkwon/works/ultralytics_sp/yolo11s-pose.pt")

student_model_2.train(
    data="/home/sbkwon/works/ultralytics_sp/ultralytics/cfg/datasets/dataset_fd8.yaml",
    cfg='/home/sbkwon/works/ultralytics_sp/ultralytics/cfg/configuration_ym.yaml',
    teacher=teacher_model_2.model, # None if you don't wanna use knowledge distillation
    distillation_loss="mgd",
    epochs=100,
    batch=16,
    imgsz=640,
    project="runs/train",
    name="yolo11s-pose-widefd3-kd-0922-mgd",
    device="3",  
    workers=8,
)
