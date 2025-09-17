

```
from ultralytics import YOLO

teacher_model = YOLO("<teacher-path>")

student_model = YOLO("yolo11n.pt)

student_model.train(
    data="<data-path>",
    teacher=teacher_model.model, # None if you don't wanna use knowledge distillation
    distillation_loss="cwd",
    epochs=100,
    batch=16,
    workers=0,
    exist_ok=True,
)
```
