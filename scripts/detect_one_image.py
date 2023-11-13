from typer import Typer
import cv2
from motrackers.detectors import YOLOv3

app = Typer()

@app.command()
def detect(image: str, output: str):

    image = cv2.imread(image)
    model = YOLOv3(
        confidence_threshold=0.5,
        nms_threshold=0.2,
        draw_bboxes=True,
        use_gpu=False,
        weights_path='../../multi-object-tracker/examples/pretrained_models/yolo_weights/yolov3.weights',
        configfile_path='../../multi-object-tracker/examples/pretrained_models/yolo_weights/yolov3.cfg',
        labels_path='../../multi-object-tracker/examples/pretrained_models/yolo_weights/coco_names.json'
    )
    
    bboxes, confidences, class_ids = model.detect(image)
    updated_image = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)
    cv2.imwrite(output, updated_image)


if __name__ == '__main__':
    app()