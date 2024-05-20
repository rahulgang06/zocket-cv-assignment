from roboflow import Roboflow
import supervision as sv
import cv2

rf = Roboflow(api_key="7iMfivyXV2bmm6U6rzGm")
project = rf.workspace().project("coco-dataset-vdnr1")
model = project.version(12).model

result = model.predict("data/demo.jpg", confidence=40).json()
labels = [item["class"] for item in result["predictions"]]

detections = sv.Detections.from_roboflow(result)

label_annotator = sv.LabelAnnotator()
mask_annotator = sv.MaskAnnotator()

image = cv2.imread("data/demo.jpg")

annotated_image = mask_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(image=annotated_image, size=(16, 16))
print(result)
