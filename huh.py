from ultralytics import YOLO
import cv2

model = YOLO("weights of my model downloaded from hugging face")

img = cv2.imread("image path")

results = model.predict(
    source=img,
    imgsz=256
)

for r in results:
    probs = r.probs
    top_class = probs.top1
    confidence = probs.top1conf.item()
    THRESHOLD = 0.5
    if confidence < THRESHOLD:
        print('unknown')
    else:
        res = model.names[top_class]
        species, disease = res.split('___')
        disease = disease.replace('_', ' ')
        print(species)
        print(disease)
        print(f"Predicted class: {model.names[top_class]} ({confidence:.2%} confidence)")
