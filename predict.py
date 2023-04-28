from ultralytics import YOLO
import cv2 as cv

model = YOLO('runs/detect/train/weights/best.pt')

result = model.predict(
    source='datasets/images/train/IMG_2306_jpeg_jpg.rf.9bba6ce48724517b19474b85178391c1.jpg',
    device=0
)[0]

print(result)

result_plotted = result.plot(line_width=1)
cv.imwrite('output.png')