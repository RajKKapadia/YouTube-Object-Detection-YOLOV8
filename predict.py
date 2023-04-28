from ultralytics import YOLO
import cv2 as cv

model = YOLO('runs/detect/train/weights/best.pt')

result = model.predict(
    source='datasets/images/test/IMG_2301_jpeg_jpg.rf.2c19ae5efbd1f8611b5578125f001695.jpg',
    device=0
)[0]

print(result)

result_plotted = result.plot(line_width=1)
cv.imwrite('output.png', result_plotted)