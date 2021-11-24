import cv2
import matplotlib.pyplot as plt

bgr_img = cv2.imread(r'qr_code.png')
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
hsv_army = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

qrCodeDetector = cv2.QRCodeDetector()
decodedText, points, _ = qrCodeDetector.detectAndDecode(bgr_img)
if points is not None:
    print('QR detected')
    print(f'Decoded text: {decodedText}')
    detected_img = rgb_img.copy()
    cv2.rectangle(detected_img, (points[0][0][0],points[0][0][1]), (points[0][2][0],points[0][2][1]), (0, 255, 0), 2)
    crop_image = detected_img[int(points[0][0][1]):int(points[0][2][1]),int(points[0][0][0]):int(points[0][2][0])]
    cv2.imwrite('qr.png', cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))