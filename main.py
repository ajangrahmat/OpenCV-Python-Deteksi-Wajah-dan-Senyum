import cv2

cascade_wajah = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
cascade_senyum = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def detect(gray, frame):
	wajah = cascade_wajah.detectMultiScale(gray, 1.29,6)
	for (x, y, w, h) in wajah:
		cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 255, 0), 2)
		cv2.putText(frame,'Wajah',(x, y + -12), font, 1, (255, 255, 0), 2)
		roi_gray = gray[y:y + h, x:x + w]
		roi_color = frame[y:y + h, x:x + w]
		senyum = cascade_senyum.detectMultiScale(roi_gray, 5.3, 6)

		for (sx, sy, sw, sh) in senyum:
			cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 255, 255), 2)
			cv2.putText(roi_color,'Senyum',(sx, sy + -12), font, 1, (0, 255, 255), 2)
	return frame

gambar = cv2.imread('gambar.jpg')
ubahKeGray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
detect(ubahKeGray, gambar)
cv2.imshow('Deteksi Wajah dan Senyum', gambar)
cv2.waitKey()