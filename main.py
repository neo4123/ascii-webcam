import cv2
import numpy as np
from numba import jit
@jit(nopython=True)
def to_ascii_art(frame, image, box_height=12, box_width=16):
	height, width = frame.shape
	for i in range(0, height, box_height):
		for j in range(0, width, box_width):
			roi = frame[i:i + box_height, j:j + box_width]
			best_match = np.inf
			best_match_index = 0
			for k in range(1, image.shape[0]):
				total_sum = np.sum(np.absolute(np.subtract(roi, image[k])))
				if total_sum < best_match:
					best_match = total_sum
					best_match_index = k
			roi[:,:] = image[best_match_index]
		return frame

def generate_ascii_letters():
	image=[]
	letters = " \\ '(),-./:;[]_`{|}~"
	for letter in letters:
		img = np.zeros((12, 16), np.uint8)
		img = cv2.putText(img, letter, (0, 11), cv2.FRONT_HERSHEY_SIMPLEX, 0.5, 225)



cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)
	cv2.imshow("webcam", "frame")
	if cv2.waitkey(1) == ord('q'):
		break

cap.release()
cap.destroyAllWindows()
