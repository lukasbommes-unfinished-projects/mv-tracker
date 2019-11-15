import numpy as np
import cv2

image = np.zeros((400,400,3), dtype=np.uint8)

mvs_motion_x = np.array([-21/4])
mvs_motion_y = np.array([-74/4])

mvs_motion_y *= -1

#mvs_motion_x = np.array([100., 500., 300.])
#mvs_motion_y = np.array([100., 500., 300.])

mvs_motion_magnitude, mvs_motion_angle = cv2.cartToPolar(mvs_motion_x, mvs_motion_y)
mvs_motion_angle = mvs_motion_angle * 180 / (2 * np.pi)  # hue channel is [0, 180]
mvs_motion_magnitude = cv2.normalize(mvs_motion_magnitude, None, 0, 255, cv2.NORM_MINMAX)

print(mvs_motion_angle)
print(mvs_motion_magnitude)

image[0:380, 0:380, 0] = mvs_motion_angle
image[0:380, 0:380, 1] = 255

image[:, :, 2] = 255
image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

while True:
    cv2.imshow("image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
