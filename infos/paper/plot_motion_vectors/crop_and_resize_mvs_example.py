import cv2

frame_file = "frame_h264_000239.png"
#scale = 0.1
crop_region = [400, 224, 1186, 1000]

frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)
#frame_scaled = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
#crop_region = [c*scale for c in crop_region]

c = crop_region
frame_cropped = frame[int(c[1]):int(c[3]), int(c[0]):int(c[2]), :]

out_file = frame_file[:-4]
cv2.imwrite("{}_frame_cropped.png".format(out_file), frame_cropped, [cv2.IMWRITE_PNG_COMPRESSION, 0])

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 640, 360)
#cv2.namedWindow("frame_scaled", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("frame_scaled", 640, 360)
cv2.namedWindow("frame_cropped", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame_cropped", 640, 360)

while True:
    cv2.imshow("frame", frame)
    #cv2.imshow("frame_scaled", frame_scaled)
    cv2.imshow("frame_cropped", frame_cropped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
