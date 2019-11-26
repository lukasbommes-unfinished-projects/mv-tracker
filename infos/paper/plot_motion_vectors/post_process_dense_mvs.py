import cv2

dense_image_file = "mvs_image_dense_mpeg4_000651.png"
upscaled_dimensions = (768, 576)

dense_image = cv2.imread(dense_image_file, cv2.IMREAD_COLOR)
dense_image_scaled_up = cv2.resize(dense_image, upscaled_dimensions, interpolation=cv2.INTER_NEAREST)

out_file = dense_image_file[:-4]
cv2.imwrite("{}_scaled_up.png".format(out_file), dense_image_scaled_up, [cv2.IMWRITE_PNG_COMPRESSION, 0])


cv2.namedWindow("dense_image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("dense_image", 640, 360)
cv2.namedWindow("dense_image_scaled_up", cv2.WINDOW_NORMAL)
cv2.resizeWindow("dense_image_scaled_up", 640, 360)

while True:
    cv2.imshow("dense_image", dense_image)
    cv2.imshow("dense_image_scaled_up", dense_image_scaled_up)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
