"""from PIL import Image

im = Image.open("photosDPI/MicrosoftTeams-image (1).jpg")
im.save("photosDPI/jpg/OneAGVPosERROR.jpg", dpi=(600,600))"""
import cv2

vid = cv2.VideoCapture('LargeWarehouseVideo.avi')
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
ret, frame = vid.read()
out = cv2.VideoWriter('videotest.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while ret:
    out.write(frame)
    ret, frame = vid.read()

vid.release()
out.release()