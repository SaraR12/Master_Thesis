import cv2

# Set Path
videoPath = './input/simtest05FEB3.mkv'
cap = cv2.VideoCapture(videoPath)


# Set output filetype:
filetype = '.png'
folder = './output/'

# Start naming pictures at index:
startIndex = 1000

ret = True
index = startIndex
while ret:
    ret, frame = cap.read()

    if index % 50 == 0:
        filename = folder + str(startIndex) + filetype
        print(filename)
        cv2.imwrite(filename, frame)
        startIndex += 1


    index += 1


cap.release()
