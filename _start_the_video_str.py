
# start the video stream thread

print("[INFO] starting video stream thread...")

vs = VideoStream(src=args["webcam"]).start()

time.sleep(1.0)

# loop over frames from the video stream

while True:

    # grab the frame from the threaded video file stream, resize

    # it, and convert it to grayscale

    # channels)

    frame = vs.read()

    frame = imutils.resize(frame, width=450)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame

    rects = detector(gray, 0)
