
# loop over the face detections

for rect in rects:

    # determine the facial landmarks for the face region, then

    # convert the facial landmark (x, y)-coordinates to a NumPy

    # array

    shape = predictor(gray, rect)

    shape = face_utils.shape_to_np(shape)

    # extract the left and right eye coordinates, then use the

    # coordinates to compute the eye aspect ratio for both eyes

    leftEye = shape[lStart:lEnd]

    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)

    rightEAR = eye_aspect_ratio(rightEye)

    # average the eye aspect ratio together for both eyes

    ear = (leftEAR + rightEAR) / 2.0
