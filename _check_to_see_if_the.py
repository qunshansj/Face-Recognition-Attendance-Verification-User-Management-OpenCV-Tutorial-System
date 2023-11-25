
# check to see if the eye aspect ratio is below the blink

# threshold, and if so, increment the blink frame counter

if ear < EYE_AR_THRESH:

    COUNTER += 1

    # if the eyes were closed for a sufficient number of

    # then sound the alarm

    if COUNTER >= EYE_AR_CONSEC_FRAMES:

        # if the alarm is not on, turn it on

        if not ALARM_ON:

            ALARM_ON = True

            # check to see if an alarm file was supplied,

            # and if so, start a thread to have the alarm

            # sound played in the background

            if args["alarm"] != "":

                t = Thread(target=sound_alarm,

                    args=(args["alarm"],))

                t.deamon = True

                t.start()

        # draw an alarm on the frame

        frame=cv2ImgAddText(frame,"醒醒，别睡!",10,30,(255, 0, 0),30)

# otherwise, the eye aspect ratio is not below the blink

# threshold, so reset the counter and alarm

else:

    COUNTER = 0

    ALARM_ON = False
