import cv2  # importing cv2 that library for open CV


# Face Classifer that is already classified
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# how to get the web cam , instead of 0 add the file so it runa on that mp4
webcam = cv2.VideoCapture(0)

# show the currrent frame
while True:
    # here  is how we read the current frame
    successful_frame_read, frame = webcam.read()  # reads the single frame
    # Just a safe check if there is a break in the frame
    if not successful_frame_read:
        break
    # Change to grey scale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first , will tell us where all the face are , return an array of points where the faces are
    faces = face_detector.detectMultiScale(frame_grayscale)

    # print(faces)
    # run this face detector on each of these faces
    for (x, y, w, h) in faces:

        # draw a rectangle aroung the face for clearity, cv2 allows us that by ->  rectangle ,
        # we want to draw it on the color frame
        #                      topleft,bottomright,color,pixles
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        #                x: , y:
        # from 0 to every single row and coloumn in this frame
        # getting the sub frame
        the_face = frame[y:y+h, x:x+w]

        # change it from color to grey
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        # for smiles
        # we blur the frame to detect the smile, minNeighbors a lot of redundant rectandles for it to be a smile
        smiles = smile_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=20)
        # find all the smiles in the face
        # for (x_, y_, w_, h_) in smiles:
        # draw all the smiles rectangles around the face
        # draw a rectangle aroung the smile for clearity, cv2 allows us that by ->  rectangle ,
        # we want to draw it on the color frame
        #                      topleft,bottomright,color,pixles
        #cv2.rectangle(the_face, (x_, y_),(x_+w_, y_+h_), (87, 150, 200), 3)

        # label the smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'WHY SO SERIOUS!!???', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # show the current frame
    cv2.imshow('Smile Detector', frame)
   # lets wait for a key to be pressed and then quit
    cv2.waitKey(1)
# release the web cam after the app is done running
webcam.release()
cv2.destroyAllWindows()


print("whats up")
