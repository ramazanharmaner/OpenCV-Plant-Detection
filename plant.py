import cv2

vid = cv2.VideoCapture("tarla.mp4")
plant_cascade = cv2.CascadeClassifier("plant_cascade\\cascade.xml")

while True:
    ret, frame = vid.read()
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plants = plant_cascade.detectMultiScale(gray, 3, 5)

    for (x, y, w, h) in plants:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 90), 3)
    

    cv2.imshow("Bitki Tespiti", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()

