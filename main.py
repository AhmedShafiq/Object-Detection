import cv2

#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    #print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':

    # Initialize Opencv DNN
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1/255)

    # Load Classes
    classes = []
    with open('classes.txt', "r") as file_object:
        for class_name in file_object.readlines():
            class_name = class_name.strip()
            classes.append(class_name)

    # Intiallize Camera
    cap = cv2.VideoCapture('input.mp4')
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (1280, 720))

    while True:
        # Get Frames
        ret, frame = cap.read()

        # Object Detection
        (class_ids, scores, boxes) = model.detect(frame)
        for class_id, score, box in zip(class_ids, scores, boxes):
            (x, y, w, h) = box
            cv2.putText(frame, classes[class_id], (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 0, 100*int(class_id)), 3)


        # Display Frames
        cv2.imshow("Frame", frame)
        out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()