import yolo_object_detection as yolo_obj
import cv2
from imutils.video import FPS

if __name__ == '__main__':
    img = cv2.imread('./image/0_1.jpg')

    fps = FPS().start()
    yolo = yolo_obj.yolo_object_detection('person')
    one_core_to_run = True
    if one_core_to_run == True:
        final_bboxes = []
        yolo_boxes = yolo.run_detection(img)
        # for test IOU function
        # use_draw_bboxes is mean user bounding in VOTT
        user_draw_bboxes = [[978, 526, 21, 47], [1089, 1076, 49, 131]]
        for i, bbox in enumerate(user_draw_bboxes):
            final_bboxes.append(yolo.IOU_check_for_first_frame(bbox, yolo_boxes))
        for i, box in enumerate (final_bboxes):
            print("1")
            # extract the bounding box coordinates
            (x, y) = (final_bboxes[i][0], final_bboxes[i][1])
            (w, h) = (final_bboxes[i][2], final_bboxes[i][3])
            # x = x + int(w/5)
            # draw a bounding box rectangle and label on the frame
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = "person"
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255,), 2)
        cv2.imwrite("IOU_frame.jpg", img)

    else:
        yolo.run_multi_core_detection_setting(img)
        yolo.run_multi_core_detection(img)

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
