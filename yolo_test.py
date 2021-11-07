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
        yolo_boxes=yolo.run_detection(img)
        """for i, bbox in enumerate(user_draw_bboxes):
            final_bboxes.append(yolo.IOU_check_for_first_frame(bbox, yolo_boxes))"""
    else:
        yolo.run_multi_core_detection_setting(img)
        yolo.run_multi_core_detection(img)

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))

