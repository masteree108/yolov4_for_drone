import yolo_object_detection as yolo_obj
import cv2
from imutils.video import FPS

if __name__ == '__main__':
    img = cv2.imread('./image/047.jpg')

    fps = FPS().start()
    yolo = yolo_obj.yolo_object_detection('person')
    one_core_to_run = True
    if one_core_to_run == True:
        final_bboxes = []
        detection_intensity=1
        yolo_boxes = yolo.run_detection(img,detection_intensity)
        print("detected person qty:%d"%len(yolo_boxes))
    else:
        yolo.run_multi_core_detection_setting(img)
        yolo.run_multi_core_detection(img)

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
