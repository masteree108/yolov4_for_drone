import yolo_object_detection as yolo_obj
import cv2

if __name__ == '__main__':
    img = cv2.imread('./image/0_1.jpg')
    yolo = yolo_obj.yolo_object_detection()
    yolo.run_detection(img, 'person')
