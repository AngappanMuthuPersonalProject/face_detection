import cv2

class FaceDetector(object):
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.classifier = cv2.CascadeClassifier(self.xml_path)
        print('class initiated')

    def detect(self, image):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True

        flag = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
               cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
            cv2.CASCADE_SCALE_IMAGE

        faces_coord = self.classifier.detectMultiScale(
            image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            flags=flag
        )
        return faces_coord