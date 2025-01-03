import cv2
import onepose

if __name__ == '__main__':
    img = cv2.imread('people.jpg')
    model = onepose.create_model()

    keypoints = model(img)
    onepose.visualize_keypoints(img, keypoints, model.keypoint_info, model.skeleton_info)
    cv2.imwrite('people_keypoints.png', img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)