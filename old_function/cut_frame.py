import cv2

#a function to save the fist frame of a video as an image
def cut_frame(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cv2.imwrite(save_path, frame)
    cap.release()
    return frame

cut_frame('./data/video/8-1.mp4', 'frame.jpg')