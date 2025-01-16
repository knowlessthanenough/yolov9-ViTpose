import cv2

def test_video_display(video_path):
    """
    Reads and displays a video frame by frame using OpenCV.
    Press 'q' to exit the video window.
    
    Args:
        video_path (str): Path to the video file. Use 0 for the webcam.
    """
    # Open the video file or webcam
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video source. Path: {video_path}")
        return

    print("Press 'q' to quit.")
    
    while True:
        # Read frame by frame
        ret, frame = cap.read()
        
        if not ret:
            print("End of video or unable to fetch frame.")
            break
        
        # Display the frame
        cv2.imshow("Video Display Test", frame)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close display window
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# Replace 'video.mp4' with the path to your video file, or use 0 for webcam
test_video_display("./data/video/8-1.mp4")
