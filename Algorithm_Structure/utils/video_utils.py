import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        raise ValueError("No frames to save. The input list is empty.")
        # Get frame size
    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))  
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"Video saved: {output_video_path}")

def  test():
    video = read_video('E:\Omar 3amora\Football Analysis Proj\Input Videos\Input.mp4')
    save_video(video,'Output Videos/ffas.mp4')
    print("Done")
