import cv2
import os


def extract_frames_from_video(video_path, output_folder):
    """
    Extract frames from a mp4 video and save them as images in a folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

def extract_frames_from_video_folder():
    """
    Extract frames from all mp4 videos in a folder and save them as images in a folder
    """
    video_folder = "video_segmentation/input"
    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_folder, video_file)
            output_folder = f'video_segmentation/output/{video_file}_frames'
            extract_frames_from_video(video_path, output_folder)
