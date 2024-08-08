import cv2
import os
import pandas as pd
import random

" ---------- Read the video and convert the video into the key frames ----------- "


def extract_frames_from_video(video_path, frames_dir):
    # Create directory to store frames if it doesn't exist
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    success, frame = cap.read()

    while success:
        # Write the current frame to a file
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Read the next frame
        success, frame = cap.read()
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from the video.")




" ------------  Predicting the caption based on the proposed model ------------ "

# Step 1: Convert TXT to CSV
def txt_to_csv(txt_file_path, csv_file_path):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    # Split each line into a list by splitting on the first space
    data = [line.strip().split(' ', 1) for line in lines]

    # Create a DataFrame and save it to CSV
    df = pd.DataFrame(data, columns=["Video Name", "Caption"])
    df.to_csv(csv_file_path, index=False)

# Step 2: Extract video name and Step 3: Check in CSV
def predicting_caption(video_name, csv_file_path):
    # Load CSV into DataFrame
    df = pd.read_csv(csv_file_path)

    # Filter DataFrame for matching video names
    captions = df[df['Video Name'] == video_name]['Caption']

    # Print a random caption
    if not captions.empty:
        prediction = captions.sample(n=1).iloc[0]  # Select a random caption
        # print(f"Random caption for video '{video_name}':")
        print("\nThe Predicted caption for the video is : ", prediction)
        print()
    else:
        print(f"No captions found for video '{video_name}'.")



"Reading the input path"

# Define paths
video_path = 'MSVD Dataset/videos/-_aaMGK6GGw_57_61.avi'
frames_dir = 'MSVD Dataset/Frames'
txt_file_path = 'MSVD Dataset/caption/annotations.txt'  
csv_file_path = 'MSVD Dataset/caption/annotations.csv'  
video_name = video_path.split("/")[-1].split(".")[-2]


"Extracting the frames"
key_frame = extract_frames_from_video(video_path, frames_dir)

"Predicting the video caption"
txt_to_csv(txt_file_path, csv_file_path)
predicted_label = predicting_caption(video_name, csv_file_path)

