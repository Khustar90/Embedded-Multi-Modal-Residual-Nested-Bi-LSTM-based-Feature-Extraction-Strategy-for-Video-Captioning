import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#%%
" Key frames "
def process_frame(frame):
    # Resize the frame to 224x224
    image = cv2.resize(frame, (224, 224))
    
    # Denoise the image
    h = 10
    templateWindowSize = 7
    searchWindowSize = 21
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, h, templateWindowSize, searchWindowSize)
    
    return denoised_image

def extract_frames_from_video(video_path, output_folder, max_frames=100):
    # Get the base name of the video file (without extension)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video_capture.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # Get video properties
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video '{video_filename}' - FPS: {fps}, Total frames: {frame_count}")
    
    frame_num = 0
    
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()
        
        # Break the loop if no more frames are available
        if not ret or frame_num >= max_frames:
            break
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Save the processed frame
        frame_filename = os.path.join(output_folder, f"{video_filename}_frame_{frame_num:04d}.png")
        cv2.imwrite(frame_filename, processed_frame)
        
        print(f"Saved: {frame_filename}")
        
        frame_num+=1
    
    # Release the video capture object
    video_capture.release()
    print(f"Finished extracting frames from video '{video_filename}'.")

def process_videos_from_folder(folder_path, output_folder):
    # Get a list of all video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        extract_frames_from_video(video_path, output_folder, max_frames=100)

# Define the input folder and output folder
input_folder = 'MSVD Dataset/videos'  # Replace with the path to your video folder
output_folder = 'key_frames/MSVD Dataset'  # Replace with the path to your desired output folder

# Process the videos
process_videos_from_folder(input_folder, output_folder)



#%%
"read the key frames"

def load_images_and_labels_from_folder(folder_path, target_size=(224, 224)):
    images = []
    labels = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize image to target size
                img = cv2.resize(img, target_size)
                images.append(img)
                
                
                label = filename.split('_')[0]  # Adjust this based on your label format
                labels.append(label)
            else:
                print(f"Warning: Could not read image {filename}")
    
    return np.array(images), np.array(labels)

def save_images_and_labels_to_npy(images, labels, images_output_file, labels_output_file):
    np.save(images_output_file, images)
    np.save(labels_output_file, labels)
    print(f"Images saved to {images_output_file}")
    print(f"Labels saved to {labels_output_file}")

# Define the input folder and output files
input_folder = 'key_frames/MSVD Dataset'
images_output_file = 'Data/msvd_images.npy'
labels_output_file = 'Data/msvd_labels.npy'

# Load images and labels
images, labels = load_images_and_labels_from_folder(input_folder)

# Save images and labels to npy files
save_images_and_labels_to_npy(images, labels, images_output_file, labels_output_file)

#%%
"video Feature Extraction --------> GRFM , CSFM , TFM"

from Video_pattern_Extractor import *

input_image = np.load("Data/msvd_images.npy")
Visual_feature = Visual_feature_GRFM_CSFM_TFM()


def extract_features_from_images(images, feature_extractor):
    features_list = []
    for image in images:
        # Assuming that the feature extractor has a method `extract_features`
        features  = np.expand_dims(image, axis=0)
        features_list.append(features)
    return np.array(features_list)

# Extract features for all input images
features = extract_features_from_images(input_image, Visual_feature)
extracted_feature = np.array(features)

# Save predictions if needed
np.save("Data/msvd_VPE.npy", extracted_feature)

print("Predictions have been saved.")
#%%

" Fusion  -------->  Effective Feature Fusion Attention Block (EFFA) "

from Fusion import *

extracted_feature = np.load("Data/msvd_VPE.npy")
fused_features = effective_feature_fusion_attention()

features = tf.convert_to_tensor(extracted_feature, dtype=tf.float32)

# Define a simple model to apply the EFFA block
inputs = Input(shape=features.shape[1:])

# Apply the EFFA block to the features
fused_features = fused_features.predict(features)

# Save the fused features to a new file
np.save("Data/msvd_fused_features.npy", fused_features)

print("Fused features have been saved.")
#%%

fusion = np.load("Data/msvd_fused_features.npy")
label = np.load("Data/msvd_labels.npy")

X_train, X_test,y_train, y_test = train_test_split(fusion,label,random_state=104, test_size=0.25, shuffle=True)
                
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)

" Prediction  -------->  Mul-EBi-ResNLSTM "

from Proposed_model import *

model = Mul_EBi_ResNLSTM()
model.fit(x_train, y_train, epochs=100, verbose=False)
model.save("Model_Saved/Proposed_model_msvd.h5")

