"""
File: KFE_fuse.py
Author: Ella Bitterman
Email: bitterman.el@northeastern.edu
Desc: Fuse motion intensity and color contrast curves into one weighted attention curve.
  Then using this curve, extract key frames with either the segment method or peak method.
  Lastly, use VideoMAE to process videos and extract embeddings

  Segment Method:

  Peak Method:

"""
import json
import os
import cv2
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as s
import pandas as pd
from transformers import VideoMAEImageProcessor, VideoMAEModel
import torch
import time



def normalize(lst):
    """ Given a list of numerical values, return the normalized list."""
    to_norm = np.array(lst)
    normalized = preprocessing.normalize([to_norm])

    return normalized[0].tolist()

def fuse_curves(vid, current_dir, save_pngs=False, motion_weight=3):
    """ Fuse all attention curves PER VIDEO into one PER VIDEO. Motion_weight is set to 3,
    meaning motion is weighted 3x more than color in creating the final curve.
    """
    mag_data = vid["magnitude_per_frame"]
    color_data = vid["color_contrast"]

    if len(mag_data) != len(color_data):
        print("Length of magnitude and color data are not equal")
        return None

    # Calculate weighted average
    fused = [(motion_weight * mag_data[i]) + ((1 - motion_weight) * color_data[i])
             for i in range(len(mag_data))]
    # MAY BE INCORRECT!! Claude suggests updating to:
    # fused = [(motion_weight * mag_data[i] + color_data[i]) / (motion_weight + 1) for i in range(len(mag_data))]

    if save_pngs:
        fuse_path = os.path.join(current_dir, 'verb_classes', 'fuse')  # TEMP
        os.makedirs(fuse_path, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.title(f"Fused Attention Curve - {vid["video_name"]}")
        plt.xlabel("Frame Number")
        plt.plot(fused, linewidth=2)
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(fuse_path, f"{vid["video_name"]}_fused.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Fused attention curve saved to: {plot_path}")

    return fused

def identify_peaks(data):
    """ Identify all local maxima for each fused curve """
    df = pd.DataFrame(data, columns=['value'])
    return df.iloc[s.argrelmax(np.array(data), order=1)[0]]

def too_few_frames_extract(current_n_frames, fused_data, target_n_frames):
    """ Given a video with less than the target number of frames, find frames with highest fused attention scores
    and duplicate them. Return a list w/ frames to duplicate that will get the video to the target number of frames.
    :params
        current_n_frames: current number of frames
        fused_data: fused data
        target_n_frames: target number of frames
    :returns
        final_lst: list of frames indices to duplicate
    """
    fused_sorted = sorted(fused_data, reverse=True)
    n_needed = target_n_frames - current_n_frames
    frames_to_duplicate = fused_sorted[:n_needed]

    # Ensure that original indices are used, not the sorted ones
    dct = {}
    for index, value in enumerate(fused_data):
        dct[index] = value

    final_lst = [key for key, val in dct.items() if val in frames_to_duplicate]
    return final_lst

def extract_keyframes_per_segment(fused_data, vid_name, n_keyframes=16):
    """
    Extract exactly n_keyframes by dividing video into segments
    and selecting the highest attention frame from each segment.

    :param fused_data: list of fused attention values per frame
    :param n_keyframes: number of keyframes to extract (default 16)
    :return: list of frame indices (0-indexed)
    """
    n_frames = len(fused_data)
    segment_size = n_frames / n_keyframes
    keyframe_indices = []

    if n_frames <= n_keyframes:
        return list(range(0, n_keyframes))

    for i in range(n_keyframes):
        # Calculate start and end of this segment
        start_idx = int(i * segment_size)
        end_idx = int((i + 1) * segment_size)

        # Handle last segment to include any remaining frames
        if i == n_keyframes - 1:
            end_idx = n_frames

        # Get the segment
        segment = fused_data[start_idx:end_idx]

        # Find index of max value within segment
        local_max_idx = np.argmax(segment)

        # Convert to global frame index
        global_idx = int(start_idx + local_max_idx)
        keyframe_indices.append(global_idx)

    return keyframe_indices

def extract_keyframes_from_peaks(fused_data, peaks_dict, vid_name, n_keyframes=16):
    """
    Alternative: Extract exactly n_keyframes from detected peaks,
    with temporal spreading to avoid clustering.

    :param fused_data: list of fused attention values per frame
    :param peaks_dict: dict of {frame_idx: peak_value}
    :param vid_name: video name
    :param n_keyframes: number of keyframes to extract
    :return: list of frame indices
    """
    if len(peaks_dict) <= n_keyframes:
        # If we have fewer peaks than needed, fall back to segment method
        return extract_keyframes_per_segment(fused_data, vid_name, n_keyframes)

    # Sort peaks by value (descending)
    sorted_peaks = sorted(peaks_dict.items(), key=lambda x: x[1], reverse=True)

    selected_frames = []
    min_distance = len(fused_data) // (n_keyframes * 2)  # Initial minimum distance

    for frame_idx, peak_value in sorted_peaks:
        # Check if this frame is far enough from already selected frames
        if all(abs(frame_idx - selected) >= min_distance for selected in selected_frames):
            selected_frames.append(frame_idx)

            if len(selected_frames) == n_keyframes:
                break

    # If we still don't have enough frames, reduce minimum distance and try again
    while len(selected_frames) < n_keyframes and min_distance > 1:
        min_distance = min_distance // 2
        selected_frames = []

        for frame_idx, peak_value in sorted_peaks:
            if all(abs(frame_idx - selected) >= min_distance for selected in selected_frames):
                selected_frames.append(frame_idx)

                if len(selected_frames) == n_keyframes:
                    break

    # Final fallback: just take top n_keyframes peaks
    if len(selected_frames) < n_keyframes:
        selected_frames = [frame_idx for frame_idx, _ in sorted_peaks[:n_keyframes]]

    # Sort by frame index for temporal order
    selected_frames.sort()

    return selected_frames

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  ###
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  ###

    # Pre-allocate numpy array
    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)

    ind = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames[ind] = frame
        ind += 1

    cap.release()
    return frames[:ind]

def extract_frames(video_path, kf_lst, fused_data, resize=(224, 224), target_frames=16):

    # Read all frames once using the optimized function
    all_frames = read_video(video_path)
    total_frames = len(all_frames)

    frames = []
    failed_indices = []

    for idx in kf_lst:
        if idx >= total_frames:
            #print(f"Warning: Frame {idx} is beyond video length ({total_frames})")
            failed_indices.append(idx)
            continue

        # Directly access the frame from the array
        frame = all_frames[idx]

        # Convert color space and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(frame)

    # If the video has less than the target number of frames:
    if len(frames) < target_frames:
        # Get list of frames to duplicate
        frames_to_duplicate = too_few_frames_extract(len(frames), fused_data, target_frames)

        # Build new list with duplicates in correct places
        dup_frames = []
        for idx, frame in enumerate(frames):
            dup_frames.append(frame)

            # If this frame index should be duplicated, add it again
            if idx in frames_to_duplicate:
                dup_frames.append(frame.copy())
        frames = dup_frames

    # frames = np.array(frames) # Convert list to numpy array

    return frames


# Different methods for embedding extraction
def vidmae(emb_dir, video_dir, video_files, keyframe_dct, fuse_dct):

    # Create videomae directory inside embeddings directory
    videomae_dir = os.path.join(emb_dir, "VideoMAE_embs")
    os.makedirs(videomae_dir, exist_ok=True)

    # Load VideoMAE processor and model
    videomae_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_dir, video_file)

        # Change which keyframe dictionary depending on method chosen !!!!
        frames = extract_frames(video_path, keyframe_dct[video_name], fuse_dct[video_name],
                                resize=(224, 224), target_frames=16)

        # Preprocess and generate embedding
        inputs = videomae_processor(frames, return_tensors="pt")

        # Run VideoMAEModel
        with torch.no_grad():
           outputs = videomae_model(**inputs)

        # Get video representation vector
        videomae_emb = outputs.last_hidden_state[:, 0]

        # Save embeddings to jsons -> one json per video
        videomae_emb_path = os.path.join(videomae_dir, f"{video_name}_VideoMAE_emb.json")
        videomae_emb_data = videomae_emb.cpu().numpy().tolist()
        with open(videomae_emb_path, 'w', encoding='utf-8') as f:
            json.dump({"embedding": videomae_emb_data}, f, ensure_ascii=False, indent=2)
        print(f"{video_name} embedding saved as json to VideoMAE_embs")

    print("All videos processed with VideoMAE Model!")






def main():
    # Set up directories
    current_dir = os.getcwd()
    json_path = os.path.join(current_dir, "verb_classes", "jsons")
    video_dir = "../all_videos"
    embedding_dir = os.path.join(current_dir, "verb_classes", "embeddings")
    os.makedirs(embedding_dir, exist_ok=True)

    # Create list of each filename in jsons folder
    json_files = [file for file in os.listdir(json_path)
                  if file.endswith('.json') and '_embedding' not in file]

    # Create list of each filename in videos folder
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    # Create keyword args dictionaries for fuse_curves function
    fuse_kwargs = {"current_dir": current_dir, "save_pngs": False, "motion_weight": 3}

    # Initialize dictionaries
    fuse_dct = {} # e.g. {class1_lift_1: [0.324, 0.565, ..., n_frames]}
    peak_dct = {} # e.g. {class1_lift_1: {5: 0.554, 13: 0.68, ..., peak_index: y-value}, class2_kick_4: { ...

    keyframe_seg_dct = {}  # Method 1: Segment-based
    keyframe_peak_dct = {} # Method 2: Peak-based

    # Open json files and normalize motion magnitude and color contrast data
    for filename in json_files:
        file_path = os.path.join(json_path, filename)
        with open(file_path, 'r', encoding='utf-8') as data:
            vid_data = json.load(data)
            vid_data["magnitude_per_frame"] = vid_data["magnitude_per_frame"][:1] + normalize(vid_data["magnitude_per_frame"][1:])
            vid_data["color_contrast"] = normalize(vid_data["color_contrast"])
            fuse_per_vid = fuse_curves(vid_data, **fuse_kwargs)
            # fuse_per_vid = vid_data["magnitude_per_frame"] # Use for motion only -> when excluding color contrast
            peaks_per_vid = identify_peaks(fuse_per_vid)

            fuse_dct[vid_data["video_name"]] = fuse_per_vid

            peaks_only = {}
            for i in range(len(peaks_per_vid["value"])):
                peaks_only[int(peaks_per_vid["value"].index[i])] = float(peaks_per_vid["value"].iloc[i])

            peak_dct[vid_data["video_name"]] = peaks_only

    # Extract keyframes for each video
    for vid_name in fuse_dct.keys():
        # Method 1: Segment-based
        keyframes_segmented = extract_keyframes_per_segment(fuse_dct[vid_name], vid_name, n_keyframes=16)
        keyframe_seg_dct[vid_name] = keyframes_segmented

        # Method 2: Peak-based
        keyframes_peaks = extract_keyframes_from_peaks(fuse_dct[vid_name], peak_dct[vid_name], vid_name, n_keyframes=16)
        keyframe_peak_dct[vid_name] = keyframes_peaks

    # 1. VideoMAE method -> same framework as extractVideoEmbedding.py
    vidmae(embedding_dir, video_dir, video_files, keyframe_peak_dct, fuse_dct)



if __name__ == "__main__":
    main()

