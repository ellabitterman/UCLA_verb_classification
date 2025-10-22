"""
File: attention_curves.py
Author: Ella Bitterman
Email: bitterman.el@northeastern.edu
Note: This program takes me about 25-30 minutes to run,
      but it only took 13 minutes without color contrast calculations.

Sources:
 - This video helped immensely with the optical_flow_frames function: https://www.youtube.com/watch?v=_EsQYRU_krc
 - The color contrast calculations were modeled directly after a KFE technique created by Lai & Ying (2012)
"""
import os
import numpy as np
import cv2
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, reduces run time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# checking git!!!!!!

def read_video(video_path):
    """ Given a path to a video file, return a numpy array of all readable frames in video."""
    cap = cv2.VideoCapture(video_path)

    # Get array dimensions
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Set up empty array
    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)

    # Read frames in video and add to array
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames[idx] = frame
        idx += 1

    cap.release()
    return frames[:idx]


def optical_flow_frames(video, video_path, output_path, save_pngs=False):
    """ Given a video file and its path, return a value to represent optical flow for each frame in a video."""
    # Read video
    vid_array = read_video(video_path)
    num_frames = len(vid_array)
    gray_frames = np.empty((num_frames, vid_array.shape[1], vid_array.shape[2]), dtype=np.uint8)

    for i in range(num_frames):
        gray_frames[i] = cv2.cvtColor(vid_array[i], cv2.COLOR_BGR2GRAY)

    # Create empty image with same dimensions as input frame
    magnitude_data = np.zeros(num_frames)

    # Iterate over total number frames
    for i in range(1, num_frames):
        if i % 10 == 0:
            print(f"Calculating {video} optical flow: {i}/{len(vid_array)}")

        # Compute optical flow map -> returns 2D vector of input frame size
        op_flow = cv2.calcOpticalFlowFarneback(gray_frames[i-1], gray_frames[i], None, 0.5, 3,
                                               15, 3, 5, 1.2, 0)

        # Calculate magnitude and angle for each vector
        magnitude, angle = cv2.cartToPolar(op_flow[..., 0], op_flow[..., 1])
        magnitude_data[i] = np.mean(magnitude)

    if save_pngs:
        video_name = os.path.splitext(video)[0]
        plt.figure(figsize=(10, 5))
        plt.title(f"Motion Magnitude - {video_name}")
        plt.xlabel("Frame Number")
        plt.ylabel("Average Magnitude")
        plt.plot(magnitude_data, linewidth=2)
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(output_path, f"{video_name}_motion_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Motion curve saved to: {plot_path}")

    return magnitude_data.tolist()


def color_histogram(patch, bins_per_channel=4):
    """Given a patch of a frame, calculate the color histogram."""
    # Reduce number of possible values per channel from 256 -> 4
    quantized = patch // (256 // bins_per_channel)

    # Flatten 3D colorspace (R, G, B)
    flat = quantized[:, :, 0] * 16 + quantized[:, :, 1] * 4 + quantized[:, :, 2]

    # Counts occurrences of each index value
    hist = np.bincount(flat.flatten(), minlength=64)

    # Normalize histogram and return proportions instead of counts
    total = hist.sum()
    return hist / total if total > 0 else hist


def bhattacharyya(hist1, hist2):
    """Calculate Bhattacharyya distance, which is a measure of similarity between
    two histograms, hist1 and hist2.
    Returns: a value between 0 and 1 where 0 = identical and 1 = very different.
    """
    sum1 = np.sum(hist1)
    sum2 = np.sum(hist2)

    if sum1 == 0 or sum2 == 0:
        return 0

    # Calculate Bhattacharyya coefficient
    bhat_co = np.sum(np.sqrt(hist1 * hist2)) / np.sqrt(sum1 * sum2)
    return np.sqrt(max(0, 1 - bhat_co))


def color_contrast(video, video_path, output_path, save_pngs=False, k=3, step=32):
    """Given a video file and the path to said file, return a list of
    average calculated color contrast per frame."""

    # Read video
    frames = read_video(video_path)
    print(f"Processing {video}: {len(frames)} frames")

    # Set up empty array for results
    color_contrast_data = np.zeros(len(frames))

    # Process each frame
    for i, frame in enumerate(frames):

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Calculate all patches and associated histograms
        patch_histograms = {}
        for y in range(0, height - step, step):
            for x in range(0, width - step, step):
                patch = frame[y:y + step, x:x + step] # patch refers to a frame subsection with size step x step
                if patch.shape[0] == step and patch.shape[1] == step:
                    patch_histograms[(y, x)] = color_histogram(patch)

        # Initialize values needed to calculate color contrasts
        total_contrast = 0
        num_patches = 0
        half_k = k // 2

        # Iterate through each patch position and histogram
        for (y, x), hist_p in patch_histograms.items():
            patch_contrast = 0
            neighbor_count = 0

            # Calculate boundaries of neighborhood around patch
            y_start = max(0, y - half_k * step)
            y_end = min(height, y + (half_k + 1) * step)
            x_start = max(0, x - half_k * step)
            x_end = min(width, x + (half_k + 1) * step)

            # Iterate through each patch in the neighborhood
            for ny in range(y_start, y_end, step):
                for nx in range(x_start, x_end, step):

                    # Skip current patch to avoid comparing it to itself
                    if ny == y and nx == x:
                        continue

                    # Calculate bhattacharyya between current patch and neighbor
                    if (ny, nx) in patch_histograms:
                        hist_q = patch_histograms[(ny, nx)]
                        distance = bhattacharyya(hist_p, hist_q)
                        patch_contrast += distance
                        neighbor_count += 1

            # Add current patch's total contrast to frame total
            if neighbor_count > 0:
                total_contrast += patch_contrast
                num_patches += 1

        # Store average contrast for current frame
        color_contrast_data[i] = total_contrast / num_patches if num_patches > 0 else 0

    if save_pngs:
        video_name = os.path.splitext(video)[0]
        plt.figure(figsize=(10, 5))
        plt.plot(color_contrast_data, linewidth=2, color='purple')
        plt.title(f"Color Contrast (Bhattacharyya Distance) - {video_name}")
        plt.xlabel("Frame Number")
        plt.ylabel("Color Contrast Score")
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(output_path, f"{video_name}_color_contrast.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

    return color_contrast_data.tolist()


def process_single_video(video, video_dir, verb_path, json_path, save_pngs=False):
    """Process a single video."""
    try:
        video_path = os.path.join(video_dir, video)
        video_name = os.path.splitext(video)[0]

        # Create output directories IF saving curve pngs
        if save_pngs:
            output_path_opflow = os.path.join(verb_path, "optical_flow_vis")
            os.makedirs(output_path_opflow, exist_ok=True)
            output_path_color = os.path.join(verb_path, "color_contrast_vis")
            os.makedirs(output_path_color, exist_ok=True)
        else:
            output_path_opflow = None
            output_path_color = None

        # Create keyword args dictionaries
        opflow_kwargs = {"save_pngs": save_pngs, "output_path": output_path_opflow}
        color_kwargs = {"save_pngs": save_pngs, "output_path": output_path_color, "k": 3, "step": 32}

        # Process video
        magnitude_data = optical_flow_frames(video, video_path, **opflow_kwargs)
        color_attention_data = color_contrast(video, video_path, **color_kwargs)

        # Store in dictionary
        video_data = {
            "video_name": video_name,
            "class": int(video_name[5:6]),
            "num_frames": int(len(magnitude_data)),
            "magnitude_per_frame": [float(x) for x in magnitude_data],
            "color_contrast": [float(x) for x in color_attention_data],
        }

        # Save to JSON
        output_json_file = os.path.join(json_path, f"{video_name}.json")
        with open(output_json_file, "w") as f:
            json.dump(video_data, f, indent=4)

        print(f"Saved {output_json_file}")
        return True

    except Exception as e:
        print(f"Error processing {video}: {e}")
        return False


def main():
    # Set up directories
    current_dir = os.getcwd()
    verb_path = os.path.join(current_dir, "verb_classes")
    json_path = os.path.join(verb_path, "jsons")
    os.makedirs(json_path, exist_ok=True)

    # Pre-existing directory with ALL videos
    video_dir = "all_videos"
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    # Start timer
    start_time = time.time()

    # Parallel processing to speed up code!
    success_count = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_single_video, video, video_dir, verb_path, json_path, save_pngs = False): video
            for video in video_files
        }

        for future in as_completed(futures):
            if future.result():
                success_count += 1

    print(f"Successfully processed: {success_count}/{len(video_files)} videos")

    # End timer and print total run time
    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(f"Total run time: {int(minutes)}m {int(seconds):02d}s")


if __name__ == "__main__":
    main()
