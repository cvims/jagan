""" Compiles dataset by downloading videos from YouTube and extracting faces. """
import math
import os
from argparse import ArgumentParser
from ast import literal_eval
from typing import Optional
from urllib.error import URLError

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytube import YouTube
from pytube.cli import on_progress
from pytube.exceptions import PytubeError


def compile_dataset(dataset: str, video_idx_start: int, video_idx_stop: int, delete_source_videos: bool, verbose: bool):
    """
    Compiles the dataset by downloading the video from YouTube, extracting the faces from the videos and saving the
    faces as .png files.

    :param dataset: Specifies whether to compile training, validation or test set.
    :param video_idx_start: Start processing the list of unique video IDs at this position in the list.
    :param video_idx_stop: Stop processing the list of unique video IDs at this position in the list.
    :param delete_source_videos: If set to True, the downloaded videos are removed from disk once the faces have been
                                 extracted.
    :param verbose: If true, program prints more output.
    """
    # 1. Read in dataset csv.
    # =======================
    # The dataset csv contain information about the video IDs, sequence IDs, numbers of frame where the face was
    # detected and bounding boxes of detected faces.
    df = pd.read_csv(f'dataset_csvs/{dataset}_sequences.csv', index_col=0)

    # Get unique videos IDs. This list contains the IDs of all videos that will be downloaded from YouTube.
    video_ids = list(df.index.unique())

    # To make the compilation process more robust to failures, only process a specified subset of videos IDs. E.g.,
    # if supplying video_idx_start=100 and video_idx_stop=200, the 100 video IDs at positions 100 to 199 are
    # processed.
    video_ids_to_process = video_ids[video_idx_start: video_idx_stop]
    print(f"In this iteration, the videos IDs {video_ids_to_process} will be processed.")

    video_download_successes = []
    face_extraction_successes = []
    for video_id in video_ids_to_process:
        # 2. Download Video.
        # ==================
        # In order to download the correct resolution get frame heights for this video ID.
        frame_height = df.loc[video_id, 'height'][0]
        # Download the video with the provided video ID. You can specify how many retries are performed if the first
        # attempt at downloading a video was not successful.
        path_to_video = download_video(video_id=video_id, frame_height=frame_height, max_connection_attempts=2,
                                       dataset=dataset, verbose=verbose)
        # Record successful downloads.
        if path_to_video is not None:
            video_download_successes.append(True)
        else:
            video_download_successes.append(False)

        # 3. Select Faces in Video.
        # =========================
        if path_to_video:
            success = collect_faces_in_video(video_id, path_to_video, df.loc[video_id], verbose=verbose)
            face_extraction_successes.append(success)

            # 4. Delete Downloaded Videos.
            # ============================
            if delete_source_videos:
                print(f"DELETING video with ID {video_id} from {path_to_video}")
                try:
                    os.remove(path_to_video)
                except Exception as e:
                    print(f"Could not remove {path_to_video} because of {e}.")

    print(f'Completed! '
          f'Video download was tried on {len(video_download_successes)} videos and successful on '
          f'{sum(video_download_successes)} video. \n' 
          f'Face extraction was tried on {len(face_extraction_successes)} videos and successful on '
          f'{sum(face_extraction_successes)} videos.')


def download_video(video_id: str, frame_height: int, max_connection_attempts: int,
                   dataset: str, verbose: bool) -> Optional[str]:
    """
    Download the video with the provided video ID. You can specify how attempts are made to download a video.

    :param video_id: The video ID.
    :param frame_height: Determines if video will be downloaded in 720p or 1080p.
    :param max_connection_attempts: How many attempts are made to download a video.
    :param dataset: training, validation or test. Is used to name the directory where the video is downloaded to.
    :param verbose: If true, program prints more output.
    :return: The path to the downloaded video or None if the download was not successful.
    """
    # Init variable to count connection fails.
    connection_fails = 0
    # Init variable to store path to video.
    path_to_video = None

    # Try to download video from YouTube.
    while True:
        try:
            # Set yt object.
            yt = YouTube(f'http://youtube.com/watch?v={video_id}', on_progress_callback=on_progress)

            # Get streams from yt object.
            streams = yt.streams

            # Select stream with proper resolution.
            for x in [streams.filter(resolution=f"{frame_height}p"),
                      streams.filter(resolution=f"{frame_height}p", only_video=True),
                      ]:
                stream = x.first()
                if stream:
                    break
            else:
                raise Exception("No valid stream found")

            # Define path to video folder and path to video file.
            path_to_video_folder = os.path.join('dataset', dataset, video_id)
            path_to_video = os.path.join(path_to_video_folder, video_id + '.' + str(stream.subtype))

            # Download video to path_to_video_folder. Skip if video already exists.
            print(f"Downloading video with ID {video_id} to ./{path_to_video} ...")
            stream.download(output_path=path_to_video_folder, filename=video_id, skip_existing=True)
            print(f" ... finished downloading video ID {video_id}.")

            # Break while loop on successful download.
            break

        except (URLError, PytubeError) as e:
            # Skip video if not loaded after defined number of attempts.
            print(f"Could not download video with ID {video_id} due to {e}, retrying...")
            connection_fails += 1

            if connection_fails == max_connection_attempts:
                try:
                    print(f"Removing file {path_to_video}")
                    os.remove(path_to_video)
                except Exception as e:
                    print(f"Could not remove file {path_to_video} due to Exception: {e}.")
                finally:
                    return None

    return path_to_video


def collect_faces_in_video(video_id: str, path_to_video: str, face_dataframe: pd.DataFrame, verbose: bool) -> bool:
    """
    For a given video ID, extracts all faces given in the face_dataframe from the video.

    :param video_id: The video ID.
    :param path_to_video: Path to the saved video.
    :param face_dataframe: The pandas dataframe with informations about the dataset.
    :param verbose: If true, program prints more output.
    :return: True if collecting 30 images was successful, False otherwise.
    """
    print(f"Collecting faces for video ID {video_id}... ")
    path_to_video_folder = os.path.dirname(path_to_video)

    # Read in video with OpenCV.
    video_capture = cv2.VideoCapture(path_to_video)

    # Check if stream was opened correctly.
    if video_capture.isOpened() is False:
        print("Error opening video stream. Skipping this video.")
        return False

    # Compile sorted list of all timestamps where faces had been detected.
    all_timestamps_sorted = np.sort(face_dataframe.ts.unique())

    # First, read in all 30 frames to ensure that they all frames can be read. They will be saved to a temporary
    # dictionary.
    temp_frame_dict = {}

    for current_frame in all_timestamps_sorted:
        # Skip to frame to read.
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        # Read frame
        ret, frame = video_capture.read()
        if ret:
            temp_frame_dict[current_frame] = frame
        else:
            print(f"Error reading frame at timestep {current_frame}, skipping this video.")
            video_capture.release()
            cv2.destroyAllWindows()
            return False

    # After reading in all frames, get the faces from the frames.
    for current_frame in all_timestamps_sorted:
        frame = temp_frame_dict[current_frame]

        # Get all bounding boxes for current timestamp and extract all faces for current timestamp.
        # This is done by iterating all rows in the sub-dataframe of the current frame.
        for idx, row in face_dataframe[face_dataframe.ts == current_frame].iterrows():
            # Bounding boxes are saved as string. Transform into list with ast.literal_eval.
            x1, y1, x2, y2 = literal_eval(node_or_string=row['boundings'])

            # Calculate pixel ints to extract face from frame img.
            face = frame[
                   max(0, math.floor(y1)): min(frame.shape[0] - 1, math.ceil(y2)),
                   max(0, math.floor(x1)): min(frame.shape[1] - 1, math.ceil(x2)),
                   [2, 1, 0]
            ]

            # Get sequence ID for face.
            sequence_id = row['seq_id']

            # Save face as .png.
            save_folder = os.path.join(path_to_video_folder, str(sequence_id))
            os.makedirs(save_folder, exist_ok=True)
            plt.imsave(os.path.join(save_folder, f'{current_frame}.png'), face)

    print(f"... collected faces for video ID {video_id}.")
    video_capture.release()
    cv2.destroyAllWindows()
    return True


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['training', 'validation', 'test'])
    parser.add_argument('--start_idx', type=int, required=True)
    parser.add_argument('--stop_idx', type=int, required=True)
    parser.add_argument('--delete_source', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)

    args = parser.parse_args()

    compile_dataset(dataset=args.dataset, video_idx_start=args.start_idx, video_idx_stop=args.stop_idx,
                    delete_source_videos=args.delete_source, verbose=args.verbose)
