# JAGAN Dataset Compilation Script
This code can be used to compile the dataset that was used to produce 
the results in the paper *JAGAN - Temporally coherent video anonymization
through GAN inpainting* by downloading the source videos from YouTube
and extracting the faces provided by the enclosed .csv files. 
**We cannot guarantee that all videos are still available on YouTube
that were downloaded when creating the dataset!**

## Installation
To install the necessary requirements, install a new conda env with the 
provided `environment.yml`: 

```shell
conda env create -f environment.yml
```

## Basic Usage
Note: for this to work, you need the 3 files
- dataset_csvs/training_sequences.csv
- dataset_csvs/validation_sequences.csv
- dataset_csvs/test_sequences.csv

In each of these csvs, we have one column that contains the video ID. 
When running the script, all the IDs are combined into one list of video IDs
for specified the dataset (i.e., for training, validation or test).
This list of video ID is indexed by supplying a start and an stop index.

For instance, you can obtain the training dataset for
videos with the video IDs in `video_IDs[0:2]` by executing

```shell
python compile_dataset.py --dataset=training --start_idx=0 --stop_idx=2
```

This makes it possible to compile the dataset step by step. 


Available command line arguments are 
- `--dataset`: Specifies which dataset you want to compile. Available
    choices are 'training', 'validation' and 'test'. 
- `--start_idx`: Specifies the index in the list of video IDs of the 
    video ID that you want to start with. 
- `--stop_idx`: Specifies the last index in the list of video IDs you 
  want to process.
- `--delete_source`: If you set this to `True`, the downloaded videos
    are removed once faces have been extracted. Default is `False`
- `--verbose`: Gives you more output on STDOUT if set to `True`. Default
    is `False`.
  
The number of videos
in the different datasets are: 
- training: 28182
- validation: 5709
- test: 5711

This means that the command for a download of the complete training set with 
deletion of downloaded videos after face extraction would be 
```shell
python compile_dataset.py --dataset=training --start_idx=0 --stop_idx=28182 --delete_source=True
```

  
## Folder Structure
Videos and images of faces will be downloaded to the folder `./dataset`
according to the following folder structure: 

```
dataset
|___training
|   |
|   |___video ID 
|       |
|       |___ sequence ID 
|            | frame_{x}.png
|            | frame_{...}.png
|            | frame_{x+30}.png
|       
|___validation
|   |
|   |___video ID 
|       |
|       |___ sequence ID 
|            | frame_{x}.png
|            | frame_{...}.png
|            | frame_{x+30}.png
|
|___test 
|   |
|   |___video ID 
|       |
|       |___ sequence ID 
|            | frame_{x}.png
|            | frame_{...}.png
|            | frame_{x+30}.png
```

Each `sequence_ID` sub-folder will contain 30 images from subsequent 
frames. Note that `x` does not always start with 0 but with the frame 
timestamp it was extracted from, for instance 8755.png up to 8785.png.
