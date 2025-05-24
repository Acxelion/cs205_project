import argparse
from pathlib import Path

import numpy as np
from pypianoroll import Multitrack, Track
import pypianoroll

import os
import os.path
import random

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect training data from MIDI files"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        required=True,
        help="directory containing .npz files",
    )
    parser.add_argument(
        "-l",
        "--label_dir",
        type=Path,
        required=True,
        help="directory containing text files for labels"
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        required=True,
        help="output filename",
    )
    parser.add_argument(
        "-k",
        "--count",
        type=int,
        required=True,
        help="number of random samples"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="whether to search directory recursively",
    )
    return parser.parse_args()

# converts msd_id to the path to it
def msd_id_to_dirs(msd_id):
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

# given a dataset_root, a id_dir, returns k random unique samples' file paths
def select_random(dataset_root: str, id_dir: str, k: int):
    _root = Path(dataset_root)
    id_list = []
    for path in os.listdir(id_dir):
        filepath = os.path.join(id_dir, path)
        if os.path.isfile(filepath):
            if("masd_labels.txt" in filepath or "masd_labels_cleansed.txt" in filepath):
              continue
            with open(filepath) as f:
                id_list.extend([line.rstrip() for line in f])
    id_list = list(set(id_list))

    return random.sample(id_list, k)

def main():
    # parses arguments
    args = parse_arguments()

    # selects a set of random multitrack objects
    filenames = select_random(args.input_dir, args.label_dir, args.count)

    print(filenames)

    if not os.path.exists("./data/test"):
        os.mkdir("./data/test")

    filepaths = []
    # for i,f in enumerate(filenames):
    #     # get path to folder containing the .npz Multitrack
    #     song_dir = args.input_dir / msd_id_to_dirs(f)
    #     _path = list(song_dir.glob("*.npz"))[0]

    #     # transform multitrack file for comparison
    #     multitrack = pypianoroll.load(_path)
    #     tracks = [track.pianoroll for track in multitrack.tracks]
        
    #     #if(any([t.shape[0] == 0 for t in tracks])):
    #     #  continue
        
    #     print([t.shape for t in tracks])
    #     print([t.shape[0] == 0 for t in tracks], end="\n\n")
    #     pianoroll_data = np.stack(tracks, axis=0)

    #     if pianoroll_data.shape == (5,6000,128): #multitrack.stack().shape == (5,6000,128):
    #       filepaths.append(_path)

    for i, f in enumerate(filenames):
      song_dir = args.input_dir / msd_id_to_dirs(f)
      npz_files = list(song_dir.glob("*.npz"))
      if not npz_files:
          print(f"No .npz found for {f} in {song_dir}")
          continue
      _path = npz_files[0]
      try:
          multitrack = pypianoroll.load(_path)
          tracks = [track.pianoroll for track in multitrack.tracks]
          shapes = [t.shape for t in tracks]

          if any(t.shape[0] == 0 for t in tracks):
              print(f"Skipping {f}: At least one track is empty. Shapes: {shapes}")
              continue
          if len(set(shapes)) != 1:
              print(f"Skipping {f}: Not all tracks have the same shape: {shapes}")
              continue

          pianoroll_data = np.stack(tracks, axis=0)
          if pianoroll_data.shape == (5, 6000, 128):
              filepaths.append(str(_path))
      except Exception as e:
          print(f"Error loading {f}: {e}")


    print(f"Found { len(filepaths) } files.")

    with open("./data/paths.npy", 'wb') as f:
      np.save(f, np.array(filepaths))

if __name__ == "__main__":
  main()
