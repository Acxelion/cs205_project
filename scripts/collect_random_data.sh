#!/bin/bash
# This script collect training data from a given directory by looking
# for all the files in that directory that end wih ".mid" and converting
# them to a five-track pianoroll dataset.
# Usage: ./generate_data.sh [INPUT_DIR] [OUTPUT_FILENAME]
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
python3.7 "$DIR/../src/random_select.py" -i "$1" -l "$2" -o "$3" -k "$4"
