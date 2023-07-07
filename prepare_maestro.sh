#! /bin/bash
# set -e

USAGE="Usage: $0 -s <path/to/where/the/data/will/be/stored> [-m <path/to/existing/MAESTRO/dir/]"

# Check the arguments
download_dataset="True"
while getopts ":s:m:h:" opt; do
    case "${opt}" in
        s)
            DATA_DIR=$OPTARG
            # Delete the older ./data/ symbolic link if exists
            if [ -L "./data" ]; then
                echo "./data" symbolic link exists already. If it is outdated delete it manually and if needed rerun this script.
            fi
            ;;
        m)
            echo "Using the MAESTRO dataset in $OPTARG. It is now being copied to $DATA_DIR/MAESTRO"
            rsync -r `echo $OPTARG | sed 's![^/]$!&/!'` ./data/MAESTRO  # We use sed to add a trailing / to the m) path if missing.
            download_dataset="False"
            ;;
        h)
            echo $USAGE
            ;;
        *)
            echo "Invalid option, must be either s, m or h"
    esac
done

# Check if ffmpeg is installed, and if its not install it depending on the OS
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg could not be found"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Installing ffmpeg ..."
        sudo apt-get install ffmpeg
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Installing ffmpeg ..."
        brew install ffmpeg
    else
        echo "Please install ffmpeg manually."
        exit 1
    fi
fi

# Get the directory where the data will be stored, if not specified, throw an error.
if [ -z "$DATA_DIR" ]; then
    echo "Please specify the directory where the data will be downloaded."
    echo $USAGE
    exit 1
fi

# Check if the directory exists already
if [ -d "$DATA_DIR/MAESTRO" && $download_dataset = "False" ]; then
    echo "The directory MAESTRO in $DATA_DIR already exists."
    echo "Please specify a new directory."
    exit 1
else
    mkdir $DATA_DIR
fi


# Create a symbolic link to the data directory
ln -s $DATA_DIR ./data
echo "[WARNING] ./data/ is now a symbolic link to $DATA_DIR. If this directory is an external drive it might change its location in the future."
echo "If you have problems with the data in the future, please check if the symbolic link is still valid."



# If the variable download_dataset is True, Download the data inside the symbolic link
if [ $download_dataset = "True" ]; then 
    echo "Downloading the MAESTRO dataset"
    curl -O "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip" -o ./data/maestro-v3.0.0.zip

    echo "Extracting the files ..."
    unzip -o ./data/maestro-v3.0.0.zip | awk 'BEGIN{ORS=""} {print "\rExtracting " NR "/2383 ..."; system("")} END {print "\ndone\n"}'

    rm ./data/maestro-v3.0.0.zip
    mv ./data/maestro-v3.0.0 ./data/MAESTRO
fi

echo Converting the audio files to FLAC ...
COUNTER=0
for f in ./data/MAESTRO/*/*.wav; do
    COUNTER=$((COUNTER + 1))
    echo -ne "\rConverting ($COUNTER/1184) ..."
    ffmpeg -y -loglevel fatal -i $f -ac 1 -ar 16000 ${f/\.wav/.flac}
    rm $f
done

echo
echo Preparation complete!
