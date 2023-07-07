# PyTorch code for _TriAD_: Capturing harmonics with 3D convolutions

This is the code accompayning **TriAD: Capturing harmonics with 3D convolutions**. 
It is mostly based on Jong Wook's [repository](https://github.com/jongwook/onsets-and-frames).

### Downloading Dataset
You need to get two datasets: Maestro and Maps. Maestro is hosted in Google's servers, and you can download it and
parse it using the `prepare_maestro.sh` script.
When calling the script, use `-s` to indicate where will be Maestro downloaded; a symbolic link `data/MAESTRO` will be
created pointing at the location where maestro was downloaded & unzipped. 
It will also take care resampling and encoding the files as FLAC.

In case you have Maestro already in your computer, you can just use the bash script in Jong Wook's 
[repository](https://github.com/jongwook/onsets-and-frames).

To obtain the MAPS dataset just download it from Jong Wook's 
[repository](https://github.com/jongwook/onsets-and-frames), and place it in data/MAPS

### Training

All package requirements are contained in `requirements.txt`. To train the model, run:

```bash
pip install -r requirements.txt
python train.py
```

`train.py` is written using [sacred](https://sacred.readthedocs.io/), and accepts configuration options such as:

```bash
python train.py with logdir=runs/model iterations=1000000
```

Trained models will be saved in the specified `logdir`, otherwise at a timestamped directory under `runs/`.

### Testing

To evaluate the trained model using the MAPS database, run the following command to calculate the note and frame metrics:

```bash
python evaluate.py <path/to/your/saved/model>
```

Specifying `--save-path` will output the transcribed MIDI file along with the piano roll images:

```bash
python evaluate.py <path/to/your/saved/model> --save-path output/
```

In order to test on the Maestro dataset's test split instead of the MAPS database, run:

```bash
python evaluate.py <path/to/your/saved/model> MAESTRO test
```

## Citing

Please, if you use this repository or the model consider citing:
```text
@inproceedings{Perez2023triad,
  author       = {Perez, Miguel and Kirchhoff, Holger and Serra, Xavier}
  title        = {TriAD: Capturing harmonics with 3D convolutions},
  booktitle    = {Proceedings of the 24th International Society for Music Information
                  Retrieval Conference, {ISMIR} 2023, Milan, November 5-9, 2023},
}
```

