import json
import os
from abc import abstractmethod
from glob import glob

import numpy as np
import soundfile
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

from os import remove as removefile

from .midi import parse_midi


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device='cpu', preload: bool = True, fixed_chunks: bool = False,
                 sample_rate: int = 16000, pianoroll_time_resolution: int = 32, onset_offset_time_tolerance: int = 32, 
                 max_midi: int = 108, min_midi: int = 21):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.preload = preload
        self.fixed_chunks = fixed_chunks
        self.sample_rate = sample_rate
        self.max_midi = max_midi
        self.min_midi = min_midi
        self.time_resolution = pianoroll_time_resolution  # time resolution in ms
        self.onset_offset_time_tolerance = onset_offset_time_tolerance
        hop_size, hops_in_onset, hops_in_offset = self._get_hop_sizes(self.time_resolution, self.onset_offset_time_tolerance, self.sample_rate)
        self.hop_size = hop_size
        self.hops_in_onset = hops_in_onset
        self.hops_in_offset = hops_in_offset

        self.data = []
        self.input_files = []

        for group in groups:
            for audio_tsv_file in self.files(group):
                self.input_files.append(audio_tsv_file)

        if preload:
            print(f"Preloading {len(groups)} group{'s' if len(groups) > 1 else ''} "
                f"of {self.__class__.__name__} at {path}")
            for audio_tsv_file in tqdm(self.input_files, desc='Loading group: %s' % group):
                    self.data.append(self.load(*audio_tsv_file))
        
        self.fixed_beginning_end_indexes = np.full((len(self.input_files), 2), -1, dtype=np.int16)

    def __getitem__(self, index):
        if self.preload:
            data = self.data[index]
        else:
            data = self.load(*self.input_files[index])
        
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            if self.fixed_chunks:
                step_begin, step_end = self.fixed_beginning_end_indexes[index]

                # if any of the indexes is negative, then we need to generate a new random chunk
                if (step_begin < 0) or (step_end < 0):
                        audio_length = len(data['audio'])
                        step_begin = self.random.randint(audio_length - self.sequence_length) // self.hop_size
                        n_steps = self.sequence_length // self.hop_size
                        step_end = step_begin + n_steps

                        begin = step_begin * self.hop_size
                        end = begin + self.sequence_length

                        self.fixed_beginning_end_indexes[index] = [step_begin, step_end] # save the indexes for later
                else: # If we already have a chunk, then we just need to get the audio data
                    begin = step_begin * self.hop_size
                    end = step_end * self.hop_size
            else: # If we are not using fixed chunks, then we just generate a random chunk
                audio_length = len(data['audio'])
                step_begin = self.random.randint(audio_length - self.sequence_length) // self.hop_size
                n_steps = self.sequence_length // self.hop_size
                step_end = step_begin + n_steps

                begin = step_begin * self.hop_size
                end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)

        return result

    def __len__(self):
        return len(self.input_files)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError
    
    def _get_hop_sizes(self, pianoroll_time_resolution, onset_offset_time_resolution, sample_rate):
        hop_size = sample_rate * pianoroll_time_resolution // 1000
        onset_length = sample_rate * onset_offset_time_resolution
        offset_length = onset_length
        hops_in_onset = onset_length // hop_size
        hops_in_offset = offset_length // hop_size
        return hop_size, hops_in_onset, hops_in_offset

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path):
            try:
                return torch.load(saved_data_path)
            except RuntimeError:
                removefile(saved_data_path)

        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == self.sample_rate

        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = self.max_midi - self.min_midi + 1
        n_steps = (audio_length - 1) // self.hop_size + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(round(onset * self.sample_rate / self.hop_size))
            onset_right = min(n_steps, left + self.hops_in_onset)
            frame_right = int(round(offset * self.sample_rate / self.hop_size))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + self.hops_in_offset)

            f = int(note) - self.min_midi
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='data/MAESTRO', groups=None, sequence_length=None, seed=42, device="cpu", preload=True, fixed_chunks=True):
        super().__init__(path, groups=groups if groups is not None else ['train'], sequence_length=sequence_length, seed=seed, device=device, preload=preload, fixed_chunks=fixed_chunks)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = json.load(open(os.path.join(self.path, "maestro-v3.0.0.json")))
            audios = [os.path.join(self.path, row.replace('.wav', '.flac')) for row in metadata["audio_filename"].values()]
            midifiles = [os.path.join(self.path, row) for row in metadata["midi_filename"].values()]
            splits = [row for row in metadata["split"].values()]
            files = list(zip(audios, midifiles, splits))
            files = filter(lambda x: x[2] == group, files)
            files = [(audio if os.path.exists(audio) else audio.replace(".flac", ".wac"), midi) for audio, midi, _ in files]

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, device="cpu", preload=True, fixed_chunks=True):
        super().__init__(path, groups=groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length=sequence_length, seed=seed, device=device, preload=preload, fixed_chunks=fixed_chunks)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
        tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]

        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(flacs, tsvs))
