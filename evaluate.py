import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from tqdm import tqdm

from typing import Optional, Iterable

import pandas as pd

import onsets_and_frames.dataset as dataset_module
from onsets_and_frames import *

from sklearn.metrics import auc

eps = sys.float_info.epsilon


def evaluate(data, model, onset_threshold: float = 0.5, frame_threshold: float = 0.5, save_path=None, pr_au_thresholds: Optional[Iterable[float]] = np.arange(0.1, 0.9, 0.05)):
    metrics = defaultdict(list)

    for label in data:
        pred, losses = model.run_on_batch(label)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        for key, value in pred.items():
            value.squeeze_(0).relu_()

        p_ref, i_ref, v_ref = extract_notes(label['onset'], label['frame'], label['velocity'])
        p_est, i_est, v_est = extract_notes(pred['onset'], pred['frame'], pred['velocity'], onset_threshold, frame_threshold)

        t_ref, f_ref = notes_to_frames(p_ref, i_ref, label['frame'].shape)
        t_est, f_est = notes_to_frames(p_est, i_est, pred['frame'].shape)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
        t_est = t_est.astype(np.float64) * scaling
        f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                  offset_ratio=None, velocity_tolerance=0.1)
        metrics['metric/note-with-velocity/precision'].append(p)
        metrics['metric/note-with-velocity/recall'].append(r)
        metrics['metric/note-with-velocity/f1'].append(f)
        metrics['metric/note-with-velocity/overlap'].append(o)

        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
        metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
        metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
        metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
        metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

        for key, loss in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            label_path = os.path.join(save_path, os.path.basename(label['path']) + '.label.png')
            save_pianoroll(label_path, label['onset'], label['frame'])
            pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.png')
            save_pianoroll(pred_path, pred['onset'], pred['frame'])
            midi_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.mid')
            save_midi(midi_path, p_est, i_est, v_est)

        if pr_au_thresholds is not None:
            p_frames = np.zeros_like(pr_au_thresholds)
            r_frames = np.zeros_like(pr_au_thresholds)
            for idx, threshold in enumerate(pr_au_thresholds):
                p_est, i_est, v_est = extract_notes(pred['onset'], pred['frame'], pred['velocity'], threshold, threshold)
                t_est, f_est = notes_to_frames(p_est, i_est, pred['frame'].shape)

                i_est = (i_est * scaling).reshape(-1, 2)
                p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])
                t_est = t_est.astype(np.float64) * scaling
                f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

                frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
                p_frames[idx] = frame_metrics['Precision']
                r_frames[idx] = frame_metrics['Recall']
            metrics['metric/frame/pr-auc'].append((auc(r_frames, p_frames)))

    return metrics


@torch.no_grad()
def evaluate_file(model_file, dataset, dataset_group, sequence_length, save_path,
                  onset_threshold, frame_threshold, device, output_dir: Optional[str] = None):
    dataset_class = getattr(dataset_module, dataset)
    kwargs = {'sequence_length': sequence_length, 'device': device}
    if dataset_group is not None:
        kwargs['groups'] = [dataset_group]
    dataset = dataset_class(**kwargs)

    model = torch.load(model_file, map_location=device).eval()
    summary(model)

    if output_dir:
        os.makedirs(output_dir, exist_ok=False)
        df = pd.DataFrame(columns=['category', 'name', 'mean', 'std'])
        csv_file = os.path.join(output_dir, 'metrics.csv')
        summary_file = os.path.join(output_dir, 'model.txt')
        with open(summary_file, "w") as f:
            summary(model=model, file=f)

    metrics = evaluate(tqdm(dataset), model, onset_threshold, frame_threshold, save_path)

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
            if output_dir:
                df = pd.concat([df, pd.DataFrame([{'category': category, 'name': name, 'mean': np.mean(values), 'std': np.std(values)}])], ignore_index=True)
    if output_dir:
        df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('dataset', nargs='?', default='MAPS')
    parser.add_argument('dataset_group', nargs='?', default=None)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default=None)

    with torch.inference_mode():
        evaluate_file(**vars(parser.parse_args()))
