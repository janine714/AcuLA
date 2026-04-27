import random
import numpy as np
import librosa
import augly.audio as audaugs
import torch
from torch.utils.data import Dataset
# from opera.src.util import get_entire_signal_librosa


class AudioTextPairDataset(Dataset):
    """Dataset for paired audio and text data"""
    def __init__(self, audio_paths, text_reports, 
            text_tokenizer, max_length=64, audio_input_sec=8):
        self.audio_paths = audio_paths
        self.text_reports = text_reports
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.target_audio_seconds = audio_input_sec
        self.sample_rate = 16000

    def _read_audio_sample(self, audio_path):
        waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
        return waveform

    def _pre_process_audio_mel_t(self, audio, 
        sample_rate=16000, n_mels=64, f_min=50, 
        f_max=8000, nfft=1024, hop=512):
        
        S = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=n_mels, fmin=f_min, 
            fmax=f_max, n_fft=nfft, hop_length=hop)

        # convert scale to dB from magnitude
        S = librosa.power_to_db(S, ref=np.max)
        if S.max() != S.min():
            mel_db = (S - S.min()) / (S.max() - S.min())
        else:
            mel_db = S
            print("warning in producing spectrogram!")

        return mel_db.T

    def apply_random_augmentation(self, audio):
        augmentations = [
            lambda audio: audaugs.change_volume(audio, volume_db=5.0)[0],
            lambda audio: audaugs.normalize(audio)[0],
            lambda audio: audaugs.low_pass_filter(audio, cutoff_hz=300)[0],
            lambda audio: audaugs.high_pass_filter(audio, cutoff_hz=3000)[0]
        ]
        augmentation = random.choice(augmentations)
        augmented_audio = augmentation(audio)

        target_length = int(self.target_audio_seconds * self.sample_rate)
        if len(augmented_audio) < target_length:
            augmented_audio = np.pad(
                augmented_audio,
                (0, target_length - len(augmented_audio)),
                mode="constant"
            )
        else:
            augmented_audio = augmented_audio[:target_length]

        return augmented_audio

    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        waveform = self._read_audio_sample(self.audio_paths[idx])
        target_length = int(self.target_audio_seconds * self.sample_rate) + 3000
        waveform = np.pad(
            waveform,
            (0, max(0, target_length - len(waveform))),
            mode='constant'
        )[:target_length]
    
        audio_data = self._pre_process_audio_mel_t(waveform.squeeze())

        text = self.text_reports[idx].lower()
        tokenized_text = self.text_tokenizer(text, 
                                max_length=self.max_length, 
                                padding="max_length", 
                                truncation=True,
                                return_tensors="pt")
        
        return {
            "audio": audio_data,
            "input_ids": tokenized_text.input_ids.squeeze(),
            "attention_mask": tokenized_text.attention_mask.squeeze(),
            # "text": text  # Keep raw text for reference
        }
