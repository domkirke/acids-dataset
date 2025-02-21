import enum
from absl import logging
from typing import List, Callable
import math
import torch, torchaudio
from dataclasses import dataclass
from pathlib import Path
import gin
from .utils import FileNotReadException
from ..utils import loudness


class PadMode(enum.Enum):
    DISCARD = 0
    ZERO_PAD = 1
    REPEAT = 2
    REPLICATE = 3
    REFLECT = 4

class TorchaudioBackend(object):
    @classmethod
    def parse_file(
        cls,
        audio_path, 
        chunk_length, 
        hop_length, 
        pad_mode,
        sr,
        channels,
        bformat,
        loudness_threshold 
    ) -> List[Callable]:
        def _load_file(fragment_class):
            try:
                wav, orig_sr = torchaudio.load(audio_path)
            except RuntimeError: 
                raise FileNotReadException(audio_path, cls)
            if orig_sr != sr:
                wav = torchaudio.functional.resample(wav, orig_sr, sr)
            if wav.shape[0] > channels:
                wav = wav[:channels]
            elif wav.shape[0] < channels: 
                wav = wav[torch.arange(channels)%wav.shape[0]]
            chunk_length_smp = int(chunk_length * sr)
            hop_length_smp = int(hop_length * sr)
            n_chunks = math.ceil(wav.shape[-1] / hop_length_smp)
            start_pos = [i*hop_length_smp for i in range(n_chunks)]
            wav = [wav[..., start_pos[i]:start_pos[i]+chunk_length_smp] for i in range(n_chunks)]
            fragments = []
            for i in reversed(range(len(wav))):
                if wav[i].shape[-1] != chunk_length_smp:
                    padded_chunk = cls.pad_chunk(wav[i], chunk_length_smp, pad_mode)
                    if padded_chunk is None: 
                        del wav[i]
                        continue
                    else:
                        wav[i] = padded_chunk
                
                
                if loudness_threshold is not None:
                    chunk_loudness = loudness(wav[i], sr)
                    if chunk_loudness < loudness_threshold:
                        del wav[i]
                        continue
                current_fragment = fragment_class(
                        audio_path = audio_path, 
                        audio = None, 
                        start_pos = start_pos[i] / sr,
                        bformat = bformat
                    )
                fragments.insert(0, current_fragment)
            return fragments 
        return [_load_file]

    @classmethod
    def pad_chunk(
        cls, 
        chunk,
        target_size,
        pad_mode
    ):
        if pad_mode == PadMode.DISCARD:
            return None
        elif pad_mode == PadMode.ZERO_PAD:
            return torch.nn.functional.pad(chunk, (0, target_size - chunk.shape[-1]), mode="constant", value=0.)
        elif pad_mode == PadMode.REPEAT:
            n_iter = 0
            while chunk.shape[-1] < target_size:
                if n_iter > 1: 
                    logging.warning('applying repeat pad more than once ; may provoke undesired behaviour.')
                chunk = torch.nn.functional.pad(chunk, (0, min(target_size - chunk.shape[-1], chunk.shape[-1] - 1)), mode="circular")
                n_iter += 1
            return chunk
        elif pad_mode == PadMode.REPLICATE:
            return torch.nn.functional.pad(chunk, (0, target_size - chunk.shape[-1]), mode="replicate")
        elif pad_mode == PadMode.REFLECT:
            n_iter = 0
            while chunk.shape[-1] < target_size:
                if n_iter > 1: 
                    logging.warning('applying reflect pad more than once ; may provoke undesired behaviour.')
                chunk = torch.nn.functional.pad(chunk, (0, min(target_size - chunk.shape[-1], chunk.shape[-1] - 1)), mode="reflect")
                n_iter += 1
            return chunk
        else:
            raise ValueError('pad mode %s not recognized.'%pad_mode)
        

class FFMPEGBackend(object):
    @classmethod
    def parse_file(
        cls,
        path, 
        chunk_length, 
        hop_length, 
        pad_mode,
        sr,
        channels,
        bformat,
        loudness_threshold 
    ) -> List[Callable]:
        raise NotImplementedError()

    @classmethod
    def pad_chunk(
        cls, 
        chunk,
        target_size,
        pad_mode
    ):
        raise NotImplementedError()


class ImportBackend(enum.Enum):
    TORCHAUDIO = 0
    # FFMPEG = 1

    @property
    def backend(self):
        if self == ImportBackend.TORCHAUDIO:
            return TorchaudioBackend
        # elif self == ImportBackend.FFMPEG:
        #     return FFMPEGBackend
        else:
            raise ValueError('No backend for %s'%self)

@dataclass
class ParsingRequest():
    filepath: str
    start: float 
    end: float 
    duration: float
    sr: int
    pad_mode: PadMode


@gin.configurable(module="parser")
class RawParser(object):
    def __init__(
            self, 
            audio_path: str,
            chunk_length: int | float, 
            sr: int, 
            channels: int, 
            hop_length: int | float | None = None,
            overlap: float | None = None, 
            pad_mode: PadMode | str = PadMode.DISCARD,
            import_backend: ImportBackend | str = ImportBackend.TORCHAUDIO,
            bformat: str = "int16", 
            loudness_threshold: float | None = None
        ):
        self.audio_path = audio_path 
        assert Path(self.audio_path).exists(), f"{self.audio_path} does not seem to exist. Please provide a valid file"
        if isinstance(chunk_length, int):
            chunk_length = chunk_length / sr
        self.chunk_length = float(chunk_length)
        assert self.chunk_length > 0, "chunk_length must be positive"
        self.sr = int(sr)
        assert self.sr > 0, "sr must be positive" 
        if (hop_length is not None) and (overlap is None):
            assert hop_length > 0, "hop_length must be positive"
            if isinstance(hop_length, int):
                hop_length = hop_length / self.sr
            self.hop_length = hop_length
        elif (hop_length is None) and (overlap is not None):
            self.hop_length = (1 - overlap) * chunk_length
        elif (hop_length is None) and (overlap is None):
            self.hop_length = chunk_length
        else:
            raise ValueError("overlap and overlap_ratio must not be given at the same time.")
        if isinstance(pad_mode, str):
            pad_mode = getattr(PadMode, pad_mode.upper())
        self.pad_mode = pad_mode
        self.bformat = bformat
        if isinstance(import_backend, str):
            import_backend = getattr(ImportBackend, import_backend.upper())
        self.import_backend = import_backend
        self.loudness_threshold = loudness_threshold
        self.channels = channels

    @property
    def chunk_length_smp(self):
        return int(self.chunk_length * self.sr)
    
    @property
    def hop_length_smp(self):
        return int(self.hop_length * self.sr)
    
    def get_metadata(self):
        return {
            'chunk_length': self.chunk_length, 
            'hop_length': self.hop_length, 
            'import_backend': self.import_backend.value,
            'pad_mode': self.pad_mode.value, 
            'format': self.bformat,
            'sr': self.sr, 
            'channels': self.channels,
            'loudness_threshold': self.loudness_threshold
        }

    def __iter__(self):
        parsing_method = self.import_backend.backend.parse_file
        return iter(parsing_method(
            audio_path=self.audio_path, 
            chunk_length=self.chunk_length, 
            hop_length=self.hop_length, 
            pad_mode=self.pad_mode,
            sr=self.sr,
            channels=self.channels,
            bformat=self.bformat,
            loudness_threshold=self.loudness_threshold
        ))


__all__ = ['RawParser', "PadMode", "ImportBackend", "FileNotReadException"]




