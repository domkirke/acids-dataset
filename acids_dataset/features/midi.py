import os
from pathlib import Path
from typing import Any
from .base import AcidsDatasetFeature, FeatureException
from ..utils import load_file
from ..transforms.pitch import BasicPitch
import gin
import torch
import pretty_midi

MIDI_EXTS = [".mid", ".midi"]

def get_midi_from_folder(folder):
    local_files = os.path.listdir(folder)
    midi_files = list(filter(lambda x: os.path.splitext(x)[1].lower() in MIDI_EXTS), folder)
    midi_files = [os.path.join(folder, x) for x in midi_files]
    return midi_files

def get_midi_from_candidates(midi_paths, original_name=None):
    if len(midi_paths) == 0:
        return None
    elif len(midi_paths) == 1:
        return midi_paths[0]
    else:
        if original_name is not None:
            for f in midi_paths:
                if os.path.splitext(os.path.basename(f))[0] == original_name:
                    return f
        raise FeatureException("Several candidates found for original name %s."%original_name)

def find_candidates(folder, original_name=None):
    midi_files = get_midi_from_folder(folder)
    midi_file = get_midi_from_candidates(midi_files, original_name=original_name)
    return midi_file

    

@gin.configurable(module="features", denylist=AcidsDatasetFeature.denylist)
class MIDIExtractor(AcidsDatasetFeature):
    # dictionary of BasicPitch instances, referenced by devices, shared across instances.
    bp = {} 

    def __init__(self, audio_path: str | None, allow_basic_pitch: bool = True, relative_midi_path: str | None = None, device: torch.device = None,
                 **kwargs):
        super().__init__(audio_path=audio_path, **kwargs)
        self.midi_path = self.get_midi_path_candidates(audio_path, relative_midi_path=relative_midi_path)
        if self.midi_path is None:
            if audio_path is None: raise FeatureException('no candidates for MIDI files, and audio_path is None')
            self._extract_with_basic_pitch(self.audio_path, device=device)
        else:
            self._extract_from_file(self.midi_path)

    @classmethod
    def _get_bp_instance(cls, device):
        if cls.bp.get(device) is None:
            cls.bp[device] = BasicPitch(device=device)
        return cls.bp[device]

    def _extract_from_basic_pitch(self, audio_path, device=None):
        device = device or self.device
        bp = MIDIExtractor._get_bp_instance(device)
        audio_data = self.audio_data or self.load_file(path=audio_path)
        return bp(audio_data)

    def _extract_from_file(self, midi_path):
        return pretty_midi.PrettyMIDI(midi_path)

    def get_midi_path(self, path, relative_midi_path=None):
        path = Path(path)
        if relative_midi_path:
            midi_path = find_candidates(path / relative_midi_path)
            if midi_path is not None: 
                return midi_path
        midi_path = get_midi_from_folder(path.parent)
        if midi_path is not None: return midi_path
        for p in self.searchable_midi_paths:
            if (path.parent / p).exists():
                midi_path = get_midi_from_folder(path.parent / p)
                if midi_path is not None: return midi_path
        return None

        

        
          

        