import os
import fnmatch
from pathlib import Path
from typing import List

_VALID_EXTS = ['.mp3', '.wav', '.aif', '.aiff', '.flac', '.opus']

def audio_paths_from_dir(
        dir_path : str | Path,
        valid_exts: List[str] | None = None, 
        flt: List[str] = [],
        exclude: List[str] = []
    ):
    valid_exts = valid_exts or _VALID_EXTS
    valid_exts = list(map(lambda x: x.lower(), valid_exts)) + list(map(lambda x: x.upper(), valid_exts))
    audio_candidates = []
    base_dir = Path(dir_path)
    for ext in valid_exts:
        parsed_candidates = list(map(lambda x: x.resolve(), base_dir.glob(f'**/*{ext}')))
        if len(flt) > 0:
            filtered_candidates = []
            for f in flt: 
                filtered_candidates.extend(list(filter(lambda x, r = f: fnmatch.fnmatch(x.relative_to(base_dir), r), parsed_candidates)))
            parsed_candidates = list(set(filtered_candidates))
        for e in exclude:
            parsed_candidates = list(filter(lambda x, r = e: not fnmatch.fnmatch(x.relative_to(base_dir), r), parsed_candidates))
        audio_candidates.extend(map(str, parsed_candidates))
    return audio_candidates 