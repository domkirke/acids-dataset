# features
from .base import AcidsDatasetFeature, check_feature_configs
from .regexp import RegexpFeature, append_meta_regexp, parse_meta_regexp
from .mel import Mel
from .loudness import Loudness
from .midi import AfterMIDI
from .module import *

# advanced operations
from .clustering import hash_from_clustering
