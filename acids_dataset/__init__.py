from pathlib import Path
import gin
gin.add_config_file_search_path(Path(__file__).parent / "configs")
gin.add_config_file_search_path(Path(__file__).parent.parent / "custom_configs")

from . import utils
from . import transforms
from . import features
from . import fragments

def get_fragment_class(class_name):
    return getattr(fragments,class_name) 

from . import datasets