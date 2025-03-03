import os
import gin
from pathlib import Path
from .datasets import LMDBWriter
from .utils import checklist


@gin.configurable()
def get_writer_class(writer_class = LMDBWriter, filters=[], exclude=[]):
    gin.bind_parameter(f"{writer_class.__name__}.filters", gin.get_bindings(writer_class.__name__).get('filters',[]) + checklist(filters))
    gin.bind_parameter(f"{writer_class.__name__}.exclude", gin.get_bindings(writer_class.__name__).get('exclude', []) + checklist(exclude)) 
    return writer_class


@gin.configurable()
def append_features(features=None):
    if features is None: 
        return []
    else:
        return [f() for f in checklist(features)]


def get_default_output_path(path):
    return (Path(".") / f"{path.stem}_preprocessed").resolve().absolute()



def preprocess_dataset(
    path, 
    out = None, 
    configs = ['default.gin'], 
    check=False, 
    sample_rate = 41000,
    channels = 1,
    flt = [],
    exclude = []
):
    gin.add_config_file_search_path(path)
    gin.constant('SAMPLE_RATE', sample_rate)
    gin.constant('CHANNELS', channels)
    features = []
    for config in configs:
        if os.path.splitext(config)[1] == "": config += ".gin"
        if os.path.exists(config):
            gin.add_config_file_search_path(Path(config).parent)
        try:
            gin.parse_config_file(config)
        except TypeError as e:
            print('[error] problem parsing configuration %s'%config)
            raise e
        features.extend(append_features())
    path = Path(path)
    out = out or get_default_output_path(path)

    writer = get_writer_class(filters=flt, exclude=exclude)(path, out, features=features, check=check)
    writer.build()