import json

class Namespace(object):
    def __init__(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)

class BackendDefinitions(object):
    def __init__(self):
        self.__hash = {}
    def __getitem__(self, backend):
        if not backend in self.__hash: 
            raise RuntimeError('Backend %s not available, or does not exist.'%backend)
        return Namespace(**self.__hash[backend])
    def __contains__(self, obj):
        return obj in self.__hash
    def define(self, backend, defintions):
        self.__hash[backend] = dict(defintions)


def dict_to_buffer(metadata):
    for k, v in metadata.items():
        if isinstance(v, set):
            metadata[k] = list(v)

    data = json.dumps(metadata).encode('utf-8')
    return data


def dict_from_buffer(b):
    metadata = json.loads(b.data.decode())
    return metadata
