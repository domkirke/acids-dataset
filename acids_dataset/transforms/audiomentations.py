import audiomentations as ta
import gin.torch
import inspect
from .base import Transform
from . import misc
from ..utils import get_subclasses_from_package


__augmentations_to_import = list(filter(lambda x: x.get_class_fullname() not in dir(misc), 
                                        get_subclasses_from_package(ta.augmentations, ta.core.transforms_interface.BaseTransform) 
                            ))



def build_audiomentation_wrapper(cls): 
    class _AudiomentationWrapper(Transform):
        obj_class = cls
        takes_as_input = Transform.input_types.torch
        dont_export_to_gin_config = ["self", "name", "args", "kwargs", "sample_rate"]
        def __init__(self, *args, p=None, **kwargs):
            transform_args = {'sr': kwargs.get('sr'), 'name': kwargs.get('name')}
            super().__init__(p=None, **transform_args)
            if p is None: p = 1.
            aug_kwargs = self.get_cls_kwargs(self.obj_class, kwargs)
            if "sample_rate" in list(self.init_signature().parameters):
                aug_kwargs['sample_rate'] = kwargs.get('sr')
            self._obj = self.obj_class(*args, p=p, **aug_kwargs)
            
        def __repr__(self):
            return self._obj.__repr__()

        @classmethod
        def init_signature(cls):
            return inspect.signature(cls.obj_class.__init__)

        def get_cls_kwargs(self, aug_cls, kwargs):
            sig = inspect.signature(aug_cls.__init__)
            cls_kwargs = {}
            for param in sig.parameters:
                if param in kwargs:
                    cls_kwargs[param] = kwargs[param]
            return cls_kwargs

        def apply(self, x):
            return self._obj(x)
        
    new_class = type(obj.__name__, (_AudiomentationWrapper,), dict(_AudiomentationWrapper.__dict__))
    new_class = gin.configurable(new_class, module="transforms")
    return new_class


__all__ = []
for obj in __augmentations_to_import:
    wrapper = build_audiomentation_wrapper(obj)
    locals()[obj.get_class_fullname()] = wrapper
    __all__.append(obj.get_class_fullname())
    