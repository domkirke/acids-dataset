from pathlib import Path
import gin
import json, pickle
import numpy as np
from . import check_compiled_proto
from typing import Literal, Optional, Any
from .utils import dict_from_buffer, dict_to_buffer
from ..utils import get_backend


RaveExampleClass = check_compiled_proto(__file__)


DTYPE_TO_PRECISION = {
    np.int16: RaveExampleClass.Precision.INT16,
    np.int32: RaveExampleClass.Precision.INT32,
    np.int64: RaveExampleClass.Precision.INT64,
    np.float16: RaveExampleClass.Precision.FLOAT16,
    np.float32: RaveExampleClass.Precision.FLOAT32,
    np.float64: RaveExampleClass.Precision.FLOAT64,
}

PRECISION_TO_DTYPE = {
    RaveExampleClass.Precision.INT16: np.int16,
    RaveExampleClass.Precision.INT32: np.int32,
    RaveExampleClass.Precision.INT64: np.int64,
    RaveExampleClass.Precision.FLOAT16: np.float16,
    RaveExampleClass.Precision.FLOAT32: np.float32,
    RaveExampleClass.Precision.FLOAT64: np.float64,
}

@gin.configurable(module="fragments")
class AcidsFragment(object):

    def __init__(
            self,
            byte_string: Optional[str] = None,
            audio_path: Optional[str] = None, 
            audio: Optional[Any] = None,
            start_pos: Optional[float] = None, 
            bformat: Optional[str] = None,
            output_type: Literal["numpy", "torch", "jax"] = "numpy") -> None:
        if byte_string is not None:
            self.ae = RaveExampleClass.FromString(byte_string)
        else:
            self.ae = RaveExampleClass()
            if audio is not None:
                self.put_buffer("waveform", audio, audio.shape)
            metadata = {}
            if audio_path is not None: metadata['audio_path'] = audio_path
            if start_pos is not None: metadata['start_pos'] = str(start_pos)
            self.set_metadata(metadata)

    def serialize(self):
        return self.ae.SerializeToString()

    def get(self, key: str):

        if key == "midi":
            midi = self.ae.buffers["midi"].data
            midi = pickle.loads(midi)
            return midi

        buf = self.ae.buffers[key]
        if buf is None:
            raise KeyError(f"key '{key}' not available")

        array = np.frombuffer(
            buf.data,
            dtype=PRECISION_TO_DTYPE[buf.precision],
        ).reshape(buf.shape).copy()

        if buf.precision == RaveExampleClass.Precision.INT16:
            array = array.astype(np.float32) / (2**15 - 1)

        if self.output_type == "numpy":
            pass
        elif self.output_type == "jax":
            jnp = get_backend('jax')
            array = jnp.array(array)
        elif self.output_type == "torch":
            torch = get_backend('torch')
            array = torch.from_numpy(array)
        else:
            raise ValueError(f"Output type {self.output_type} not available")

        return array

    def set_metadata(self, metadata: dict):
        meta_buffer = self.ae.buffers["metadata"]
        meta_buffer.data = dict_to_buffer(metadata)

    def update_metadata(self, **kwargs):
        meta_buffer = self.ae.buffers["metadata"]
        meta_buffer = dict_from_buffer(meta_buffer)
        meta_buffer.update(kwargs)
        meta_buffer.data = dict_to_buffer(meta_buffer)

    def get_metadata(self):
        buf = self.ae.buffers["metadata"]
        return dict_from_buffer(buf)

    def put_buffer(self, key: str, b: bytes, shape: list):
        buffer = self.ae.buffers[key]
        buffer.data = b
        if shape is not None:
            buffer.shape.extend(shape)
        # why? 
        buffer.precision = RaveExampleClass.Precision.INT16

    def put_array(self, key: str, array: np.ndarray, dtype: np.dtype):
        buffer = self.ae.buffers[key]
        buffer.data = np.asarray(array).astype(dtype).tobytes()
        for i in range(len(buffer.shape)):
            buffer.shape.pop()
        buffer.shape.extend(array.shape)
        buffer.precision = DTYPE_TO_PRECISION[dtype]

    def as_dict(self):
        return {k: self.get(k) for k in self.ae.buffers}

    def __str__(self) -> str:
        repr = []
        repr.append("AudioFragment(")
        for key in self.ae.buffers:
            if key == "metadata":
                repr.append(str(self.get_metadata()))
            else:
                array = self.get(key)
                repr.append(f"\t{key}[{array.dtype}] {array.shape},")
        repr.append(")")
        return "\n".join(repr)

    def __bytes__(self) -> str:
        return self.ae.SerializeToString()

