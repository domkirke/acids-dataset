import re, os, sys
from pathlib import Path
import importlib
PROTO_DIR = Path(__file__).parent / "interfaces"
COMPILED_DIR = Path(__file__).parent / "compiled"


def is_message_class(obj_class):
    if not isinstance(obj_class, type): return False
    if not hasattr(obj_class, "DESCRIPTOR"): return False
    return True

def check_compiled_proto(filepath):
    module_name = Path(filepath).stem
    proto_path = PROTO_DIR / f"{module_name}.proto"
    compiled_name = rf"{module_name}_pb\d+"
    compiled_path = COMPILED_DIR / f"{compiled_name}.py"
    valid_compiled_files = list(filter(lambda x: x is not None, [re.match(compiled_name, f) for f in os.listdir(COMPILED_DIR)]))
    if len(valid_compiled_files) == 0:
        # try to compile buffer
        compilation_command = f"python -m grpc_tools.protoc -I=\"{PROTO_DIR}\" --python_out=\"{COMPILED_DIR}\"  {proto_path}"
        result = os.system(compilation_command)
        if result != 0:
            raise ImportError(f'Got problem with compilation command: \n{compilation_command}. Error code : {result}')
        valid_compiled_files = list(filter(lambda x: x is not None, [re.match(compiled_name, f) for f in os.listdir(COMPILED_DIR)]))[0]
    compiled_name = valid_compiled_files[0].string
    # module = importlib.import_module(f"compiled.{compiled_path}")
    module_spec = importlib.util.spec_from_file_location(compiled_name, COMPILED_DIR / compiled_name)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if is_message_class(obj):
            return obj
    raise ImportError(f'Could not import interface for module {filepath}')

from .acids import *