import re


fragment_interfaces = {
    'audio': 'get_audio', 
    'array': 'get_array', 
    "buffer": 'get_buffer',
    None: 'get_audio'}

def _from_fragment(fragment, item):
    item = item.split(':')
    if len(item) == 1:
        item, method_type = item[0], None
    elif len(item) == 2:
        item, method_type = item
    else:
        raise ValueError('invalid item found : %s'%item)
    return getattr(fragment, fragment_interfaces[method_type])(item)

def _outs_from_pattern(fragment, output_pattern):
    re_result = re.match(r'^(\w+)$', output_pattern)
    if re_result is not None:
        return _from_fragment(fragment, output_pattern)
    re_result = re.match(r'^\(?([\w\,\:]+)\)?$', output_pattern)
    if re_result is not None:
        items = list(filter(lambda x: x != "", re_result.groups()[0].split(',')))
        items = [_from_fragment(fragment, i) for i in items]
        return tuple(items)
    re_result = re.match(r'^\{([\w\,\:]+)\}$', output_pattern)
    if re_result is not None:
        items = list(filter(lambda x: x != "", re_result.groups()[0].split(',')))
        items = {i: _from_fragment(fragment, i) for i in items}
        return items

def _transform_outputs(outs, transforms):
    if isinstance(outs, (list, tuple)):
        outs = list(*outs)
        assert isinstance(transforms, (tuple, list))
        for i, t in enumerate(transforms):
            outs[i] = t(outs[i])
    elif isinstance(outs, dict):
        assert isinstance(transforms, dict)
        for k, t in transforms.items():
            assert k in outs
            outs[k] = t(outs[k])
    else:
        if isinstance(transforms, (list, tuple)):
            if len(transforms) > 0:
                outs = transforms[0](outs)
        else:
            if transforms is not None:
                outs = transforms(outs)
    return outs

        