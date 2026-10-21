"""Inspect executed Inductor Triton launchers during an untimed diagnostic call.

This uses private PyTorch runtime attributes. Unknown/unresolved selections are
reported explicitly. External library calls such as cuBLAS are outside its scope.
"""
from collections.abc import Mapping
from functools import wraps
from unittest.mock import patch


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _json_value(value):
    return value if value is None or isinstance(value, (str, bool, int, float)) else str(value)


def _describe(autotuner):
    fn = getattr(autotuner, 'fn', None)
    record = dict(kernel=getattr(fn, '__name__', type(autotuner).__name__),
                  source_file=getattr(autotuner, 'filename', None))
    launchers = getattr(autotuner, 'launchers', ())
    if len(launchers) != 1:
        return {**record, 'status': 'unresolved', 'launcher_count': len(launchers)}
    launcher = launchers[0]
    config = getattr(launcher, 'config', None)
    if config is None:
        return {**record, 'status': 'unresolved', 'reason': 'launcher has no config'}
    arg_names = getattr(fn, 'arg_names', ())
    constants = getattr(autotuner, 'triton_meta', {}).get('constants', {})
    parameters = {}
    for key, value in constants.items():
        name = arg_names[key] if isinstance(key, int) and 0 <= key < len(arg_names) else str(key)
        parameters[name] = _json_value(value)
    parameters.update({str(k): _json_value(v) for k, v in config.kwargs.items()})
    binary = getattr(launcher, 'bin', None)
    record.update(status='selected', parameters=parameters,
                  num_warps=getattr(config, 'num_warps', None),
                  num_stages=getattr(config, 'num_stages', None),
                  n_regs=getattr(launcher, 'n_regs', getattr(binary, 'n_regs', None)),
                  n_spills=getattr(launcher, 'n_spills', getattr(binary, 'n_spills', None)),
                  shared_bytes=getattr(launcher, 'shared',
                                       _field(getattr(binary, 'metadata', None), 'shared')))
    return record


def capture_selected_configs(fn):
    """Return (output, selected launch records), restoring the runtime on exit.

    Call only after compilation/tuning and outside measurement/capture. Observe
    post-dispatch launchers rather than config candidates, so cache hits work too.
    Repeated executions of the same runtime object produce one record.
    """
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    original = CachingAutotuner.run
    records = {}

    @wraps(original)
    def observed(autotuner, *args, **kwargs):
        result = original(autotuner, *args, **kwargs)
        records[id(autotuner)] = _describe(autotuner)
        return result

    with patch.object(CachingAutotuner, 'run', observed):
        output = fn()
    return output, list(records.values())
