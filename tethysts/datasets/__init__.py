import os


__all__ = ['available', 'get_path']

_module_path = os.path.dirname(__file__)
_available_dir = [p for p in next(os.walk(_module_path))[1]
                  if not p.startswith('__')]
_ts_dir_files = os.listdir(os.path.join(_module_path, 'time_series'))
_available_ts = {p.split('.')[0]: p for p in _ts_dir_files if p.endswith('.zst')}
available = list(_available_ts.keys())


def get_path(dataset):
    """
    Get the path to the data file.
    Parameters
    ----------
    dataset : str
        The name of the dataset.

    Returns
        str path
    """
    if dataset in _available_ts:
        return os.path.abspath(
            os.path.join(_module_path, 'time_series', _available_ts[dataset]))
    else:
        msg = "The dataset '{data}' is not available. ".format(data=dataset)
        msg += "Available datasets are {}".format(", ".join(available))
        raise ValueError(msg)
