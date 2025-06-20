# This file was adapted from:
#   - pyvisgen (https://github.com/radionets-project/pyvisgen/blob/main/pyvisgen/version.py)
#     Originally licensed under MIT License. Copyright (c) 2021 radionets-project.
#   - astropy (https://github.com/astropy/astropy/blob/main/astropy/version.py)
#     Originally licensed under BSD-3-Clause license. Copyright (c) 2011-2024, Astropy Developers.

try:
    try:
        from ._dev_version import version
    except ImportError:
        from ._version import version
except Exception:
    import warnings

    warnings.warn(
        "Could not determine simtools version. This indicates"
        " a broken installation. Please install simtools from"
        " the local git repository."
    )
    del warnings
    version = "0.0.0"

__version__ = version
