# This file was adapted from https://github.com/radionets-project/pyvisgen/blob/main/pyvisgen/version.py

# this is adapted from https://github.com/astropy/astropy/blob/master/astropy/version.py
# see https://github.com/astropy/astropy/pull/10774 for a discussion on why this needed.
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
