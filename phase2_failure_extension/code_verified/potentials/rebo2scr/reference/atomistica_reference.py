"""Authoritative reference implementation: Atomistica's Rebo2Scr (Fortran) via its ASE interface."""
from __future__ import annotations

import importlib.metadata

REFERENCE_INFO = {
    "implementation": "Atomistica Rebo2Scr (Fortran, screened REBO2)",
    "version": None,
    "source": "https://github.com/Atomistica/atomistica",
    "source_commit_reviewed": "14a86f2c79788d30aa678ae013f1187f5edcc51e (2025-10-21)",
    "python_call": "atomistica.Rebo2Scr()  # defaults: dihedral=False, Cmin=1, Cmax=2",
}


def available():
    try:
        import atomistica  # noqa
        from atomistica import Rebo2Scr  # noqa
        return True
    except Exception:
        return False


def version():
    try:
        return importlib.metadata.version("atomistica")
    except Exception:
        return None


def make_reference_calculator(**kwargs):
    """Return the authoritative Atomistica Rebo2Scr ASE calculator (default parameters)."""
    from atomistica import Rebo2Scr
    return Rebo2Scr(**kwargs)


REFERENCE_INFO["version"] = version()
