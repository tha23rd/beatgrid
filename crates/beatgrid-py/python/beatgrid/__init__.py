"""DJ-grade beat grids: the exact tempo, every beat on the attack, bar 1 on the one.

    >>> import beatgrid
    >>> r = beatgrid.analyze("track.mp3")
    >>> r["grid"]["bpm"], r["grid"]["first_downbeat"], r["review"]
"""

from ._beatgrid import __version__, analyze, analyze_many

__all__ = ["analyze", "analyze_many", "__version__"]
