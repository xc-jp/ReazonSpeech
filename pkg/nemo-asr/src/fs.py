import contextlib
import os
import tempfile
from typing import Generator


@contextlib.contextmanager
def win32_tempfile() -> Generator[tempfile._TemporaryFileWrapper, None, None]:
    tmpf = tempfile.NamedTemporaryFile(delete=False)
    try:
        yield tmpf
    finally:
        os.unlink(tmpf.name)  # noqa: PTH108


def create_tempfile() -> tempfile._TemporaryFileWrapper:
    if os.name == "nt":
        return win32_tempfile()
    else:
        return tempfile.NamedTemporaryFile()
