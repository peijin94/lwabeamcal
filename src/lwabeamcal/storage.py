"""Atomic local products; never overwrite an existing target implicitly."""
import hashlib
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager


def sha256_file(path):
    h=hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


@contextmanager
def atomic_target(path, overwrite=False):
    path=Path(path)
    if path.is_symlink(): raise ValueError('refusing symlink output')
    if path.exists() and not overwrite: raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name=tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    os.close(fd)
    temp=Path(name)
    try:
        yield temp
        if overwrite:
            if path.is_symlink(): raise ValueError('refusing symlink output')
            os.replace(temp, path)
        else:
            # Atomic no-clobber commit: a competing writer cannot be overwritten.
            os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
