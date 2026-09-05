# Vendored wheels

`astrocats` 0.3.37's `setup.cfg` sets `python-tag = universal`, which produces a
`universal-none-any` wheel. `uv` rejects that tag on Python 3.14.

This wheel is the same 0.3.37 sdist rebuilt with `python-tag = py3`
(`astrocats-0.3.37-py3-none-any.whl`).
