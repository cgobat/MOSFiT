.. _installation:

============
Installation
============

Several installation methods for ``MOSFiT`` are outlined below. If you run into
issues, `open a new issue <https://github.com/guillochon/mosfit/issues>`_ on
GitHub.

.. _anaconda:

-------------------------------
Setting up MOSFiT with Anaconda
-------------------------------

**Platforms:** macOS, Linux, and Windows

We recommend using the `Anaconda <https://www.anaconda.com/download>`__ or
Miniconda Python distribution as your Python environment.

After installing conda, ``MOSFiT`` can be installed via:

.. code-block:: bash

    conda install -c conda-forge mosfit

.. _pip:

-------------------
Installing with pip
-------------------

**Platforms:** macOS, Linux, and Windows

Installing ``MOSFiT`` with pip is straightforward:

.. code-block:: bash

    pip install mosfit

.. _source:

----------------------
Installing from source
----------------------

**Platforms:** macOS, Linux, and Windows

For development work, clone the repository and sync the project with
`uv <https://docs.astral.sh/uv/>`__ (Python 3.11 or newer):

.. code-block:: bash

    git clone https://github.com/guillochon/MOSFiT.git
    cd MOSFiT
    uv sync

This creates a local virtual environment in ``.venv``. Run MOSFiT with
``uv run mosfit ...``, or activate the environment and call ``mosfit``
directly.

MPI support is optional (``mpi4py`` needs a system MPI library):

.. code-block:: bash

    uv sync --extra mpi

The SESN SEDONA emulator (``sesn_sedona``) needs PyTorch, which is an
optional extra:

.. code-block:: bash

    uv sync --extra sedona

For a published pip install, use ``pip install 'mosfit[sedona]'``. Default
models that do not use this SED do not require PyTorch.
