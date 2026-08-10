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

For development work, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/guillochon/MOSFiT.git
    cd MOSFiT
    pip install -e .
