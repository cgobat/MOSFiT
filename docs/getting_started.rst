.. _getting-started:

===============
Getting started
===============
.. toctree::

Once installed, ``MOSFiT`` can be run from any directory, and it's typically convenient to make a new directory for your project.

.. code:: bash

    mkdir mosfit_runs
    cd mosfit_runs

``MOSFiT`` can be invoked either via :code:`python -m mosfit` or simply :code:`mosfit`. Pass **paths** to catalog-format JSON (or ASCII for conversion) with ``-e``:

.. code-block:: bash

    mosfit -e ./LSQ12dlf.json

The above command will prompt the user to choose a model (of those distributed with ``MOSFiT``) to fit against the data, using the event's claimed type to provide a list of suggested models. A specific model can be specified with ``-m``:

.. code-block:: bash

    mosfit -e ./LSQ12dlf.json -m slsn

Multiple JSON files can be fit in succession (paths with spaces in quotes):

.. code-block:: bash

    mosfit -e ./LSQ12dlf.json ./SN2015bn.json

The code outputs JSON files for each event/model combination that each contain a set of walkers that have been relaxed into an equilibrium about the posterior parameter distributions. This output is visualized via an example Jupyter notebook (``mosfit.ipynb``), which is copied to the ``products`` folder in the run directory, and by default shows output from the last ``MOSFiT`` run.

.. _parallel:

------------------
Parallel execution
------------------

``MOSFiT`` is parallelized and can be run in parallel by prepending ``mpirun -np #``, where ``#`` is the number of processors in your machine +1 for the master process. So, if you computer has 4 processors, the above command would be:

.. code-block:: bash

    mpirun -np 5 mosfit -e ./LSQ12dlf.json

``MOSFiT`` can also be run without specifying an event, which will yield a collection of light curves for the specified model described by the priors on the possible combinations of input parameters specified in the ``parameters.json`` file. This is useful for determining the range of possible outcomes for a given theoretical model:

.. code-block:: bash

    mpirun -np 5 mosfit -m magnetar

.. _own-data:

-------------------
Using your own data
-------------------

``MOSFiT`` has a built-in converter that can take input data in a number of formats and convert that data to the Open Catalog JSON format. Using the converter is straightforward, simply pass the path to the file(s) using the same ``-e`` option:

.. code-block:: bash

    mosfit -e my_ascii_data_file.csv

``MOSFiT`` will convert the files to JSON format and immediately begin processing the new files (append ``-G`` to immediately exit after conversion). For more information, please see the :ref:`Private data` section.

.. _producing-outputs:

-----------------
Producing outputs
-----------------

All outputs (except for converted observational data) are stored in the ``products`` directory, which is created by ``MOSFiT`` automatically in the current run directory. By default (without ``--quick-save``), the main fit results are stored in fixed filenames under ``products``—chiefly ``walkers.json``—which merges the information from the original event JSON with the walkers produced by sampling. Passing ``--quick-save`` instead writes similarly named output files prefixed by the transient name (see :ref:`io`).

Additional outputs can be produced via some optional options that can be passed to ``MOSFiT``. Please see the :ref:`arbitrary outputs <arbitrary>` section.

.. _visualizing:

-------------------
Visualizing outputs
-------------------

The outputs from ``MOSFiT`` can be visualized using the Jupyter notebook ``mosfit.ipynb`` copied by the code into a ``jupyter`` directory within the current run directory. This notebook is intended to be a simple demonstration of how to visualize the output data, and can be modified by the users to their own needs.

First, the user should make sure that Jupyter is installed, then execute the Jupyter notebook from the run directory:

.. code-block:: bash

    jupyter notebook jupyter/mosfit.ipynb

In this notebook, there are four cells which should require minimal editing to visualize your results; the cells should be evaluated in order. The first cell imports modules and loads the data output from the last run (store in ``walkers.json``). The second cell displays the ensemble of light curve fits and the data the model was fitted to:

.. image:: images/light-curve.png

The third cell shows X-ray observations, if the transient had any.

The fourth cell shows the evolution of free parameters as a function of time (the Monte Carlo chain).

The last cell produces a corner plot using the `corner package <https://corner.readthedocs.io>`_.

.. _sharing:

-------------------------------------------
Outputs and provenance
-------------------------------------------

Runs write products under ``products/`` (JSON with walkers and model metadata, optionally chains and extras). This fork does **not** upload fits or observations anywhere: distribution, archiving, and DOIs are your responsibility (e.g. Zenodo, your collaboration’s pipeline, version-controlled paths + hashes).

Treat ``products/*.json`` as the canonical deliverable for reproducibility alongside the exact input catalog JSON (or ASCII) and pinned ``MOSFiT`` commit or release.

.. _troubleshooting:

---------------
Troubleshooting
---------------

If you are having trouble getting ``MOSFiT`` working, please first consult our :ref:`FAQ` page, which addresses many common issues. If the answers there do not answer your questions, feel free to join our `#mosfit Slack channel on AstroChats <https://slack.astrocats.space>`_ and ask for assistance.
