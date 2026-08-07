.. _package:

==================
Using as a package
==================

If you wish to produce light curves or other data products for a given model without using the fitting and evidence accumulation features of ``MOSFiT``, functions within the code can be accessed by importing the ``mosfit`` package into your Python code.

.. _run:

---------------------
Produce model outputs
---------------------

In the snippet below, we resolve a **local** catalog-format JSON path with ``Fetcher``, build a ``Model``, load that event, and evaluate the model stack once. ``Fetcher`` does not query the Open Astronomy Catalogs; the file must exist on disk.

.. code-block:: python

    import mosfit
    import numpy as np

    event_path = 'path/to/my_supernova.json'  # AstroCats-compatible JSON file

    my_fetcher = mosfit.fetcher.Fetcher()
    fetched = my_fetcher.fetch([event_path])[0]

    my_model = mosfit.model.Model(model='slsn')
    my_model.load_data(my_fetcher.load_data(fetched), event_name=fetched['name'])

    # Generate a random input vector of free parameters.
    x = np.random.rand(my_model.get_num_free_parameters())

    # Produce model output.
    outputs = my_model.run(x)
    print('Keys in output: `{}`'.format(', '.join(list(outputs.keys()))))
