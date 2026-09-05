.. _help:

====
Help
====

.. _faq:

--------------------------
Frequently Asked Questions
--------------------------

What do I do if MOSFiT or one of its requirements isn't installing?
===================================================================

For development, install from source with ``uv sync`` (see :ref:`source`). Published installs via ``conda`` or ``pip`` can skip some compilation steps that are common sources of error. If you are still having issues installing ``MOSFiT``, please ask us directly in the `#mosfit Slack channel on AstroChats <https://slack.astrocats.space>`_.

What can I try if MOSFiT won't run?
===================================

If ``MOSFiT`` is the first ``conda`` program you've used, and you previously used your system's built-in Python install, your shell environment may still be set up for your old Python setup, which can cause problems both for ``MOSFiT`` and your old Python programs. One common issue is that your ``PYTHON_PATH`` environment variable might be set to your build-in Python's install location, this will supercede conda's paths and potentially cause issues. Edit your ``.bashrc`` or ``.profile`` file to remove any ``PYTHON_PATH`` variable declarations, this will prevent path conflicts.

Is MOSFiT using the correct data?
=================================

``MOSFiT`` reads **only** the files you pass with ``-e`` (catalog-format JSON on disk or ASCII consumed by the converter). It does **not** download event payloads from catalog websites. Correctness is therefore entirely governed by **your** inputs—validate photometric systems, source tags, units, and time standards before trusting fits.

Keep a reproducible snapshot: immutable input JSON path or checksum, exact command line, and the ``products/walkers.h5`` (and optional ``chain.h5`` / extras) emitted by that run.

Can I fit private data with MOSFiT?
===================================

Yes! Pass the path to your ASCII datafile or catalog-format JSON using the ``-e`` flag. ``MOSFiT`` no longer fetches events over the network or uploads via ``-u``. More info can be found in :ref:`Event data / ASCII <private>` in the fitting docs.

How do I exclude particular instruments/bands/sources from my fit?
==================================================================

Excluding instruments can be accomplished by using the ``--exclude-instruments`` option, and excluding bands can be accomplished using the ``--exclude-bands`` option. All the data from a particular source (e.g. a paper or survey) can be excluded using ``--exclude-sources`` (see :ref:`here <restricting>` for more information on restricting your dataset). Finer exclusions (one band–instrument combination but not others) are often simplest by editing your input catalog JSON directly before fitting.

.. _contact:

-------
Contact
-------

If you need additional help, the most rapid way to receive it is to join `our Slack channel <https://astrochats.slack.com/messages/mosfit>`_. Barring that, feel free to `contact us via e-mail <mailto:guillochon@gmail.com>`_.
