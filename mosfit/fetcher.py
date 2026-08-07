"""Resolve local catalog-format JSON paths for MOSFiT events."""
import codecs
import json
import os
from collections import OrderedDict

from mosfit.printer import Printer
from mosfit.utils import listify


class Fetcher(object):
    """Locate event JSON on disk only (no Open Catalog downloads)."""

    _NO_NETWORK_MSG = (
        'MOSFiT no longer downloads events from the Open Astronomy Catalogs. '
        'Pass a path to a catalog-format JSON file (already on disk), or an '
        'ASCII table path for conversion. Example: `-e ./my_supernova.json`.')

    def __init__(self, test=False, printer=None, **kwargs):
        """Initialize class."""
        self._test = test
        self._printer = Printer() if printer is None else printer

    def fetch(self, event_list):
        """Resolve events to paths of existing catalog-format JSON files."""
        prt = self._printer

        levent_list = listify(event_list)
        events = [None for _ in levent_list]

        for ei, event in enumerate(levent_list):
            if not event:
                continue

            cand = os.path.abspath(os.path.expanduser(event.strip()))
            if not os.path.isfile(cand):
                prt.prt(self._NO_NETWORK_MSG + '\n' +
                        'Missing or non-file argument: `{}`'.format(event),
                        wrapped=True, warning=True)
                raise RuntimeError('Event file not found: {}'.format(event))

            basename = os.path.basename(cand)
            if basename.lower().endswith('.json'):
                name = basename[:-5]
            else:
                name = os.path.splitext(basename)[0]

            od = OrderedDict()
            od['name'] = name
            od['path'] = cand
            events[ei] = od

            prt.message('event_file', [cand], wrapped=True)

        return events

    def load_data(self, event):
        """Return data from specified path."""
        if event is None or 'path' not in event:
            return None
        if not os.path.exists(event['path']):
            return None
        with codecs.open(event['path'], 'r', encoding='utf-8') as f:
            return json.load(f, object_pairs_hook=OrderedDict)
