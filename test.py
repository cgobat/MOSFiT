"""Smoke test: `Fitter` + generative `fit_events` (no local JSON paths)."""

import mosfit

# Instantiate without Open Catalog downloads (historic `offline=` removed).
my_fitter = mosfit.fitter.Fitter(quiet=False, test=True)

print('Running generative `fit_events` smoke test.')
entries, ps, lnprobs = my_fitter.fit_events(
    events=[], models=['magni', 'slsn'], iterations=1)

print(
    'Model WAIC / score values: ',
    [[y['models'][0]['score']['value'] for y in x] for x in entries])
