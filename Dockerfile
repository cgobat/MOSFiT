# MOSFiT CPU runtime for the OSG / SkyPortal analysis service.
#
# A generic MOSFiT runtime: the osg-skyportal-plugin ships mosfit_wrapper.py +
# mosfit_bridge.py per-job, so no plugin code is baked in. Built from this repo
# (this fork's fetcher is local-only, so fits run fully offline on a worker).
# ZTF transmission curves are baked in so ZTF-band fits never touch the network.
FROM python:3.11-slim

# openmpi so MOSFiT's mpi4py dependency builds and imports (the fit itself runs
# single-process via schwimmbad's SerialPool); curl for the SVO filter fetch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates git \
        libopenmpi-dev openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# numpy<2 (astropy compat) + pandas<3 per MOSFiT's pins, installed before the
# source install so the resolver can't pull numpy 2.
RUN pip install --no-cache-dir "numpy>=1.23,<=1.26.4" "pandas>=2.1,<3"

# Install MOSFiT from this repo (the offline-fetcher fork). Pure Python — no
# Cython extensions are compiled.
COPY . /opt/mosfit
RUN pip install --no-cache-dir /opt/mosfit

# Bake ZTF (Palomar/ZTF.{g,r,i}) transmission curves into the installed package
# so ZTF-band fits resolve filters locally instead of downloading from SVO at
# runtime (a worker has no network).
RUN FILTERS=$(python -c "import mosfit, os; print(os.path.join(os.path.dirname(mosfit.__file__), 'modules', 'observables', 'filters'))") \
    && for b in g r i; do \
        curl -fsSL "http://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php?PhotCalID=Palomar/ZTF.${b}/AB" \
            -o "$FILTERS/Palomar_ZTF.${b}_AB.xml"; \
    done \
    && for b in g r i; do \
        test -s "$FILTERS/Palomar_ZTF.${b}_AB.xml" || { echo "missing baked ZTF ${b} filter"; exit 1; }; \
    done

# Fail the build if the runtime isn't importable.
RUN python -c "import mosfit, emcee, dynesty; from mosfit.fitter import Fitter; print('mosfit OK')"

ENTRYPOINT ["python"]
