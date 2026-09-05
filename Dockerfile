# MOSFiT CPU runtime for the OSG / SkyPortal analysis service.
#
# A generic MOSFiT runtime: the osg-skyportal-plugin ships mosfit_wrapper.py +
# mosfit_bridge.py per-job, so no plugin code is baked in. Built from this repo
# (this fork's fetcher is local-only, so fits run fully offline on a worker).
# ZTF transmission curves are baked in so ZTF-band fits never touch the network.
FROM python:3.14.7-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# openmpi so the optional mpi4py extra builds and imports (the fit itself runs
# single-process via schwimmbad's SerialPool); curl for the SVO filter fetch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates git \
        libopenmpi-dev openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mosfit
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY mosfit ./mosfit

# CPU torch first so the resolver does not pull a CUDA wheel, then the project.
RUN uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cpu torch \
    && uv pip install --system --no-cache --extra-index-url https://download.pytorch.org/whl/cpu ".[mpi]"

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
