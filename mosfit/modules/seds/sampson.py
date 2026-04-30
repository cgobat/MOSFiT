"""
sampson.py

MOSFiT SED module wrapping Karthik's stripped-envelope SN emulator ("sampson").

Weights ship under ``mosfit/emulators/sampson/`` (or ``$MOSFIT_EMULATOR_DATA/sampson``).

This class:
- builds a descriptor vector from MOSFiT parameters,
- normalizes descriptor and time using precomputed stats,
- calls a trained neural emulator (SimpleFluxMLP),
- returns model SEDs on MOSFiT's wavelength grid.
"""

from math import pi
from pathlib import Path
import os
import numpy as np
import torch

from astropy import constants as c
from astropy import units as u

from mosfit.constants import FOUR_PI, KM_CGS, M_SUN_CGS
from mosfit.emulators import emulator_weights_dir
from mosfit.modules.seds.sed import SED
from mosfit.emulators.preset_systematic import preset_systematic_mag

import torch.nn as nn
import torch.nn.functional as F


class SimpleFluxMLP(nn.Module):
    """
    Simple MLP mapping physical parameters -> flux spectrum.

    Args:
        n_physical_param: input dimension (number of physical parameters)
        n_wavelength: output dimension (number of wavelength points)
        d_model: hidden dimension
        num_layers: total number of linear layers (>= 2)
        nhead, learnedPE: kept for API compatibility, not used
    """
    def __init__(self,
                 n_physical_param=10,
                 n_wavelength=602,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 learnedPE=True):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2 for an input and output layer."

        layers = []

        # Input layer
        layers.append(nn.Linear(n_physical_param, d_model))
        layers.append(nn.GELU())

        # Hidden layers: num_layers - 2 internal linear layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.GELU())

        # Output layer
        layers.append(nn.Linear(d_model, n_wavelength))

        self.net = nn.Sequential(*layers)

    def forward(self, physical_param):
        """
        physical_param: [B, n_physical_param]
        Returns: [B, n_wavelength]
        """
        return self.net(physical_param)


# ---------------------------------------------------------------------------
# Weights live under ``mosfit/emulators/sampson/`` (or MOSFIT_EMULATOR_DATA).
# ---------------------------------------------------------------------------
_EMULATOR_NAME = "sampson"
_WEIGHTS_DIR = emulator_weights_dir(_EMULATOR_NAME)
NORMALIZATION_STATS_PATH = str(
    _WEIGHTS_DIR / "normalization_stats_tmin4_tmax60_N15.pt"
)
EMULATOR_WEIGHTS_PATH = str(_WEIGHTS_DIR / "emulator.pth")


def _sampson_debug_enabled():
    v = os.environ.get("MOSFIT_SAMPSON_DEBUG", "").strip().lower()
    return v not in ("", "0", "false", "no", "off")


def _debug_band_looks_like_b(bl: str) -> bool:
    """Heuristic match for Bessell-/Johnson-/single-letter B passbands."""
    bl = bl.lower().strip()
    if not bl:
        return False
    if ("bessell" in bl) and (
            ".b" in bl.replace(" ", "") or "/b." in bl or "bessell.b" in bl):
        return True
    if "johnson_b" in bl or bl.endswith(("johnson_b", "cousins_b")):
        return True
    if bl.endswith(("_b", "/b")):
        return True
    tail = bl.replace(" ", "").split("/")[-1].split(".")[-1].rstrip("*").split("_")[
        -1]
    return tail == "b"


def _summarize_nn_debug(pred_valid_torch, flux_np_after_floor):
    """Summaries for MOSFIT_SAMPSON_DEBUG (pre-10** log target, post-10** linear flux)."""
    pv = pred_valid_torch.detach().cpu().numpy().astype(np.float64).ravel()
    ok = np.isfinite(pv)
    lines = []
    if np.any(ok):
        lines.append(
            "pred before 10** (log10-like target flux) min/max: {:.6g} {:.6g}".format(
                float(np.min(pv[ok])), float(np.max(pv[ok]))
            )
        )
        lines.append("pred finite frac: {:.4f}".format(float(np.mean(ok))))
    else:
        lines.append("pred before 10**: no finite values")
    fl = flux_np_after_floor.ravel().astype(np.float64, copy=False)
    okf = np.isfinite(fl) & (fl > 0)
    lines.append(
        "linear flux after exp(clip(log10 target))+floor nan_frac: {:.4f}".format(
            float(np.mean(~np.isfinite(fl)))
        )
    )
    if np.any(okf):
        lines.append(
            "linear flux log10 min/max (finite, after floor): {:.6g} {:.6g}".format(
                float(np.min(np.log10(fl[okf]))),
                float(np.max(np.log10(fl[okf]))),
            )
        )
    return lines


# ---------------------------------------------------------------------------
# SED implementation
# ---------------------------------------------------------------------------


class Sampson(SED):
    """Stripped-envelope spectral energy distribution from NN emulator."""
    
    # Physical constants (cgs)
    C_CONST = c.c.cgs.value
    FLUX_CONST = (
        FOUR_PI * (2.0 * c.h * c.c ** 2 * pi).cgs.value * u.Angstrom.cgs.scale
    )
    X_CONST = (c.h * c.c / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    EMULATOR_T_MIN = 4.0
    EMULATOR_T_MAX = 55.0   # ← was 60.0, buffer against edge effects

    # ``10**log`` / interp can yield 0, tiny negative floats, or underflow — keep linear
    # flux strictly positive for photometry (matches standalone emulator floor spirit).
    MIN_LINEAR_FLUX = 1.0e-40

    # Fixed σ_sys (mag) added in quadrature on the likelihood diagonal — see ``preset_systematic_mag``.
    # Cosine U-shape: σ_sys MIN_MAG at PEAK_T_REST_DAY (middle), MAX_MAG at emulator time ends.
    PRESET_SYSTEMATIC_ENABLED = True
    PRESET_SYS_MIN_MAG = 0.2
    PRESET_SYS_MAX_MAG = 0.7
    PRESET_SYS_PEAK_T_REST_DAY = 20.0

    # Wavelength grid used by the emulator.
    # Your preprocessing used 602 points from 2200–9700 Å, then N=15 downsampling.
    # 602 -> indices 0,15,...,600 → 41 points.
    _orig_wav_grid = np.linspace(2000.0 + 200.0, 10000.0 - 300.0, 602)
    fixed_wav_grid = _orig_wav_grid[::15]  # length 41

    # Shared state
    mean_std_dict = None
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _loaded = False

    def __init__(self, *args, **kwargs):
        """Initialize module and lazily load emulator + stats once."""
        super().__init__(*args, **kwargs)

        if not Sampson._loaded:
            # ------------------------------------------------------------------
            # Load normalization statistics (mean/std for descriptor, time, flux)
            # ------------------------------------------------------------------
            msd = torch.load(
                NORMALIZATION_STATS_PATH,
                map_location=Sampson.device,
                weights_only=False,  # needed in PyTorch 2.6+ for non-state_dict
            )
            if not isinstance(msd, dict):
                raise RuntimeError(
                    f"Expected normalization stats at {NORMALIZATION_STATS_PATH} "
                    "to be a dict of tensors (means/stds). Got type: "
                    f"{type(msd)}"
                )

            # Move tensor values to device as float32
            for k, v in list(msd.items()):
                if torch.is_tensor(v):
                    msd[k] = v.to(device=Sampson.device, dtype=torch.float32)

            # Basic sanity checks
            assert "descriptor_mean" in msd and "descriptor_std" in msd, \
                "Stats dict must contain 'descriptor_mean' and 'descriptor_std'."
            assert "time_mean" in msd and "time_std" in msd, \
                "Stats dict must contain 'time_mean' and 'time_std'."
            assert "fluxes_mean" in msd and "fluxes_std" in msd, \
                "Stats dict must contain 'fluxes_mean' and 'fluxes_std'."

            # Descriptor stats
            desc_mean = msd["descriptor_mean"]
            desc_std = msd["descriptor_std"]
            assert desc_mean.shape == desc_std.shape == (9,), \
                f"Expected descriptor_mean/std shape (9,), got {desc_mean.shape}, {desc_std.shape}"

            # Flux stats are GLOBAL SCALARS in your script; no shape check here.
            # Broadcasting will apply mean/std to each wavelength bin.

            # Sanity: make sure our fixed_wav_grid and model n_wav agree
            n_wav = Sampson.fixed_wav_grid.shape[0]
            assert n_wav == 41, f"Expected 41 wavelength bins, got {n_wav}"

            Sampson.mean_std_dict = msd

            # ------------------------------------------------------------------
            # Build emulator and load weights (SimpleFluxMLP, not Transformer)
            # ------------------------------------------------------------------
            Sampson.model = SimpleFluxMLP(
                d_model=512,
                num_layers=10,
                n_wavelength=n_wav,
            ).to(Sampson.device)

            state_dict = torch.load(
                EMULATOR_WEIGHTS_PATH,
                map_location="cpu",  # state_dict is device-agnostic
                weights_only=True,   # this file is a pure state_dict
            )
            Sampson.model.load_state_dict(state_dict)
            Sampson.model.to(Sampson.device)
            Sampson.model.eval()
            torch.set_grad_enabled(False)

            Sampson._loaded = True

    def process(self, **kwargs):
        """Run the emulator and return SEDs on MOSFiT's wavelength grid."""

        # ----------------------------------------------------------------------
        # Extract common inputs from MOSFiT
        # ----------------------------------------------------------------------
        # ``dense_times`` follow ``rest_times`` = ``all_times / (1+z)`` (see
        # ``RestTimes`` / ``DenseTimes``). Emulator age must match training:
        # ``(t_obs - t_exp) / (1+z)`` = ``dense_times - t_exp / (1+z)``.
        # Using ``(dense_times - t_exp) / (1+z)`` mixes frames and shrinks age
        # by ~``z * t``, which can push epochs below ``EMULATOR_T_MIN`` (NaNs).
        obs_times = np.asarray(kwargs[self.key("dense_times")], dtype=float)
        self._dense_indices = kwargs[self.key("dense_indices")]  # [N_points]

        texp = kwargs[self.key("texplosion")]
        z = kwargs[self.key("redshift")]
        zp1 = 1.0 + float(z)
        rtk = self.key("resttexplosion")
        if rtk in kwargs:
            t_rest_origin = kwargs[rtk]
        else:
            t_rest_origin = float(texp) / zp1

        self._times = obs_times - t_rest_origin

        lum_key = self.key("luminosities")
        kwargs = self.prepare_input(lum_key, **kwargs)
        self._luminosities = kwargs[lum_key]                     # [N_points]

        valid_mask = np.zeros(len(self._luminosities), dtype=bool)

        self._bands = kwargs["all_bands"]
        self._band_indices = kwargs["all_band_indices"]          # [N_points]
        self._frequencies = kwargs["all_frequencies"]            # [N_points]

        # Descriptor parameters from MOSFiT (must match training convention)
        self._eta_vel = kwargs[self.key("eta_vel")]
        #this will be passed in, in units of km/s. I need it in cm/s
        self._min_vel = kwargs[self.key("min_vel")] * 100 * KM_CGS
        self._del_vel = kwargs[self.key("del_vel")] * 100 * KM_CGS

        self._eta_ni = kwargs[self.key("eta_ni")]
        self._eta_he = kwargs[self.key("eta_he")]
        self._eta_op = kwargs[self.key("eta_op")]

        self._mni = kwargs[self.key("m_ni")] * M_SUN_CGS
        self._mhe = kwargs[self.key("m_he")] * M_SUN_CGS
        self._mop = kwargs[self.key("m_op")] * M_SUN_CGS

        # Unnormalized descriptor vector (9 parameters; same order as training)
        unnorm_descriptor = torch.tensor(
            [
                self._eta_vel,
                self._eta_he,
                self._eta_ni,
                self._eta_op,
                self._min_vel,
                self._del_vel,
                self._mhe,
                self._mni,
                self._mop,
            ],
            dtype=torch.float32,
            device=Sampson.device,
        )

        cc = self.C_CONST
        msd = Sampson.mean_std_dict

        # ----------------------------------------------------------------------
        # Normalize descriptor and prepare batched input
        # ----------------------------------------------------------------------
        desc_mean = msd["descriptor_mean"]  # shape [9]
        desc_std = msd["descriptor_std"]    # shape [9]

        norm_descriptor = (unnorm_descriptor - desc_mean) / desc_std
        norm_descriptor_t = norm_descriptor.view(1, 9)  # (1, 9)

        # Unique dense time indices (so we call NN once per unique time)
        unique_t_indices = np.unique(self._dense_indices)
        unique_times = self._times[unique_t_indices]  # days, ndarray

        # Normalize times (training used time in seconds)
        time_secs = unique_times * 86400.0
        time_secs_t = torch.tensor(
            time_secs,
            dtype=torch.float32,
            device=Sampson.device,
        )

        time_mean = msd["time_mean"]
        time_std = msd["time_std"]
        # Handle possible scalar vs 0-dim tensor
        if not torch.is_tensor(time_mean):
            time_mean = torch.tensor(time_mean, dtype=torch.float32, device=Sampson.device)
        if not torch.is_tensor(time_std):
            time_std = torch.tensor(time_std, dtype=torch.float32, device=Sampson.device)
        time_mean = time_mean.to(Sampson.device, dtype=torch.float32)
        time_std = time_std.to(Sampson.device, dtype=torch.float32)

        norm_times_t = (time_secs_t - time_mean) / time_std
        norm_times_t = norm_times_t.view(-1, 1)  # (N_unique, 1)

        # Repeat descriptor for each time and concatenate
        descriptor_batch = norm_descriptor_t.expand(norm_times_t.shape[0], -1)  # (N_unique, 9)
        input_batch = torch.cat((descriptor_batch, norm_times_t), dim=1)        # (N_unique, 10)

        # ----------------------------------------------------------------------
        # Single batched network evaluation
        # ----------------------------------------------------------------------
        
        valid_query_mask = (
            (unique_times >= Sampson.EMULATOR_T_MIN) & 
            (unique_times <= Sampson.EMULATOR_T_MAX)
        )

        dbg_nn_pred_valid = None
        dbg_flux_np_after_floor = None
        if np.any(valid_query_mask):
            valid_input = input_batch[valid_query_mask]
            with torch.no_grad():
                pred_valid = (
                    Sampson.model(valid_input) * msd["fluxes_std"] + msd["fluxes_mean"]
                )
            dbg_nn_pred_valid = pred_valid
            # Network outputs log10-like flux bins. Avoid float32 overflow:
            # ``10.** x`` overflows in fp32 near x ≈ log10(3e38) ≈ 38.53; standalone uses
            # ``exp(clip(log(10)*x))`` in float64 (see sampson_emulator.py).
            pred_lp = pred_valid.detach().cpu().numpy().astype(np.float64)
            lin = np.exp(np.clip(pred_lp * np.log(10.0), -700.0, 700.0))
            pred_fluxes_valid_np = np.maximum(lin, Sampson.MIN_LINEAR_FLUX)
            dbg_flux_np_after_floor = pred_fluxes_valid_np

        else:
            pred_fluxes_valid_np = None

        # Map back
        flux_by_dense_index = {}
        valid_idx = 0
        for k, ti in enumerate(unique_t_indices):
            if valid_query_mask[k]:
                flux_by_dense_index[ti] = pred_fluxes_valid_np[valid_idx]
                valid_idx += 1
            else:
                flux_by_dense_index[ti] = np.full(
                    len(Sampson.fixed_wav_grid), np.nan
                )

        # ----------------------------------------------------------------------
        # Build SEDs at each requested (time, band)
        # ----------------------------------------------------------------------
        czp1 = cc / zp1

        seds = []
        rest_wavs_dict = {}

        dbg_ex_li = None
        dbg_ex_stats = None

        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            ti = self._dense_indices[li]

            dense_time = self._times[ti]  # rest-frame days since explosion
            


            # mark whether this point is inside emulator's training window
            if Sampson.EMULATOR_T_MIN <= dense_time <= Sampson.EMULATOR_T_MAX:
                valid_mask[li] = True

            # Rest-frame wavelengths for this band
            if bi >= 0:
                rest_wavs = rest_wavs_dict.setdefault(
                    bi,
                    self._sample_wavelengths[bi] / zp1,
                )
            else:
                rest_wavs = np.array([czp1 / self._frequencies[li]])

            # Emulator flux vector (on fixed_wav_grid) at this time index
            pred_fluxes_np = flux_by_dense_index[ti]

            # Interpolate emulator fluxes onto the band rest-frame wavelength grid
            rest_fluxes = np.interp(
                rest_wavs,
                Sampson.fixed_wav_grid,
                pred_fluxes_np,
            )
            rest_fluxes = np.maximum(rest_fluxes, Sampson.MIN_LINEAR_FLUX)

            # NOTE: `lum` is not currently used to rescale flux.

            if _sampson_debug_enabled() and dbg_ex_li is None and li < len(
                    self._bands):
                ost = kwargs.get("observation_types")
                bstr = str(self._bands[li])
                bl = bstr.lower()
                # Sampson DAG inputs omit ``observation_types`` → treat rows as observable
                if ost is None or li >= len(np.asarray(ost, dtype=object)):
                    ok_mag = True
                else:
                    ostr = str(np.asarray(ost, dtype=object).ravel()[li])
                    ol = ostr.strip().lower()
                    ok_mag = ol in ("magnitude", "magcount") or (
                        ol.startswith("mag") and "upper" not in ol and ol not in ("magnitudelimit",)
                    )
                ok_b_band = _debug_band_looks_like_b(bl)
                if ok_mag and ok_b_band:
                    dbg_ex_li = li
                    pgrid = np.asarray(pred_fluxes_np, dtype=np.float64).ravel()
                    rrest = np.asarray(rest_fluxes, dtype=np.float64).ravel()
                    dbg_ex_stats = {
                        "dense_t_rest": float(dense_time),
                        "dense_idx": int(ti),
                        "pred_grid_nan_frac": float(np.mean(~np.isfinite(pgrid))),
                        "interp_nan_frac": float(np.mean(~np.isfinite(rrest))),
                    }

            seds.append(rest_fluxes)

        # ----------------------------------------------------------------------
        # Clean up fluxes and add to existing MOSFiT SEDs
        # ----------------------------------------------------------------------


        seds = np.asarray(seds, dtype=float)
        seds[~np.isfinite(seds)] = np.nan
        _finite = np.isfinite(seds)
        if np.any(_finite):
            seds[_finite] = np.maximum(seds[_finite], Sampson.MIN_LINEAR_FLUX)

        if _sampson_debug_enabled():
            lines = [
                "[MOSFIT_SAMPSON_DEBUG] Sampson.process",
                "z={:.8f} (1+z)={:.8f}".format(float(z), float(zp1)),
                "rest_t_explosion ({}) = {}".format(rtk, repr(t_rest_origin)),
                "dense t_rest=self._times obs_times - rest_t_explosion "
                "→ min/max: {:.6g} {:.6g} (N={})".format(
                    float(np.min(self._times)) if len(self._times) else float("nan"),
                    float(np.max(self._times)) if len(self._times) else float("nan"),
                    len(self._times),
                ),
                "unique_times min/max: {:.6g} {:.6g} (n_unique={})".format(
                    float(np.min(unique_times)) if len(unique_times) else float("nan"),
                    float(np.max(unique_times)) if len(unique_times) else float("nan"),
                    len(unique_times),
                ),
                "valid_query_mask (time in [{}, {}]) true: {}/{}".format(
                    Sampson.EMULATOR_T_MIN,
                    Sampson.EMULATOR_T_MAX,
                    int(np.sum(valid_query_mask)),
                    len(valid_query_mask),
                ),
            ]
            if dbg_nn_pred_valid is not None and dbg_flux_np_after_floor is not None:
                lines.extend(
                    _summarize_nn_debug(dbg_nn_pred_valid, dbg_flux_np_after_floor)
                )
            else:
                lines.append(
                    "NN skipped (no valid_query_mask true in batch)."
                )
            if dbg_ex_stats is not None:
                lines.append(
                    "exemplar B-band row li={} band={!r} "
                    "dense_t_rest={:.6g} d dense_idx={} "
                    "nan_frac on 41-pt grid {:.4f} nan_frac after interp {:.4f}".format(
                        dbg_ex_li,
                        str(self._bands[int(dbg_ex_li)]),
                        dbg_ex_stats["dense_t_rest"],
                        dbg_ex_stats["dense_idx"],
                        dbg_ex_stats["pred_grid_nan_frac"],
                        dbg_ex_stats["interp_nan_frac"],
                    )
                )
            else:
                lines.append("exemplar: no B-band row matched for debug.")
            print("\n".join(lines), flush=True)

        seds = self.add_to_existing_seds(seds, **kwargs)

        tor = {
            "sample_wavelengths": self._sample_wavelengths,
            self.key("seds"): seds,
            # per-point validity mask: True where 4 <= t_rest <= 60
            "sesn_valid_mask": valid_mask,
            # original observer-frame times for downstream modules
            "times_out": obs_times,
        }

        if Sampson.PRESET_SYSTEMATIC_ENABLED:
            t_per = self._times[np.asarray(self._dense_indices, dtype=np.intp)]
            tor["emulator_preset_systematic_mag"] = preset_systematic_mag(
                t_per,
                Sampson.EMULATOR_T_MIN,
                Sampson.EMULATOR_T_MAX,
                Sampson.PRESET_SYS_MIN_MAG,
                Sampson.PRESET_SYS_MAX_MAG,
                Sampson.PRESET_SYS_PEAK_T_REST_DAY,
            )

        return tor