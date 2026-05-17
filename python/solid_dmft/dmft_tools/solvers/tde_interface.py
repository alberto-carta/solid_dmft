# %%
################################################################################
#
# solid_dmft - A versatile python wrapper to perform DFT+DMFT calculations
#              utilizing the TRIQS software library
#
# Copyright (C) 2018-2020, ETH Zurich
# Copyright (C) 2021, The Simons Foundation
#      authors: A. Carta, A. Hampel, M. Merkel, and S. Beck
#
# solid_dmft is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# solid_dmft is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# solid_dmft (in the file COPYING.txt in this directory). If not, see
# <http://www.gnu.org/licenses/>.
#
################################################################################
# pyright: reportUnusedExpression=false
'''
TDE (Thermally Dephased Ensemble) solver class for solid_dmft.

The TDE solver discovers multiple HF saddle points of the impurity problem,
Boltzmann-weights them, and assembles a frequency-dependent self-energy.

The first ``n_warmup_hf`` DMFT iterations use a plain Hartree-Fock solver to
stabilise a paramagnetic bath. Afterwards, the full TDE ensemble solver is used.
'''
import os
import importlib.util

import numpy as np
from triqs.gf import MeshReFreq, Gf
from triqs.gf.descriptors import Fourier
import triqs.utility.mpi as mpi

from triqs_dft_tools.util import compute_DC_from_density
from solid_dmft.dmft_tools.solvers.abstractdmftsolver import AbstractDMFTSolver

from triqs_hartree_fock import ImpuritySolver as hartree_solver
from triqs_hartree_fock.version import triqs_hartree_fock_hash, version as hf_version

from triqs_tde import (
    ThermallyDephasedEnsembleSolver,
    DensityMatrixProposals,
    SobolSigmaProposals,
)


def _validate_custom_proposals(proposals, gf_struct):
    """Validate that custom proposals are strictly list[dict[str, np.ndarray]].

    Raises AssertionError with a descriptive message on any violation.
    """
    block_sizes = {bl: sz for bl, sz in gf_struct}

    assert isinstance(proposals, list), (
        f"custom_proposals must return a list, got {type(proposals).__name__}"
    )

    for i, entry in enumerate(proposals):
        assert isinstance(entry, dict), (
            f"custom_proposals[{i}] must be a dict, got {type(entry).__name__}"
        )
        for key, val in entry.items():
            assert isinstance(key, str), (
                f"custom_proposals[{i}]: key must be str, got {type(key).__name__}"
            )
            assert key in block_sizes, (
                f"custom_proposals[{i}]: unknown block name '{key}', "
                f"expected one of {list(block_sizes.keys())}"
            )
            assert isinstance(val, np.ndarray), (
                f"custom_proposals[{i}]['{key}']: value must be np.ndarray, "
                f"got {type(val).__name__}"
            )
            expected_shape = (block_sizes[key], block_sizes[key])
            assert val.shape == expected_shape, (
                f"custom_proposals[{i}]['{key}']: shape {val.shape} does not match "
                f"expected {expected_shape}"
            )


def _load_custom_proposals(filepath, gf_struct, icrsh):
    """Load custom proposals (and optional orbital symmetry groups) from a Python file.

    The file must define ``get_custom_proposals(gf_struct, icrsh)`` returning
    ``list[dict[str, np.ndarray]]`` — a list of target density matrices for
    the given impurity index ``icrsh``.

    Optionally the file may also define
    ``get_orbital_symmetry_groups(gf_struct, icrsh)`` returning
    ``list[list[int]]`` — the degenerate orbital groups used for permutation
    augmentation (e.g. ``[[0, 3], [1, 2, 4]]`` for FeO eg/t2g).  When absent,
    no permutation augmentation is performed.
    """
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"custom_proposals_file not found: {filepath}")

    spec = importlib.util.spec_from_file_location("_custom_proposals", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, 'get_custom_proposals'):
        raise AttributeError(
            f"custom_proposals_file '{filepath}' must define "
            f"'get_custom_proposals(gf_struct, icrsh)'"
        )

    proposals = mod.get_custom_proposals(gf_struct, icrsh)
    _validate_custom_proposals(proposals, gf_struct)

    orbital_groups = None
    if hasattr(mod, 'get_orbital_symmetry_groups'):
        orbital_groups = mod.get_orbital_symmetry_groups(gf_struct, icrsh)

    return proposals, orbital_groups


def _save_landscape(tde_solver, output_dir, it, icrsh):
    """Save landscape data and optionally a plot for a TDE iteration.

    Parameters
    ----------
    tde_solver : ThermallyDephasedEnsembleSolver
        The solver after solve() has been called.
    output_dir : str
        Base output directory (e.g. jobname).
    it : int
        Current DMFT iteration number.
    icrsh : int
        Inequivalent correlated shell index.
    """
    if not mpi.is_master_node():
        return
    if tde_solver.solutions is None:
        return

    landscape_dir = os.path.join(output_dir, 'landscapes_per_iteration', f'it_{it}')
    os.makedirs(landscape_dir, exist_ok=True)

    stem = f'landscape_imp{icrsh}'

    # Save data file
    df = tde_solver.solutions.to_dataframe()
    df['action'] = df['action'] + df.get('regret', 0.0)
    dat_path = os.path.join(landscape_dir, f'{stem}.dat')
    df.to_csv(dat_path, sep='\t', index=True)
    mpi.report(f'  TDE landscape data saved to {dat_path}')

    # Try to save plots (scatter + histogram as separate files)
    try:
        import matplotlib
        matplotlib.use('Agg')
        landscape_path  = os.path.join(landscape_dir, f'{stem}_landscape.png')
        histogram_path  = os.path.join(landscape_dir, f'{stem}_histogram.png')
        tde_solver.plot_landscape(save_path=landscape_path, show=False)
        tde_solver.plot_action_histogram(save_path=histogram_path, show=False)
        mpi.report(f'  TDE landscape plot saved to {landscape_path}')
        mpi.report(f'  TDE histogram plot saved to {histogram_path}')
    except ImportError:
        mpi.report('  matplotlib not available — skipping landscape plots')


class TDEInterface(AbstractDMFTSolver):
    def __init__(
        self, general_params, solver_params, sum_k, icrsh, h_int,
        iteration_offset, deg_orbs_ftps, gw_params=None, advanced_params=None
    ):
        super().__init__(
            general_params, solver_params, sum_k, icrsh, h_int,
            iteration_offset, deg_orbs_ftps, gw_params, advanced_params
        )

        # ── HF solver params (used during warmup AND internally by TDE) ──
        self.triqs_solver_params = {}
        for key in ('method', 'tol', 'with_fock'):
            self.triqs_solver_params[key] = self.solver_params[key]
        # HF always runs self-consistent (one_shot=False) in this context
        self.triqs_solver_params['one_shot'] = False

        # ── GF objects on ImFreq and ReFreq ──
        self._init_ImFreq_objects()
        self._init_ReFreq_tde()

        gf_struct = self.sum_k.gf_struct_solver_list[self.icrsh]

        # ── Build spin_map: block_name -> 'up' or 'down' ──
        self._spin_map = {}
        for bl, _ in gf_struct:
            if bl in self.sum_k.spin_block_names[self.sum_k.SO]:
                # Direct match (e.g. 'up', 'down')
                self._spin_map[bl] = bl
            else:
                # Solver-rotated name (e.g. 'up_0', 'down_0')
                for spin_name in self.sum_k.spin_block_names[self.sum_k.SO]:
                    if bl.startswith(spin_name):
                        self._spin_map[bl] = spin_name
                        break
                else:
                    raise ValueError(
                        f"Cannot determine spin channel for block '{bl}'. "
                        f"Known spin names: {self.sum_k.spin_block_names[self.sum_k.SO]}"
                    )

        # ── HF solver for warmup phase ──
        # DC is handled by solid_dmft through G0 (TDE is not detected as an HF
        # solver, so solid_dmft applies DC externally). We therefore always
        # zero the internal DC of every HF solver we create, both here and
        # inside the TDE scan loop.
        self.triqs_solver = hartree_solver(
            beta=self.general_params['beta'],
            gf_struct=gf_struct,
            n_iw=self.general_params['n_iw'],
            force_real=self.solver_params['force_real'],
            symmetries=[self._make_spin_equal],
        )
        self.triqs_solver.dc_fixed_value = 0.0

        # ── TDE solver ──
        self.tde_solver = ThermallyDephasedEnsembleSolver(
            gf_struct=gf_struct,
            beta=self.general_params['beta'],
            n_iw=self.general_params['n_iw'],
            dc_fixed_value=0.0,  # DC is handled by the HF solver / solid_dmft
            force_real=self.solver_params['force_real'],
            enforce_paramagnetic=self.solver_params['enforce_paramagnetic'],
            orbital_symmetry_threshold=self.solver_params['orbital_symmetry_threshold'],
            prune_tol=self.solver_params['prune_tol'],
            denoising_bin_size=self.solver_params['denoising_bin_size'],
            verbosity=self.solver_params['verbosity'],
        )

        # ── TDE-specific config ──
        self.n_warmup_hf = self.solver_params['n_warmup_hf']
        self._warmup_done = (iteration_offset >= self.n_warmup_hf)
        self._in_tde_mode = False  # set True when current solve is TDE
        self._prior_solutions = None

        # ── Build proposal generators ──
        custom_proposals = None
        if self.solver_params['custom_proposals_file'] is not None:
            custom_proposals, orbital_groups = _load_custom_proposals(
                self.solver_params['custom_proposals_file'], gf_struct, self.icrsh
            )
            mpi.report(f'  TDE: loaded {len(custom_proposals)} custom proposals '
                       f'for impurity {self.icrsh} from '
                       f'{self.solver_params["custom_proposals_file"]}')
            if orbital_groups is not None:
                if self.solver_params['enforce_orbital_permutations']:
                    self.tde_solver.orbital_symmetry_groups = orbital_groups
                    mpi.report(f'  TDE: orbital symmetry groups for impurity '
                               f'{self.icrsh}: {orbital_groups}')
                else:
                    mpi.report(
                        f'  TDE: get_orbital_symmetry_groups found in '
                        f'custom_proposals_file but enforce_orbital_permutations=False '
                        f'— Goldstone analysis skipped.'
                    )

        self._proposal_generators = self._build_proposal_generators(custom_proposals)

        # ── Metadata ──
        self.git_hash = triqs_hartree_fock_hash
        self.version = hf_version

    def _interface_dc(self, hartree_instance):
        """Apply DC configuration to the HF solver, same as hartree_interface."""
        setattr(hartree_instance, 'dc', self.general_params['dc'])
        if self.general_params['dc_type'][self.icrsh] is not None:
            setattr(hartree_instance, 'dc_type', self.general_params['dc_type'][self.icrsh])

        for key in ['dc_factor', 'dc_fixed_value']:
            if key in self.advanced_params and self.advanced_params[key] is not None:
                setattr(hartree_instance, key, self.advanced_params[key])

        for key in ['dc_U', 'dc_J', 'dc_fixed_occ']:
            if key in self.advanced_params and self.advanced_params[key][self.icrsh] is not None:
                setattr(hartree_instance, key, self.advanced_params[key][self.icrsh])

        if 'dc_dmft' in self.general_params:
            if self.general_params['dc_dmft'] == False:
                mpi.report(
                    'TDE SOLVER: Warning dft occupation in the DC calculations '
                    'are meaningless for the hartree solver, reverting to dmft occupations'
                )

        dc_type = hartree_instance.dc_type
        if dc_type == 0 and not self.general_params['magnetic']:
            mpi.report(f"TDE SOLVER: Detected dc_type = {dc_type}, changing to 'cFLL'")
            hartree_instance.dc_type = 'cFLL'
        elif dc_type == 0 and self.general_params['magnetic']:
            mpi.report(f"TDE SOLVER: Detected dc_type = {dc_type}, changing to 'sFLL'")
            hartree_instance.dc_type = 'sFLL'
        elif dc_type == 1:
            mpi.report(f"TDE SOLVER: Detected dc_type = {dc_type}, changing to 'cHeld'")
            hartree_instance.dc_type = 'cHeld'
        elif dc_type == 2 and not self.general_params['magnetic']:
            mpi.report(f"TDE SOLVER: Detected dc_type = {dc_type}, changing to 'cAMF'")
            hartree_instance.dc_type = 'cAMF'
        elif dc_type == 2 and self.general_params['magnetic']:
            mpi.report(f"TDE SOLVER: Detected dc_type = {dc_type}, changing to 'sAMF'")
            hartree_instance.dc_type = 'sAMF'

    def _build_proposal_generators(self, custom_proposals):
        """Build list of proposal generators from solver_params."""
        generators = []
        ptype = self.solver_params['proposal_type']

        if ptype in ('density_matrix', 'both'):
            dm_gen = DensityMatrixProposals(
                n_proposals=self.solver_params['n_proposals'],
                n_targeting_steps=self.solver_params['n_targeting_steps'],
                targeting_alpha=self.solver_params['targeting_alpha'],
                force_real=self.solver_params['force_real'],
                custom_proposals=custom_proposals,
            )
            generators.append(dm_gen)

        if ptype in ('sobol', 'both'):
            sobol_gen = SobolSigmaProposals(
                n_samples=self.solver_params['n_proposals'],
                sigma_bound=self.solver_params['sigma_bound'],
                offdiag_fraction=self.solver_params['offdiag_fraction'],
            )
            generators.append(sobol_gen)

        return generators

    def _init_ReFreq_tde(self):
        """Initialize ReFreq objects for self-energy storage."""
        self.n_w = self.general_params['n_w']
        self.Sigma_Refreq = self.sum_k.block_structure.create_gf(
            ish=self.icrsh, gf_function=Gf, space='solver',
            mesh=MeshReFreq(n_w=self.n_w, window=self.general_params['w_range'])
        )

    def solve(self, **kwargs):
        it = kwargs.get('it', 0)

        if it <= self.n_warmup_hf and not self._warmup_done:
            # ── HF warmup phase ──
            self._in_tde_mode = False
            mpi.report(f'\n  TDE SOLVER: HF warmup iteration {it}/{self.n_warmup_hf}')

            self.triqs_solver.G0_iw << self.G0_freq
            self.triqs_solver.solve(h_int=self.h_int, **self.triqs_solver_params)
            self._postprocess_hf()

            if it == self.n_warmup_hf:
                self._warmup_done = True
                mpi.report('  TDE SOLVER: HF warmup complete, switching to TDE mode')
        else:
            # ── TDE ensemble phase ──
            self._in_tde_mode = True
            mpi.report(f'\n  TDE SOLVER: TDE ensemble iteration {it}')

            # Copy bath into TDE solver
            for bl, gf in self.G0_freq:
                self.tde_solver.G0_iw[bl].data[:] = gf.data[:]

            # Set up warm-start from prior solutions
            if (self.solver_params['warm_start']
                    and self._prior_solutions is not None):
                for gen in self._proposal_generators:
                    if isinstance(gen, DensityMatrixProposals):
                        gen.prior_solutions = self._prior_solutions

            # Get mu and n_elec_total from solid_dmft infrastructure
            mu = self.sum_k.chemical_potential
            # Compute total electron count from the interacting G, not G0
            n_elec_total = 0.0
            for bl, gf in self.G_freq:
                n_elec_total += gf.density().real.trace()

            # ── Sanity checks before TDE solve ────────────────────────────
            if mpi.is_master_node():
                mpi.report('\n  --- TDE pre-solve sanity check ---')
                mpi.report(f'  mu                = {mu:.6f}')
                mpi.report(f'  n_elec (G_freq)   = {n_elec_total:.6f}')
                n_g0 = sum(gf.density().real.trace() for _, gf in self.G0_freq)
                mpi.report(f'  n_elec (G0_freq)  = {n_g0:.6f}')
                mpi.report(f'  dc (internal HF) = 0.0 (DC handled externally by solid_dmft through G0)')
                mpi.report('  Sigma_HF (warmup, diagonal):')
                for bl, mat in self.triqs_solver.Sigma_HF.items():
                    mpi.report(f'    [{bl}] diag = {np.diag(np.real(mat))}')
                n_tde_g0 = sum(
                    self.tde_solver.G0_iw[bl].density().real.trace()
                    for bl in self.tde_solver.blocks
                )
                mpi.report(f'  n_elec (tde G0_iw) = {n_tde_g0:.6f}')
                mpi.report('  -----------------------------------\n')

            self.tde_solver.solve(
                h_int=self.h_int,
                proposal_generators=self._proposal_generators,
                mu=mu,
                n_elec_total=n_elec_total,
                with_fock=self.solver_params['with_fock'],
                hf_method=self.solver_params['method'],
                hf_tol=self.solver_params['tol'],
                spin_map=self._spin_map,
                regret_fn=self._build_regret_fn(n_elec_total),
            )

            # Store solutions for warm-start
            if mpi.is_master_node() and self.tde_solver.solutions is not None:
                self._prior_solutions = self.tde_solver.solutions

            self._postprocess_tde()

            # Save landscape if enabled
            if self.solver_params['save_landscapes']:
                _save_landscape(
                    self.tde_solver,
                    self.general_params['jobname'],
                    it,
                    self.icrsh,
                )

    def _build_regret_fn(self, n_target: float):
        """Return S_regret(N_sol) = N_sol * (V_DC(N_sol) − V_DC(N_ref)).

        Derived from the first-order variation of the kinetic embedding action:

            δE_kin = -Tr[G δG⁻¹]  with  dG⁻¹/dV_DC = 1
                   = -Tr[G] · δV_DC = -N_sol · δV_DC

        When the bath G0 carries V_DC(N_ref) but the saddle point has occupation
        N_sol, the action contains a DC mismatch equal to:

            S_regret = N_sol · (V_DC(N_sol) - V_DC(N_ref))

        V_DC is evaluated via triqs_dft_tools.util.compute_DC_from_density using
        the same DC method as the surrounding solid_dmft calculation.  The slope
        dV_DC/dN is precomputed once (finite difference, output suppressed).

        Enabled when solver_params['regret_dc'] = true.
        """
        if not self.solver_params.get('regret_dc', False):
            return None


        dc_U_list = self.advanced_params.get('dc_U')
        dc_J_list = self.advanced_params.get('dc_J')
        dc_U = dc_U_list[self.icrsh] if dc_U_list is not None else None
        dc_J = dc_J_list[self.icrsh] if dc_J_list is not None else 0.0

        if dc_U is None:
            mpi.report('  TDE regret: dc_U not found in advanced_params — regret disabled.')
            return None

        J = dc_J if dc_J is not None else 0.0

        # Translate integer dc_type → method string (mirrors _interface_dc logic)
        dc_type_int = (self.general_params.get('dc_type') or [0])[self.icrsh]
        magnetic    = self.general_params.get('magnetic', False)
        dc_method   = {0: 'sFLL' if magnetic else 'cFLL',
                       1: 'cHeld',
                       2: 'sAMF' if magnetic else 'cAMF'}.get(dc_type_int, 'cFLL')

        n_orb = self.sum_k.gf_struct_solver_list[self.icrsh][0][1]

        def _v_dc(n):
            n_spin = n / 2.0 if magnetic else None
            v, _ = compute_DC_from_density(n, dc_U, J, N_spin=n_spin,
                                            n_orbitals=n_orb, method=dc_method)
            return v

        # Evaluate V_DC(N_ref) — one informative printed call is intentional
        v_dc_ref = _v_dc(n_target)

        # Compute dV_DC/dN via finite difference; suppress the extra DFT-tools output
        _orig_report = mpi.report
        mpi.report   = lambda *a, **k: None
        slope        = (_v_dc(n_target + 0.01) - v_dc_ref) / 0.01
        mpi.report   = _orig_report

        mpi.report(
            f'  TDE regret: method={dc_method}, U={dc_U:.4f}, J={J:.4f}, '
            f'n_orb={n_orb}, N_ref={n_target:.4f}\n'
            f'    V_DC(N_ref)={v_dc_ref:.4f},  dV_DC/dN={slope:.4f}\n'
            f'    S_regret(N_sol) = N_sol * dV_DC/dN * (N_sol - N_ref)'
        )

        def regret_fn(n_sol: float) -> float:
            return - n_sol * slope * (n_sol - n_target)

        return regret_fn

    def _postprocess_hf(self):
        """Extract results from the HF warmup solver (same as hartree_interface)."""
        self.G0_freq << self.triqs_solver.G0_iw
        self.G_freq_unsym << self.triqs_solver.G_iw
        self.G_freq << self.triqs_solver.G_iw
        self.sum_k.symm_deg_gf(self.G_freq, ish=self.icrsh)
        for bl, gf in self.Sigma_freq:
            self.Sigma_freq[bl] << self.triqs_solver.Sigma_HF[bl]
            self.Sigma_Refreq[bl] << self.triqs_solver.Sigma_HF[bl]
        self.G_time << Fourier(self.G_freq)
        self.interaction_energy = self.triqs_solver.interaction_energy()
        self.DC_energy = self.triqs_solver.DC_energy()

    def _postprocess_tde(self):
        """Extract results from the TDE ensemble solver."""
        self.G_freq_unsym << self.tde_solver.G_iw
        self.G_freq << self.tde_solver.G_iw
        self.sum_k.symm_deg_gf(self.G_freq, ish=self.icrsh)

        # TDE Sigma is frequency-dependent (not static like HF)
        for bl, gf in self.Sigma_freq:
            self.Sigma_freq[bl] << self.tde_solver.Sigma_iw[bl]
        # ReFreq Sigma: set to zero — real-freq continuation requires Padé/MaxEnt
        for bl, gf in self.Sigma_Refreq:
            self.Sigma_Refreq[bl].data[:] = 0.0

        self.G_time << Fourier(self.G_freq)

        # Compute interaction energy as weighted average over saddle points
        if mpi.is_master_node() and self.tde_solver.solutions is not None:
            self.interaction_energy = sum(
                s.weight * s.action_int for s in self.tde_solver.solutions
            )
        else:
            self.interaction_energy = 0.0
        self.interaction_energy = mpi.bcast(self.interaction_energy)

        # DC energy from the HF solver infrastructure
        self.DC_energy = self.triqs_solver.DC_energy()

    def postprocess(self):
        """Dispatch to the appropriate postprocessing based on current mode."""
        if self._in_tde_mode:
            self._postprocess_tde()
        else:
            self._postprocess_hf()
