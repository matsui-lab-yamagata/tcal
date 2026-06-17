"""TcalORCA"""
import functools
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from tcal.tcal import Tcal


print = functools.partial(print, flush=True)


class TcalORCA(Tcal):
    """Calculate transfer integrals using ORCA via OPI (orca-pi)."""

    def __init__(
        self,
        file: str,
        monomer1_atom_num: int = -1,
        method: str = 'B3LYP/6-31G(d,p)',
        charge: int = 0,
        spin: int = 0,
        ncore: int = 4,
        max_memory_mb: int = 16 * 1024, # GB -> MB
        open_mpi_path: Optional[str] = None,
    ) -> None:
        """Inits TcalORCA.

        Parameters
        ----------
        file : str
            A path of xyz file.
        monomer1_atom_num : int
            Number of atoms in the first monomer. default -1
            If -1, it is half the number of atoms in the dimer.
        method : str
            Calculation method and basis set in "METHOD/BASIS" format. default 'B3LYP/6-31G(d,p)'
        charge : int
            Molecular charge. default 0
        spin : int
            Number of unpaired electrons (2S = n_alpha - n_beta). default 0 (closed-shell singlet)
        ncore : int
            Number of parallel processes for ORCA (%pal nprocs). default 4
        max_memory_mb : int
            Maximum memory in MB for ORCA (%maxcore). default 16384 (16 GB)
        open_mpi_path : str, optional
            Path to the OpenMPI installation directory (e.g. '/usr/lib/x86_64-linux-gnu/openmpi').
            Required when OpenMPI is not in $PATH and the OPI_MPI environment variable is not set.
            Sets OPI_MPI temporarily during ORCA execution.
        """
        if spin != 0:
            raise NotImplementedError('TcalORCA only supports closed-shell (spin=0) molecules.')
        super().__init__(file, monomer1_atom_num)
        self._method = method
        self._charge = charge
        self._spin = spin
        self._ncore = ncore
        self._max_memory_mb = max_memory_mb
        self._open_mpi_path = open_mpi_path
        self.extension_log = '.out'
        self._output1 = None  # cache for monomer 1 OPI Output object
        self._output2 = None  # cache for monomer 2 OPI Output object
        self._output_d = None  # cache for dimer OPI Output object

    def run_orca(self, skip_monomer_num: List[int] = [0], verbose: bool = True) -> None:
        """Run ORCA calculations for monomers and dimer.

        Parameters
        ----------
        skip_monomer_num : list[int]
            If 1 is in the list, skip 1st monomer calculation.
            If 2 is in the list, skip 2nd monomer calculation.
            If 3 is in the list, skip dimer calculation.
        verbose : bool
            If False, suppress all progress/status output (including warnings). default True
        """
        log = print if verbose else lambda *args, **kwargs: None
        atoms_dimer, atoms_m1, atoms_m2 = self._parse_xyz()
        functional, basis = self._method.split('/', 1)

        working_dir = Path(self._base_path).parent.resolve()
        stem = Path(self._base_path).name

        if 1 in skip_monomer_num:
            log('skip 1st monomer calculation')
        else:
            self._output1 = self._run_orca_calculation(
                atoms=atoms_m1,
                functional=functional,
                basis=basis,
                basename=f'{stem}_m1',
                working_dir=working_dir,
                label='1st monomer',
                verbose=verbose,
            )

        if 2 in skip_monomer_num:
            log('skip 2nd monomer calculation')
        else:
            self._output2 = self._run_orca_calculation(
                atoms=atoms_m2,
                functional=functional,
                basis=basis,
                basename=f'{stem}_m2',
                working_dir=working_dir,
                label='2nd monomer',
                verbose=verbose,
            )

        if 3 in skip_monomer_num:
            log('skip dimer calculation')
        else:
            self._output_d = self._run_orca_calculation(
                atoms=atoms_dimer,
                functional=functional,
                basis=basis,
                basename=stem,
                working_dir=working_dir,
                label='dimer',
                verbose=verbose,
            )

    def read_monomer1(
        self,
        is_matrix: bool = False,
        output_matrix: bool = False,
    ) -> None:
        """Extract MO coefficients from ORCA result of monomer 1.

        Parameters
        ----------
        is_matrix : bool
            If True, print MO coefficients. default False
        output_matrix : bool
            If True, output MO coefficients to CSV. default False
        """
        print(f'reading {self._base_path}_m1.out')
        working_dir = Path(self._base_path).parent.resolve()
        stem = Path(self._base_path).name
        output = self._output1 or self._load_output(f'{stem}_m1', working_dir)

        mos = output.get_mos()['mo']
        # shape: (n_bsuse1, n_basis1)  — rows = MOs, columns = AO basis functions
        mo_coeff = np.array([mo.mocoefficients for mo in mos])

        self.n_bsuse1 = mo_coeff.shape[0]
        self.n_basis1 = mo_coeff.shape[1]
        nel, _ = output.get_nelectrons()
        self.n_elect1 = nel // 2
        atoms_list = output._safe_get('results_gbw', 0, 'molecule', 'atoms') or []
        self.n_atoms1 = len(atoms_list)
        self.mo1 = mo_coeff

        if is_matrix:
            print(' *** Alpha MO coefficients *** ')
            self.print_matrix(self.mo1)

        if output_matrix:
            self.output_csv(f'{self._base_path}_mo1.csv', self.mo1)

    def read_monomer2(
        self,
        is_matrix: bool = False,
        output_matrix: bool = False,
    ) -> None:
        """Extract MO coefficients from ORCA result of monomer 2.

        Parameters
        ----------
        is_matrix : bool
            If True, print MO coefficients. default False
        output_matrix : bool
            If True, output MO coefficients to CSV. default False
        """
        print(f'reading {self._base_path}_m2.out')
        working_dir = Path(self._base_path).parent.resolve()
        stem = Path(self._base_path).name
        output = self._output2 or self._load_output(f'{stem}_m2', working_dir)

        mos = output.get_mos()['mo']
        # shape: (n_bsuse2, n_basis2)  — rows = MOs, columns = AO basis functions
        mo_coeff = np.array([mo.mocoefficients for mo in mos])

        self.n_bsuse2 = mo_coeff.shape[0]
        self.n_basis2 = mo_coeff.shape[1]
        nel, _ = output.get_nelectrons()
        self.n_elect2 = nel // 2
        atoms_list = output._safe_get('results_gbw', 0, 'molecule', 'atoms') or []
        self.n_atoms2 = len(atoms_list)
        self.mo2 = mo_coeff

        if is_matrix:
            print(' *** Alpha MO coefficients *** ')
            self.print_matrix(self.mo2)

        if output_matrix:
            self.output_csv(f'{self._base_path}_mo2.csv', self.mo2)

    def read_dimer(
        self,
        is_matrix: bool = False,
        output_matrix: bool = False,
    ) -> None:
        """Extract overlap and Fock matrix from ORCA result of dimer.

        Parameters
        ----------
        is_matrix : bool
            If True, print overlap and Fock matrices. default False
        output_matrix : bool
            If True, output overlap and Fock matrices to CSV. default False
        """
        print(f'reading {self._base_path}.out')
        working_dir = Path(self._base_path).parent.resolve()
        stem = Path(self._base_path).name
        output = self._output_d or self._load_output(stem, working_dir)

        self.n_basis_d = output.get_nbf()
        nel, _ = output.get_nelectrons()
        self.n_elect_d = nel // 2

        # Step 1: Restore default GBW JSON (MO data, no matrices).
        # orca_2json with a config dict produces a matrix-only JSON that omits
        # MolecularOrbitals/OrbitalLabels. Passing an empty config recreates the
        # default JSON which always includes MO data.
        output.recreate_gbw_results({})

        # Step 2: Extract AO labels and MO data while the default JSON is active.
        self._build_ao_labels_from_orca(output)
        mos = output.get_mos()['mo']
        C = np.array([mo.mocoefficients for mo in mos]).T   # (n_basis_d, n_mos)
        mo_energies = np.array([mo.orbitalenergy for mo in mos])  # Hartree

        # Step 3: Recreate the GBW JSON with all needed matrices in ONE call.
        # This overwrites the MO data in the in-memory JSON, which is fine because
        # we already stored C and mo_energies above.
        _matrix_config = {"1elIntegrals": ["S", "H"], "FockMatrix": ["F"]}
        output.recreate_gbw_results(_matrix_config)

        # Step 4: Read matrices from the now-active matrix-containing JSON.
        self.overlap = output.get_int_overlap()
        h_core = output.get_int_hcore()
        g_mat = output.get_int_f()

        # Step 5: Build full Fock/KS matrix: F = H_core + G
        # get_int_f() returns only the two-electron part G (J - αK + V_XC).
        # Fallback: reconstruct F_AO = S C ε C^T S from canonical MOs.
        if h_core is not None and g_mat is not None:
            self.fock = h_core + g_mat
        else:
            self.fock = self.overlap @ C @ np.diag(mo_energies) @ C.T @ self.overlap

        # Extend monomer MOs to dimer basis by zero-padding.
        # mo1/mo2 are already (n_bsuse, n_basis) so no transpose is needed,
        # unlike TcalPySCF which stores (n_basis, n_bsuse) and transposes here.
        zeros1 = np.zeros((self.n_bsuse1, self.n_basis2))
        self.mo1 = np.hstack([self.mo1, zeros1])  # (n_bsuse1, n_basis_d)
        zeros2 = np.zeros((self.n_bsuse2, self.n_basis1))
        self.mo2 = np.hstack([zeros2, self.mo2])  # (n_bsuse2, n_basis_d)

        if is_matrix:
            print()
            print(' *** Overlap *** ')
            self.print_matrix(self.overlap)
            print()
            print(' *** Fock matrix (alpha) *** ')
            self.print_matrix(self.fock)

        if output_matrix:
            self.output_csv(f'{self._base_path}_overlap.csv', self.overlap)
            self.output_csv(f'{self._base_path}_fock.csv', self.fock)

    def create_cube_file(self, resolution: int = 40) -> None:
        """Create cube files for NHOMO, HOMO, LUMO, NLUMO of monomers 1 and 2.

        Parameters
        ----------
        resolution : int
            Grid resolution for orca_plot. Higher values produce smoother plots
            but take longer. default 40
        """
        working_dir = Path(self._base_path).parent.resolve()
        stem = Path(self._base_path).name

        for monomer_num, (output_cache, monomer_stem) in enumerate([
            (self._output1, f'{stem}_m1'),
            (self._output2, f'{stem}_m2'),
        ], start=1):
            output = output_cache or self._load_output(monomer_stem, working_dir)
            nel, _ = output.get_nelectrons()
            n_occ = nel // 2

            orbitals = [
                ('NHOMO', n_occ - 2),
                ('HOMO',  n_occ - 1),
                ('LUMO',  n_occ),
                ('NLUMO', n_occ + 1),
            ]

            ordinal = '1st' if monomer_num == 1 else '2nd'
            print(f'generating cube files for {ordinal} monomer')
            for label, idx in orbitals:
                cube_out = output.plot_mo(idx, resolution=resolution)
                if cube_out is None:
                    print(f'WARNING: failed to generate cube file for {label}')
                    continue
                target = working_dir / f'{monomer_stem}_{label}.cube'
                cube_out.path.rename(target)
                print(f' {target}')

            print(f'cube file of the {ordinal} monomer created')

    def _parse_xyz(self) -> Tuple[List, List, List]:
        """Parse XYZ file and split into dimer, monomer 1, and monomer 2 atoms.

        Returns
        -------
        tuple
            (atoms_dimer, atoms_m1, atoms_m2) where each is a list of (symbol, (x, y, z))
        """
        atoms_dimer = []
        with open(f'{self._base_path}.xyz', 'r', encoding='utf-8') as f:
            f.readline()  # skip atom count line
            f.readline()  # skip comment line
            while True:
                line = f.readline()
                if not line:
                    break
                parts = line.strip().split()
                if len(parts) == 4:
                    symbol = parts[0]
                    x, y, z = map(float, parts[1:4])
                    atoms_dimer.append((symbol, (x, y, z)))

        if self.monomer1_atom_num == -1:
            n_m1 = len(atoms_dimer) // 2
        else:
            n_m1 = self.monomer1_atom_num

        return atoms_dimer, atoms_dimer[:n_m1], atoms_dimer[n_m1:]

    def _run_orca_calculation(
        self,
        atoms: List,
        functional: str,
        basis: str,
        basename: str,
        working_dir: Path,
        label: str,
        verbose: bool = True,
    ):
        """Set up and run an ORCA single-point calculation via OPI.

        Parameters
        ----------
        atoms : list
            List of (symbol, (x, y, z)) tuples.
        functional : str
            DFT functional name (e.g. 'B3LYP') or 'HF'.
        basis : str
            Basis set name (e.g. '6-31G(d,p)').
        basename : str
            Basename for ORCA input/output files.
        working_dir : Path
            Directory where ORCA files will be written.
        label : str
            Human-readable label for progress messages.
        verbose : bool
            If False, suppress all progress/status output (including warnings). default True

        Returns
        -------
        Output
            OPI Output object with parsed results.
        """
        log = print if verbose else lambda *args, **kwargs: None
        from opi.core import Calculator
        from opi.input.structures.structure import Structure

        # Write a temporary XYZ file for the structure
        xyz_path = working_dir / f'{basename}_input.xyz'
        with open(xyz_path, 'w', encoding='utf-8') as f:
            f.write(f'{len(atoms)}\n\n')
            for symbol, (x, y, z) in atoms:
                f.write(f'{symbol}  {x}  {y}  {z}\n')

        structure = Structure.from_xyz(
            xyz_path,
            charge=self._charge,
            multiplicity=self._spin + 1,
        )

        calc = Calculator(basename=basename, working_dir=working_dir, version_check=False)
        calc.structure = structure
        calc.input.add_arbitrary_string(f'! {functional} {basis} SP\n')
        calc.input.add_arbitrary_string(f'%pal nprocs {self._ncore} end\n')
        calc.input.add_arbitrary_string(f'%maxcore {self._max_memory_mb}\n')

        log(f'running {label} calculation')
        if self._open_mpi_path is not None:
            _prev_opi_mpi = os.environ.get('OPI_MPI')
            os.environ['OPI_MPI'] = self._open_mpi_path
        try:
            calc.write_and_run()
        finally:
            if self._open_mpi_path is not None:
                if _prev_opi_mpi is None:
                    del os.environ['OPI_MPI']
                else:
                    os.environ['OPI_MPI'] = _prev_opi_mpi

        output = calc.get_output()
        output.parse()

        if not output.terminated_normally():
            log(f'WARNING: {label} calculation did not terminate normally.')
        if not output.scf_converged():
            log(f'WARNING: {label} SCF did not converge.')
        else:
            log(f'{label} calculation completed')
            log(f' {working_dir / (basename + ".out")}')

        return output

    def _load_output(self, basename: str, working_dir: Path):
        """Load and parse an existing ORCA output from disk.

        Parameters
        ----------
        basename : str
            Basename of the ORCA job.
        working_dir : Path
            Directory containing the ORCA output files.

        Returns
        -------
        Output
            Parsed OPI Output object.

        Raises
        ------
        FileNotFoundError
            If the output file does not exist.
        """
        from opi.output.core import Output

        out_path = working_dir / f'{basename}.out'
        if not out_path.exists():
            raise FileNotFoundError(
                f'{out_path} not found. Run run_orca() first or check the file path.'
            )

        output = Output(basename=basename, working_dir=working_dir, version_check=False)
        output.parse()
        return output

    def _build_ao_labels_from_orca(self, output) -> None:
        """Populate atom_index, atom_symbol, and atom_orbital from ORCA orbital labels.

        Parameters
        ----------
        output : Output
            Parsed OPI Output object for the dimer.
        """
        labels = output._safe_get(
            'results_gbw', 0, 'molecule', 'molecularorbitals', 'orbitallabels'
        )
        if labels is None:
            return

        # Each label string is formatted as "{atom_0idx}{element}   {orbital_type}"
        # e.g. "0C   1s", "0C   2s", "0C   1pz", "12H   1s"
        # atom_0idx is 0-indexed; element follows immediately with no space.
        pattern = re.compile(r'^(\d+)([A-Za-z]+)\s+(.+)$')
        self.atom_index = []
        self.atom_symbol = []
        self.atom_orbital = []

        for label in labels:
            m = pattern.match(label)
            if m:
                atom_0idx = int(m.group(1))
                elem = m.group(2)
                orb = m.group(3).strip()
                self.atom_index.append(atom_0idx)  # already 0-indexed
                self.atom_symbol.append(elem)
                self.atom_orbital.append(orb)
