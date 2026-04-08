"""TcalPySCF"""
import functools
import os
from typing import List, Tuple

import numpy as np
from pyscf import dft, gto, lib, scf
from pyscf.gto.basis import bse
from pyscf.tools import cubegen

from tcal.tcal import Tcal


print = functools.partial(print, flush=True)


class TcalPySCF(Tcal):
    """Calculate transfer integrals using PySCF."""
    def __init__(
        self,
        file: str,
        monomer1_atom_num: int = -1,
        method: str = 'B3LYP/6-31G(d,p)',
        charge: int = 0,
        spin: int = 0,
        use_gpu: bool = False,
        ncore: int = 4,
        max_memory_gb: int = 16,
        cart: bool = False,
        bse: bool = False,
    ) -> None:
        """Inits TcalPySCF.

        Parameters
        ----------
        file : str
            A path of xyz file.
        monomer1_atom_num : int
            Number of atoms in the first monomer. default -1
            If -1, it is half the norbitalmber of atoms in the dimer.
        method : str
            Calculation method and basis set in "METHOD/BASIS" format. default 'B3LYP/6-31G(d,p)'
        charge : int
            Molecular charge. default 0
        spin : int
            Number of unpaired electrons (2S = n_alpha - n_beta). default 0 (closed-shell singlet)
        use_gpu : bool
            If True, use GPU-accelerated SCF via gpu4pyscf (requires gpu4pyscf). default False
        ncore : int
            Number of OpenMP threads for PySCF. default 4
        max_memory_gb : int
            Maximum memory in GB for PySCF. default 16
        cart : bool
            If True, use Cartesian basis functions. default False
        bse: bool
            If True, use basis set exchange. default False
        """
        if spin != 0:
            raise NotImplementedError('TcalPySCF only supports closed-shell (spin=0) molecules.')
        super().__init__(file, monomer1_atom_num)
        self._method = method
        self._charge = charge
        self._spin = spin
        self._mf1 = None
        self._mf2 = None
        self._mf_d = None
        self._use_gpu = use_gpu
        self._ncore = ncore
        self._max_memory_gb = max_memory_gb
        self._cart = cart
        self._bse = bse
        self.extension_log = '.chk'

    def create_cube_file(self) -> None:
        """Generate NHOMO/HOMO/LUMO/NLUMO cube files for both monomers."""
        for i, (chk_suffix, mf_cache) in enumerate(
            [('_m1', self._mf1), ('_m2', self._mf2)], start=1
        ):
            if i == 1:
                print('cube file of the 1st monomer created')
            else:
                print('cube file of the 2nd monomer created')
            mol, mo_coeff = self._load_from_chk_monomer(
                chkfile=f'{self._base_path}{chk_suffix}.chk',
                mf=mf_cache,
            )
            n_occ = mol.nelectron // 2

            for idx, name in [
                (n_occ - 2, 'NHOMO'),
                (n_occ - 1, 'HOMO'),
                (n_occ,     'LUMO'),
                (n_occ + 1, 'NLUMO'),
            ]:
                outfile = f'{self._base_path}{chk_suffix}_{name}.cube'
                cubegen.orbital(mol, outfile, mo_coeff[:, idx], nx=80, ny=80, nz=80)
                print(f' {outfile}')

    def run_pyscf(self, skip_monomer_num: List[int] = [0]) -> None:
        """Run PySCF calculations for monomers and dimer.

        Parameters
        ----------
        skip_monomer_num : list[int]
            If 1 is in the list, skip 1st monomer calculation.
            If 2 is in the list, skip 2nd monomer calculation.
            If 3 is in the list, skip dimer calculation.
        """
        atoms_dimer, atoms_m1, atoms_m2 = self._parse_xyz()
        functional, basis = self._method.split('/', 1)
        if self._bse:
            unique_elements = list(set(atom[0] for atom in atoms_dimer))
            basis = bse.get_basis(basis, elements=unique_elements)

        if 1 in skip_monomer_num:
            print('skip 1st monomer calculation')
        else:
            _, self._mf1 = self._run_pyscf_calculation(
                atoms=atoms_m1,
                functional=functional,
                basis=basis,
                charge=self._charge,
                spin=self._spin,
                chkfile=f'{self._base_path}_m1.chk',
                label='1st monomer',
                use_gpu=self._use_gpu,
                ncore=self._ncore,
                max_memory_gb=self._max_memory_gb,
                cart=self._cart,
            )

        if 2 in skip_monomer_num:
            print('skip 2nd monomer calculation')
        else:
            _, self._mf2 = self._run_pyscf_calculation(
                atoms=atoms_m2,
                functional=functional,
                basis=basis,
                charge=self._charge,
                spin=self._spin,
                chkfile=f'{self._base_path}_m2.chk',
                label='2nd monomer',
                use_gpu=self._use_gpu,
                ncore=self._ncore,
                max_memory_gb=self._max_memory_gb,
                cart=self._cart,
            )

        if 3 in skip_monomer_num:
            print('skip dimer calculation')
        else:
            _, self._mf_d = self._run_pyscf_calculation(
                atoms=atoms_dimer,
                functional=functional,
                basis=basis,
                charge=self._charge,
                spin=self._spin,
                chkfile=f'{self._base_path}.chk',
                label='dimer',
                use_gpu=self._use_gpu,
                ncore=self._ncore,
                max_memory_gb=self._max_memory_gb,
                cart=self._cart,
            )

    def read_monomer1(
        self,
        is_matrix: bool = False,
        output_matrix: bool = False,
    ) -> None:
        """Extract MO coefficients from PySCF result of monomer 1.

        Parameters
        ----------
        is_matrix : bool
            If True, print MO coefficients. default False
        output_matrix : bool
            If True, output MO coefficients to CSV. default False
        """
        print(f'reading {self._base_path}_m1.chk')
        mol, mo_coeff = self._load_from_chk_monomer(
            chkfile=f'{self._base_path}_m1.chk',
            mf=self._mf1,
        )

        self.n_basis1 = mol.nao_nr()
        self.n_bsuse1 = mo_coeff.shape[1]
        self.n_elect1 = mol.nelectron // 2
        self.n_atoms1 = mol.natm
        self.mo1 = mo_coeff # shape: (n_basis1, n_bsuse1)

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
        """Extract MO coefficients from PySCF result of monomer 2.

        Parameters
        ----------
        is_matrix : bool
            If True, print MO coefficients. default False
        output_matrix : bool
            If True, output MO coefficients to CSV. default False
        """
        print(f'reading {self._base_path}_m2.chk')
        mol, mo_coeff = self._load_from_chk_monomer(
            chkfile=f'{self._base_path}_m2.chk',
            mf=self._mf2,
        )

        self.n_basis2 = mol.nao_nr()
        self.n_bsuse2 = mo_coeff.shape[1]
        self.n_elect2 = mol.nelectron // 2
        self.n_atoms2 = mol.natm
        self.mo2 = mo_coeff # shape: (n_basis2, n_bsuse2)

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
        """Extract overlap and Fock matrix from PySCF result of dimer.

        Parameters
        ----------
        is_matrix : bool
            If True, print overlap and Fock matrices. default False
        output_matrix : bool
            If True, output overlap and Fock matrices to CSV. default False
        """
        print(f'reading {self._base_path}.chk')
        mol, fock = self._load_from_chk_dimer(
            chkfile=f'{self._base_path}.chk',
            mf=self._mf_d,
        )

        self.n_basis_d = mol.nao_nr()
        self.n_elect_d = mol.nelectron // 2
        self.overlap = mol.intor('int1e_ovlp')
        self.fock = fock
        self._build_ao_labels(mol)

        # Extend monomer MOs to dimer basis (same logic as parent class)
        zeros1 = np.zeros((self.n_bsuse1, self.n_basis2))
        self.mo1 = np.hstack([self.mo1.T, zeros1])  # (n_bsuse1, n_basis_d)
        zeros2 = np.zeros((self.n_bsuse2, self.n_basis1))
        self.mo2 = np.hstack([zeros2, self.mo2.T])  # (n_bsuse2, n_basis_d)

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

    @staticmethod
    def _to_numpy(arr) -> np.ndarray:
        """Convert CuPy or NumPy array to a NumPy array.

        CuPy arrays expose ``.get()``; NumPy arrays fall through to
        ``np.asarray()``.  This avoids the "Implicit conversion to a NumPy
        array is not allowed" TypeError raised by CuPy.
        """
        if hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)

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

    def _run_pyscf_calculation(
        self,
        atoms: List,
        functional: str,
        basis: str,
        charge: int,
        spin: int,
        chkfile: str,
        label: str,
        use_gpu: bool = False,
        ncore: int = 4,
        max_memory_gb: int = 16,
        cart: bool = False,
    ) -> Tuple:
        """Build a PySCF Mole, run SCF, and save results to chkfile.

        Parameters
        ----------
        atoms : list
            List of (symbol, (x, y, z)) tuples.
        functional : str
            DFT functional or 'HF'.
        basis : str
            Basis set name.
        charge : int
            Molecular charge.
        spin : int
            Number of unpaired electrons (2S = n_alpha - n_beta).
        chkfile : str
            Path to save the checkpoint file.
        label : str
            Label for print messages.
        use_gpu : bool
            If True, convert MF object to GPU via gpu4pyscf before running. default False
        ncore : int
            Number of OpenMP threads. default 4
        max_memory_gb : int
            Maximum memory in GB. default 16
        cart : bool
            If True, use Cartesian basis functions. default False

        Returns
        -------
        tuple
            (mol, mf)
        """
        lib.num_threads(ncore)
        mol = gto.Mole()
        mol.atom = atoms
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.verbose = 0
        mol.symmetry = False
        mol.max_memory = max_memory_gb * 1000  # GB → MB
        mol.cart = cart
        mol.build()

        if use_gpu:
            try:
                if functional.upper() == 'HF':
                    from gpu4pyscf.scf import hf as gpu_hf
                    mf = gpu_hf.RHF(mol)
                else:
                    from gpu4pyscf.dft import rks as gpu_rks
                    mf = gpu_rks.RKS(mol)
                    mf.xc = functional
            except ImportError:
                print('Error: gpu4pyscf is not installed.')
                print('gpu4pyscf is supported on macOS/Linux/WSL2 only.')
                print('')
                print('Install options:')
                print('  GPU (CUDA 12):  pip install "yu-tcal[gpu4pyscf-cuda12]"')
                print('  GPU (CUDA 11):  pip install "yu-tcal[gpu4pyscf-cuda11]"')
                exit(1)
        else:
            if functional.upper() == 'HF':
                mf = scf.RHF(mol)
            else:
                mf = dft.RKS(mol)
                mf.xc = functional
        mf.chkfile = chkfile  # CPU のみ: kernel() 時に numpy 配列を自動保存

        print(f'running {label} calculation')
        mf.kernel()

        if not mf.converged:
            print(f'WARNING: SCF did not converge for {label}')
        else:
            print(f'{label} calculation completed')
            lib.chkfile.save(chkfile, 'job_status/completed', True)
            print(f' {chkfile}')

        lib.chkfile.save(chkfile, 'tcal/fock', self._to_numpy(mf.get_fock()))

        return mol, mf

    def _build_ao_labels(self, mol: gto.Mole) -> None:
        """Set atom_index, atom_symbol, and atom_orbital from PySCF ao_labels.

        Parameters
        ----------
        mol : gto.Mole
            PySCF Mole object for the dimer.
        """
        ao_labels = mol.ao_labels(fmt=False)
        # Each entry: (iatom: int, atom_sym: str, nl: str, ml: str)
        self.atom_index = []
        self.atom_symbol = []
        self.atom_orbital = []

        for iatom, atom_sym, nl, ml in ao_labels:
            self.atom_index.append(iatom)
            self.atom_symbol.append(atom_sym)
            self.atom_orbital.append(nl + ml)

    def _load_from_chk_monomer(self, chkfile: str, mf) -> Tuple:
        """Load mol and mo_coeff from memory or chkfile.

        Parameters
        ----------
        chkfile : str
            Path to the checkpoint file.
        mf : SCF object or None
            Cached SCF result. If None, load from chkfile.

        Returns
        -------
        tuple
            (mol, mo_coeff)
        """
        if mf is not None:
            return mf.mol, self._to_numpy(mf.mo_coeff)

        if not os.path.exists(chkfile):
            raise FileNotFoundError(f'{chkfile} not found. Run run_pyscf() first.')

        mol = lib.chkfile.load_mol(chkfile)
        mo_coeff = lib.chkfile.load(chkfile, 'scf/mo_coeff')
        return mol, mo_coeff

    def _load_from_chk_dimer(self, chkfile: str, mf) -> Tuple:
        """Load mol and fock from memory or chkfile.

        Parameters
        ----------
        chkfile : str
            Path to the checkpoint file.
        mf : SCF object or None
            Cached SCF result. If None, load from chkfile.

        Returns
        -------
        tuple
            (mol, fock)
        """
        if mf is not None:
            return mf.mol, self._to_numpy(mf.get_fock())

        if not os.path.exists(chkfile):
            raise FileNotFoundError(f'{chkfile} not found. Run run_pyscf() first.')

        mol = lib.chkfile.load_mol(chkfile)
        fock = lib.chkfile.load(chkfile, 'tcal/fock')
        return mol, fock
