"""Tcal"""
import argparse
import csv
import functools
import math
import os
import platform
import re
import subprocess
import traceback
from datetime import datetime
from time import time
from typing import List, Literal, Optional, TextIO, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


print = functools.partial(print, flush=True)


def main():
    """This code is to execute tcal for command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='file name', type=str)
    parser.add_argument(
        '-a', '--apta', help='perform atomic pair transfer analysis', action='store_true'
    )
    parser.add_argument('-c', '--cube', help='generate cube files', action='store_true')
    # parser.add_argument('-d', '--debug', help='enable detailed output', action='store_true')
    parser.add_argument(
        '-g', '--g09', help='use Gaussian 09 (default is Gaussian 16)', action='store_true'
    )
    parser.add_argument(
        '-l', '--lumo', help='perform atomic pair transfer analysis of LUMO', action='store_true'
    )
    parser.add_argument(
        '-m', '--matrix', help='print MO coefficients, overlap matrix and Fock matrix', action='store_true'
    )
    parser.add_argument(
        '-o', '--output', action='store_true',
        help='output csv file on the result of apta and transfer integrals between diffrent orbitals etc.',
    )
    parser.add_argument(
        '-r', '--read', help='read log files without executing Gaussian', action='store_true'
    )
    parser.add_argument('-x', '--xyz', help='convert xyz to gjf', action='store_true')
    parser.add_argument(
        '--napta', type=int, nargs=2, metavar=('N1', 'N2'),
        help='perform atomic pair transfer analysis between different levels. ' \
        'N1 is the number of level in the first monomer. N2 is the number of level in the second monomer.'
    )
    parser.add_argument(
        '--hetero', type=int, default=-1, metavar='N',
        help='calculate the transfer integral of heterodimer. N is the number of atoms in the first monomer.'
    )
    parser.add_argument(
        '--nlevel', type=int, metavar='N',
        help='calculate transfer integrals between different levels. N is the number of levels from HOMO-LUMO. ' \
        + 'N=0 gives all levels.'
    )
    parser.add_argument(
        '--skip', type=int, nargs='+', default=[0], metavar='N', choices=[1, 2, 3],
        help='skip specified Gaussian calculation. If N is 1, skip 1st monomer calculation. ' \
        + 'If N is 2, skip 2nd monomer calculation. If N is 3, skip dimer calculation.'
    )
    args = parser.parse_args()

    print('----------------------------------------')
    print(' tcal 3.1.0 (2026/01/20) by Matsui Lab. ')
    print('----------------------------------------')
    print(f'\nInput File Name: {args.file}')
    Tcal.print_timestamp()
    before = time()
    print()

    tcal = Tcal(args.file, monomer1_atom_num=args.hetero)

    # convert xyz to gjf
    if args.xyz:
        tcal.convert_xyz_to_gjf()

    if not args.read:
        tcal.create_monomer_file()

        if args.g09:
            gaussian_command = 'g09'
        else:
            gaussian_command = 'g16'
        res = tcal.run_gaussian(gaussian_command, skip_monomer_num=args.skip)

        if res:
            print()
            Tcal.print_timestamp()
            after = time()
            print(f'Elapsed Time: {(after - before) * 1000:.0f} ms')
            exit()

    try:
        tcal.check_extension_log()
        tcal.read_monomer1(args.matrix)
        tcal.read_monomer2(args.matrix)
        tcal.read_dimer(args.matrix)

        if args.cube:
            tcal.create_cube_file()

        tcal.print_transfer_integrals()
        if args.nlevel is not None:
            tcal.print_transfer_integral_diff_levels(args.nlevel, output_ti_diff_levels=args.output)

        if args.apta:
            analyze_orbital = 'HOMO'
        elif args.lumo:
            analyze_orbital = 'LUMO'

        if args.napta:
            apta = tcal.custom_atomic_pair_transfer_analysis(
                analyze_orb1=args.napta[0], analyze_orb2=args.napta[1], output_apta=args.output
            )
            pair_analysis = PairAnalysis(apta)
            pair_analysis.print_largest_pairs()
            pair_analysis.print_element_pairs()
        elif args.apta or args.lumo:
            apta = tcal.atomic_pair_transfer_analysis(analyze_orbital, output_apta=args.output)
            pair_analysis = PairAnalysis(apta)
            pair_analysis.print_largest_pairs()
            pair_analysis.print_element_pairs()
        print()
    except:
        print(traceback.format_exc().strip())

    Tcal.print_timestamp()
    after = time()
    print(f'Elapsed Time: {(after - before) * 1000:.0f} ms')


class Tcal:
    """Calculate transfer integrals."""
    EV: float = 4.35974417e-18 / 1.60217653e-19 * 1000.0

    def __init__(self, file: str, monomer1_atom_num: int = -1) -> None:
        """Inits TcalClass.

        Parameters
        ----------
        file : str
            A path of gjf file.
        monomer1_atom_num: int
            Number of atoms in the first monomer. default -1
            If monomer1_atom_num is -1, it is half the number of atoms in the dimer.
        """
        if platform.system() == 'Windows':
            self._extension_log = '.out'
        else:
            self._extension_log = '.log'

        self._base_path = os.path.splitext(file)[0]
        self.monomer1_atom_num = monomer1_atom_num

        self.n_basis1 = None
        self.n_elect1 = None
        self.n_bsuse1 = None
        self.n_basis2 = None
        self.n_elect2 = None
        self.n_bsuse2 = None
        self.n_basis_d = None
        self.n_elect_d = None
        self.mo1 = None
        self.mo2 = None
        self.overlap = None
        self.fock = None
        self.atom_index = None
        self.atom_symbol = None
        self.atom_orbital = None

        self.n_atoms1 = None
        self.n_atoms2 = None

    @staticmethod
    def cal_transfer_integrals(
        bra: NDArray[np.float64],
        overlap: NDArray[np.float64],
        fock: NDArray[np.float64],
        ket: NDArray[np.float64]
    ) -> float:
        """Calculate intermolecular transfer integrals.

        Parameters
        ----------
        bra : numpy.array
            MO coefficients of one molecule.
        overlap : numpy.array
            Overlap matrix of dimer.
        fock : numpy.array
            Fock matrix of dimer.
        ket : numpy.array
            MO coefficients of the other molecule.

        Returns
        -------
        float
            Intermolecular transfer integrals.
        """
        s11 = bra @ overlap @ bra
        s22 = ket @ overlap @ ket
        s12 = bra @ overlap @ ket
        f11 = bra @ fock @ bra
        f22 = ket @ fock @ ket
        f12 = bra @ fock @ ket

        if abs(s11 - 1) > 1e-2:
            print(f"WARNING! Self overlap is not unity: S11 = {s11}")
        if abs(s22 - 1) > 1e-2:
            print(f"WARNING! Self overlap is not unity: S22 = {s22}")
        transfer = ((f12 - 0.5 * (f11 + f22) * s12) / (1 - s12 * s12)) * Tcal.EV
        return transfer

    @staticmethod
    def check_normal_termination(reader: TextIO) -> TextIO:
        """Whether the calculation of gaussian was successful or not.

        Parameters
        ----------
        reader : _io.TextIOWrapper
            Return value of open function.

        Returns
        -------
        _io.TextIOWrapper
            Return value of function.

        Examples
        --------
        >>> with open('sample.log', 'r') as f:
        ...     f = Tcal.check_normal_termination(f)
        """
        while True:
            line = reader.readline()
            if not line:
                return reader
            if 'Normal termination' in line:
                return reader

    @staticmethod
    def extract_coordinates(reader: TextIO) -> Tuple[TextIO, List[str]]:
        """Extract coordinates from gjf file of dimer.

        Parameters
        ----------
        reader : _io.TextIOWrapper
            Return value of open function.

        Returns
        -------
        _io.TextIOWrapper
            Return value of open function.
        list
            The list of coordinates.

        Examples
        --------
        >>> import re
        >>> with open(f'sample.gjf', 'r') as f:
        ...     while True:
        ...         line = f.readline()
        ...         if not line:
        ...             break
        ...         if re.search('[-0-9]+ [0-3]', line):
        ...             f, coordinates = Tcal.extract_coordinates(f)
        """
        coordinates = []
        while True:
            line = reader.readline()
            if not line.strip():
                return reader, coordinates
            coordinates.append(line)

    @staticmethod
    def extract_num(pattern: str, line: str, idx: int = 0) -> Optional[int]:
        """Extract integer in strings.

        Parameters
        ----------
        pattern : str
            Strings using regular expression.
        line : str
            String of target.
        idx : int
            Index of integer. default 0

        Returns
        -------
        int or None
            If there is integer, return it.
        """
        res = re.search(pattern, line)
        if res:
            return int(res.group().split()[idx])
        return None

    @staticmethod
    def output_csv(file_name: str, array: ArrayLike) -> None:
        """Output csv file of array.

        Parameters
        ----------
        file_name : str
            File name including extension.
        array : array_like
            Array to create csv file.
        """
        with open(file_name, 'w', encoding='UTF-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(array)

    @staticmethod
    def print_matrix(matrix: ArrayLike) -> None:
        """Print matrix.

        Parameters
        ----------
        matrix : array_like
        """
        for i, row in enumerate(matrix):
            for j, cell in enumerate(row[:-1]):
                if i == 0 or j == 0:
                    print(f'{cell:^9}', end='\t')
                else:
                    print(f'{cell:>9}', end='\t')
            if i == 0:
                print(f'{row[-1]:^9}')
            else:
                print(f'{row[-1]:>9}')

    @staticmethod
    def print_timestamp() -> None:
        """Print timestamp."""
        month = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec',
        }
        dt_now = datetime.now()
        print(f"Timestamp: {dt_now.strftime('%a')} {month[dt_now.month]} {dt_now.strftime('%d %H:%M:%S %Y')}")

    @staticmethod
    def read_matrix(reader: TextIO, n_basis: int, n_bsuse: int) -> NDArray[np.float64]:
        """Read matrix.

        Parameters
        ----------
        reader : _io.TextIOWrapper
            Return value of open function.
        n_basis : int
            The number of row.
        n_bsuse : int
            The number of column.

        Returns
        -------
        numpy.array
            Read matrix like MO coefficients.
        """
        mat = np.zeros((n_basis, n_bsuse), dtype=np.float64)
        for i in range(math.ceil(n_bsuse/5)):
            if 'Alpha density matrix' in reader.readline():
                break
            for j in range(n_basis):
                for k, val in enumerate(reader.readline().split()[1:]):
                    mat[j][i*5+k] = float(val.strip().replace('D', 'E'))

        return mat

    @staticmethod
    def read_symmetric_matrix(reader: TextIO, n_basis: int) -> NDArray[np.float64]:
        """Read symmetric matrix.

        Parameters
        ----------
        reader : _io.TextIOWrapper
            Return value of open function.
        n_basis : int
            The number of column or row.

        Returns
        -------
        numpy.array
            Read symmetrix matrix like overlap or fock matrix.
        """
        mat = np.zeros((n_basis, n_basis), dtype=np.float64)
        for i in range(math.ceil(n_basis/5)):
            reader.readline()
            for j in range(i*5, n_basis):
                for k, val in enumerate(reader.readline().split()[1:]):
                    val = float(val.strip().replace('D', 'E'))
                    mat[j][i*5+k] = val
                    mat[i*5+k][j] = val

        return mat

    def atomic_pair_transfer_analysis(
        self,
        analyze_orbital: Literal['HOMO', 'LUMO'] = 'HOMO',
        output_apta: bool = False
    ) -> NDArray[np.float64]:
        """Calculate atomic pair transfer integrals.

        Parameters
        ----------
        analyze_orbital : str
            Analyze orbital., default 'HOMO'
        output_apta : bool
            If it is True, output csv file of atomic pair transfer integrals., default False

        Returns
        -------
        numpy.array
            The array of atomic pair transfer analysis.
        """
        if analyze_orbital.upper() == 'LUMO':
            orb1 = self.mo1[self.n_elect1]
            orb2 = self.mo2[self.n_elect2]
        else:
            orb1 = self.mo1[self.n_elect1 - 1]
            orb2 = self.mo2[self.n_elect2 - 1]

        f11 = orb1 @ self.fock @ orb1
        f22 = orb2 @ self.fock @ orb2
        s12 = orb1 @ self.overlap @ orb2

        sum_f = f11 + f22
        pow_s12 = 1 - s12*s12
        atom_num = len(set(self.atom_index))
        a_transfer = np.zeros((atom_num, atom_num))
        for i in range(self.n_basis_d):
            for j in range(self.n_basis_d):
                orb_transfer = orb1[i] * orb2[j] * (self.fock[i][j] - 0.5 * sum_f * self.overlap[i][j]) / pow_s12
                a_transfer[self.atom_index[i]][self.atom_index[j]] += orb_transfer

        apta = self.print_apta(a_transfer)

        if output_apta:
            self.output_csv(f'{self._base_path}_apta_{analyze_orbital}.csv', apta)

        return apta

    def check_extension_log(self) -> None:
        """Check the extension of log file."""
        if os.path.exists(f'{self._base_path}.out'):
            self._extension_log = '.out'
        else :
            self._extension_log = '.log'

    def convert_xyz_to_gjf(
        self,
        function: str = 'B3LYP/6-31G(d,p)',
        nprocshared: int = 4,
        mem: int = 16,
        unit: str = 'GB'
    ) -> None:
        """Convert xyz file to gjf file.

        Parameters
        ----------
        function : str
            Gaussian calculation method and basis set., default 'b3lyp/6-31g(d,p)'
        nprocshared : int
            The number of nprocshared., default 4
        mem : int
            The number of memory., default 16
        unit : str
            The unit of memory., default 'GB'
        """
        coordinates = []
        with open(f'{self._base_path}.xyz', 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line_list = line.strip().split()
                if len(line_list) == 4:
                    symbol, x, y, z = line_list
                    try:
                        x, y, z = map(float, [x, y, z])
                        x = f'{x:.8f}'.rjust(14, ' ')
                        y = f'{y:.8f}'.rjust(14, ' ')
                        z = f'{z:.8f}'.rjust(14, ' ')
                        coordinates.append(f' {symbol}     {x} {y} {z}\n')
                    except ValueError:
                        continue

        with open(f'{self._base_path}.gjf', 'w') as f:
            f.write(f'%Chk={self._base_path}.chk')
            f.write('\n')
            f.write(f'%NProcShared={nprocshared}')
            f.write('\n')
            f.write(f'%Mem={mem}{unit}\n')
            f.write(f'# {function}\n')
            f.write('# Symmetry=None\n')
            f.write('\n')
            f.write(f'{self._base_path}.gjf created by tcal\n')
            f.write('\n')
            f.write('0 1\n')

            for coordinate in coordinates:
                f.write(coordinate)

            f.write('\n')
            f.write('--Link1--\n')
            f.write(f'%Chk={self._base_path}.chk')
            f.write('\n')
            f.write(f'%NProcShared={nprocshared}')
            f.write('\n')
            f.write(f'%Mem={mem}{unit}\n')
            f.write(f'# {function}\n')
            f.write('# Symmetry=None\n')
            f.write('# Geom=AllCheck\n')
            f.write('# Guess=Read\n')
            f.write('# Pop=Full\n')
            f.write('# IOp(3/33=4,5/33=3)\n')
            f.write('\n')

    def create_cube_file(self) -> None:
        """Create cube file."""
        self._execute(['formchk', f'{self._base_path}.chk', f'{self._base_path}.fchk'])
        self._create_dummy_cube_file()
        self._execute(['formchk', f'{self._base_path}_m1.chk', f'{self._base_path}_m1.fchk'])
        self._execute(['cubegen', '0', f'mo={self.n_elect1-1}', f'{self._base_path}_m1.fchk', f'{self._base_path}_m1_NHOMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        self._execute(['cubegen', '0', f'mo={self.n_elect1}', f'{self._base_path}_m1.fchk', f'{self._base_path}_m1_HOMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        self._execute(['cubegen', '0', f'mo={self.n_elect1+1}', f'{self._base_path}_m1.fchk', f'{self._base_path}_m1_LUMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        self._execute(['cubegen', '0', f'mo={self.n_elect1+2}', f'{self._base_path}_m1.fchk', f'{self._base_path}_m1_NLUMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        print('cube file of the 1st monomer created')
        print(f' {self._base_path}_m1_NHOMO.cube')
        print(f' {self._base_path}_m1_HOMO.cube')
        print(f' {self._base_path}_m1_LUMO.cube')
        print(f' {self._base_path}_m1_NLUMO.cube')

        self._execute(['formchk', f'{self._base_path}_m2.chk', f'{self._base_path}_m2.fchk'])
        self._execute(['cubegen', '0', f'mo={self.n_elect2-1}', f'{self._base_path}_m2.fchk', f'{self._base_path}_m2_NHOMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        self._execute(['cubegen', '0', f'mo={self.n_elect2}', f'{self._base_path}_m2.fchk', f'{self._base_path}_m2_HOMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        self._execute(['cubegen', '0', f'mo={self.n_elect2+1}', f'{self._base_path}_m2.fchk', f'{self._base_path}_m2_LUMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        self._execute(['cubegen', '0', f'mo={self.n_elect2+2}', f'{self._base_path}_m2.fchk', f'{self._base_path}_m2_NLUMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        print('cube file of the 2nd monomer created')
        print(f' {self._base_path}_m2_NHOMO.cube')
        print(f' {self._base_path}_m2_HOMO.cube')
        print(f' {self._base_path}_m2_LUMO.cube')
        print(f' {self._base_path}_m2_NLUMO.cube')

    def create_monomer_file(self) -> None:
        """Create gjf files of monomer from gjf file of dimer."""
        link0 = []
        # List for storing characters between coordinates and link1 when gem is used for the basis function.
        basis_gem = ['\n']
        link1 = []
        is_link0 = True
        is_link1 = False
        with open(f'{self._base_path}.gjf', 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                if re.search('[-0-9]+ [0-3]', line):
                    link0.append('0 1\n')
                    f, coordinates = self.extract_coordinates(f)
                    is_link0 = False
                elif r'%chk=' in line.lower():
                    continue
                elif '--link1--' == line.strip().lower():
                    is_link1 = True
                    link1.append(line)
                elif is_link0:
                    link0.append(line)
                elif is_link1:
                    link1.append(line)
                else:
                    basis_gem.append(line)

        if self.monomer1_atom_num == -1:
            mono_coordinates = coordinates[:len(coordinates)//2]
        else:
            mono_coordinates = coordinates[:self.monomer1_atom_num]

        for i in (1, 2):
            with open(f'{self._base_path}_m{i}.gjf', 'w', encoding='utf-8') as f:
                f.write(f'%Chk={self._base_path}_m{i}.chk\n')
                f.writelines(link0)

                if i == 2:
                    if self.monomer1_atom_num == -1:
                        mono_coordinates = coordinates[len(coordinates)//2:]
                    else:
                        mono_coordinates = coordinates[self.monomer1_atom_num:]

                f.writelines(mono_coordinates)
                f.writelines(basis_gem)

                f.write(f'{link1[0]}')
                f.write(f'%Chk={self._base_path}_m{i}.chk\n')
                f.writelines(link1[1:])

        print('monomer gjf file created')
        print(f' {self._base_path}_m1.gjf')
        print(f' {self._base_path}_m2.gjf')

    def custom_atomic_pair_transfer_analysis(
        self,
        analyze_orb1: int,
        analyze_orb2: int,
        output_apta: bool = False
    ) -> NDArray[np.float64]:
        """Calculate atomic pair transfer integrals.

        Parameters
        ----------
        analyze_orb1 : int
            Analyze orbital., default -1
        analyze_orb2 : int
            Analyze orbital., default -1
        output_apta : bool
            If it is True, output csv file of atomic pair transfer integrals., default False

        Returns
        -------
        numpy.array
            The array of atomic pair transfer analysis.
        """
        orb1 = self.mo1[analyze_orb1-1]
        orb2 = self.mo2[analyze_orb2-1]

        f11 = orb1 @ self.fock @ orb1
        f22 = orb2 @ self.fock @ orb2
        s12 = orb1 @ self.overlap @ orb2

        sum_f = f11 + f22
        pow_s12 = 1 - s12*s12
        atom_num = len(set(self.atom_index))
        a_transfer = np.zeros((atom_num, atom_num))
        for i in range(self.n_basis_d):
            for j in range(self.n_basis_d):
                orb_transfer = orb1[i] * orb2[j] * (self.fock[i][j] - 0.5 * sum_f * self.overlap[i][j]) / pow_s12
                a_transfer[self.atom_index[i]][self.atom_index[j]] += orb_transfer

        apta = self.print_apta(
            a_transfer,
            message=f'Atomic Pair Transfer Analysis (monomer 1 is {analyze_orb1}-th orbital, monomer 2 is {analyze_orb2}-th orbital)'
        )

        if output_apta:
            self.output_csv(f'{self._base_path}_apta_{analyze_orb1}_{analyze_orb2}.csv', apta)

        return apta

    def print_apta(
        self,
        a_transfer: NDArray[np.float64],
        message: str = 'Atomic Pair Transfer Analysis'
    ) -> NDArray[np.float64]:
        """Create list of apta and print it.

        Parameters
        ----------
        a_transfer : numpy.array
            Result of atomic pair transfer analysis.
        message : str
            Message to print., default 'Atomic Pair Transfer Analysis'

        Returns
        -------
        numpy.array
            The array of atomic pair transfer analysis.
        """
        n_atoms = self.atom_index[-1] + 1

        labels = [''] * n_atoms
        for i in range(self.n_basis_d):
            labels[self.atom_index[i]] = self.atom_symbol[i]

        col_sum = np.sum(a_transfer, axis=1)
        row_sum = np.sum(a_transfer, axis=0)
        total_sum = np.sum(a_transfer)

        print()
        print('-' * (len(message)+2))
        print(f' {message} ')
        print('-' * (len(message)+2))
        apta = []
        tmp_list = ['atom']
        for i in range(self.n_atoms1, n_atoms):
            tmp_list.append(f'{i+1}{labels[i]}')
        tmp_list.append('sum')
        apta.append(tmp_list)

        for i in range(self.n_atoms1):
            tmp_list = []
            tmp_list.append(f'{i+1}{labels[i]}')
            for j in range(self.n_atoms1, n_atoms):
                tmp_list.append(f'{a_transfer[i][j] * Tcal.EV:.3f}')
            tmp_list.append(f'{col_sum[i] * Tcal.EV:.3f}')
            apta.append(tmp_list)

        tmp_list = ['sum']
        for j in range(self.n_atoms1, n_atoms):
            tmp_list.append(f'{row_sum[j] * Tcal.EV:.3f}')
        tmp_list.append(f'{total_sum * Tcal.EV:.3f}')
        apta.append(tmp_list)

        self.print_matrix(apta)

        return apta

    def print_transfer_integrals(self) -> None:
        """Print transfer integrals of NLUMO, LUMO, HOMO and NHOMO."""
        print()
        print("--------------------")
        print(" Transfer Integrals ")
        print("--------------------")
        transfer = self.cal_transfer_integrals(
            self.mo1[self.n_elect1 + 1], self.overlap, self.fock, self.mo2[self.n_elect2 + 1]
        )
        print(f'NLUMO\t{transfer:>9.3f}\tmeV')

        transfer = self.cal_transfer_integrals(
            self.mo1[self.n_elect1], self.overlap, self.fock, self.mo2[self.n_elect2]
        )
        print(f'LUMO \t{transfer:>9.3f}\tmeV')

        transfer = self.cal_transfer_integrals(
            self.mo1[self.n_elect1 - 1], self.overlap, self.fock, self.mo2[self.n_elect2 - 1]
        )
        print(f'HOMO \t{transfer:>9.3f}\tmeV')

        transfer = self.cal_transfer_integrals(
            self.mo1[self.n_elect1 - 2], self.overlap, self.fock, self.mo2[self.n_elect2 - 2]
        )
        print(f'NHOMO\t{transfer:>9.3f}\tmeV')

    def print_transfer_integral_diff_levels(
        self,
        nlevel: int,
        output_ti_diff_levels: bool = False
    ) -> None:
        """Print transfer integrals between different orbitals.

        Parameters
        ----------
        nlevel : int
            The number of levels to print.
        output_ti_diff_levels : bool
            If it is True, output csv file of transfer integrals between different orbitals., default False
        """
        print()
        print('-----------------------------------------------')
        print(' Transfer Integrals between Different Orbitals ')
        print('-----------------------------------------------')

        if nlevel == 0:
            start1 = 0
            start2 = 0
            end1 = len(self.mo1)
            end2 = len(self.mo2)
        else:
            start1 = max(0, self.n_elect1 - nlevel)
            start2 = max(0, self.n_elect2 - nlevel)
            end1 = min(len(self.mo1), self.n_elect1 + nlevel)
            end2 = min(len(self.mo2), self.n_elect2 + nlevel)

        print(f'HOMO of monomer 1 is {self.n_elect1}-th orbital')
        print(f'HOMO of monomer 2 is {self.n_elect2}-th orbital')
        print(f'print all combinations between ({start1+1}, {end1}) and ({start2+1}, {end2}).')
        print()

        ti_diff_levels = []
        tmp_list = ['mono1/mono2']
        for i in range(start2, end2):
            if i+1 == self.n_elect2:
                tmp_list.append(f'HOMO({i+1})')
            elif i+1 == self.n_elect2+1:
                tmp_list.append(f'LUMO({i+1})')
            else:
                tmp_list.append(f'{i+1}')
        ti_diff_levels.append(tmp_list)

        for i in range(start1, end1):
            if i+1 == self.n_elect1:
                tmp_list = [f'HOMO({i+1})']
            elif i+1 == self.n_elect1+1:
                tmp_list = [f'LUMO({i+1})']
            else:
                tmp_list = [f'{i+1}']
            for j in range(start2, end2):
                transfer = self.cal_transfer_integrals(
                    self.mo1[i], self.overlap, self.fock, self.mo2[j]
                )
                tmp_list.append(f'{transfer:.3f}')
            ti_diff_levels.append(tmp_list)

        self.print_matrix(ti_diff_levels)

        if output_ti_diff_levels:
            self.output_csv(f'{self._base_path}_ti_diff_levels.csv', ti_diff_levels)

    def read_monomer1(
        self,
        is_matrix: bool = False,
        output_matrix: bool = False
    ) -> None:
        """Extract MO coefficients from log file of monomer.

        Parameters
        ----------
        is_matrix : bool
            If it is True, print MO coefficients., default False
        output_matrix : bool
            If it is True, Output MO coefficients., default False
        """
        print(f'reading {self._base_path}_m1{self._extension_log}')
        with open(f'{self._base_path}_m1{self._extension_log}', 'r', encoding='utf-8') as f:
            f = self.check_normal_termination(f)
            while True:
                line = f.readline()
                if not line:
                    break

                if self.n_basis1 is None:
                    self.n_basis1 = self.extract_num('[0-9]+ basis functions', line)

                if self.n_bsuse1 is None:
                    self.n_bsuse1 = self.extract_num(r'NBsUse=\s*[0-9]+', line, idx=-1)

                if self.n_elect1 is None:
                    self.n_elect1 = self.extract_num('[0-9]+ beta electrons', line)

                if self.n_atoms1 is None:
                    self.n_atoms1 = self.extract_num(r'NAtoms=\s*[0-9]+', line, idx=-1)

                if 'Alpha MO coefficients at cycle' in line:
                    self.mo1 = self.read_matrix(f, self.n_basis1, self.n_bsuse1)
                    break

        if is_matrix:
            print(' *** Alpha MO coefficients *** ')
            self.print_matrix(self.mo1)

        if output_matrix:
            self.output_csv(f'{self._base_path}_mo1.csv', self.mo1)

    def read_monomer2(
        self,
        is_matrix: bool = False,
        output_matrix: bool = False
    ) -> None:
        """Extract MO coefficients from log file of monomer.

        Parameters
        ----------
        is_matrix : bool, optional
            If it is True, print MO coefficients., default False
        output_matrix : bool, optional
            If it is True, Output MO coefficients., default False
        """
        print(f'reading {self._base_path}_m2{self._extension_log}')
        with open(f'{self._base_path}_m2{self._extension_log}', 'r', encoding='utf-8') as f:
            f = self.check_normal_termination(f)
            while True:
                line = f.readline()
                if not line:
                    break

                if self.n_basis2 is None:
                    self.n_basis2 = self.extract_num('[0-9]+ basis functions', line)

                if self.n_bsuse2 is None:
                    self.n_bsuse2 = self.extract_num(r'NBsUse=\s*[0-9]+', line, idx=-1)

                if self.n_elect2 is None:
                    self.n_elect2 = self.extract_num('[0-9]+ alpha electrons', line)

                if self.n_atoms2 is None:
                    self.n_atoms2 = self.extract_num(r'NAtoms=\s*[0-9]+', line, idx=-1)

                if 'Alpha MO coefficients at cycle' in line:
                    self.mo2 = self.read_matrix(f, self.n_basis2, self.n_bsuse2)
                    break

        if is_matrix:
            print(' *** Alpha MO coefficients *** ')
            self.print_matrix(self.mo2)

        if output_matrix:
            self.output_csv(f'{self._base_path}_mo2.csv', self.mo2)

    def read_dimer(
        self,
        is_matrix: bool = False,
        output_matrix: bool = False
    ) -> None:
        """Extract overlap and fock matrix from log file of dimer.

        Parameters
        ----------
        is_matrix : bool
            If it is True, print overlap and fock matrix., default False
        output_matrix : bool
            If it is True, Output overlap and fock matrix., default False
        """
        print(f'reading {self._base_path}{self._extension_log}')
        with open(f'{self._base_path}{self._extension_log}', 'r', encoding='utf-8') as f:
            f = self.check_normal_termination(f)
            while True:
                line = f.readline()
                if not line:
                    break

                if self.n_basis_d is None:
                    self.n_basis_d = self.extract_num('[0-9]+ basis functions', line)

                if self.n_elect_d is None:
                    self.n_elect_d = self.extract_num('[0-9]+ beta electrons', line)

                if '*** Overlap ***' in line:
                    self.overlap = self.read_symmetric_matrix(f, self.n_basis_d)

                if 'Fock matrix (alpha):' in line:
                    self.fock = self.read_symmetric_matrix(f, self.n_basis_d)

                if 'Gross orbital populations:' in line:
                    self._read_gross_orb_populations(f)
                    break

        zeros_matrix = np.zeros((self.n_bsuse1, self.n_basis2))
        self.mo1 = np.hstack([self.mo1.T, zeros_matrix])
        zeros_matrix = np.zeros((self.n_bsuse2, self.n_basis1))
        self.mo2 = np.hstack([zeros_matrix, self.mo2.T])

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

    def run_gaussian(
        self,
        gaussian_command: str,
        skip_monomer_num: List[int] = [0]
    ) -> int:
        """Execute gjf files using gaussian.

        Parameters
        ----------
        gaussian_command : str
            Command of gaussian.
        skip_monomer_num: list[int]
            If it is 1, skip 1st monomer calculation.
            If it is 2, skip 2nd monomer calculation.
            If it is 3, skip dimer calculation.

        Returns
        -------
        int
            Returncode of subprocess.run.
        """
        if 1 in skip_monomer_num:
            print('skip 1st monomer calculation')
        else:
            res = self._execute([gaussian_command, f'{self._base_path}_m1.gjf'], complete_message='1st monomer calculation completed')

            if res.returncode:
                return res.returncode

        if 2 in skip_monomer_num:
            print('skip 2nd monomer calculation')
        else:
            res = self._execute([gaussian_command, f'{self._base_path}_m2.gjf'], complete_message='2nd monomer calculation completed')

            if res.returncode:
                return res.returncode

        if 3 in skip_monomer_num:
            print('skip dimer calculation')
        else:
            res = self._execute([gaussian_command, f'{self._base_path}.gjf'], complete_message='dimer calculation completed')

            return res.returncode

        return 0

    def _create_dummy_cube_file(self) -> None:
        """Create dummy cube file."""
        min_x = 100
        max_x = -100
        min_y = 100
        max_y = -100
        min_z = 100
        max_z = -100
        cube_delta = 0.5
        bohr = 0.52917721092

        with open(f'{self._base_path}.gjf', 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if re.search('[-0-9]+ [0-3]', line):
                    f, coordinates = self.extract_coordinates(f)
                    break

        for line in coordinates:
            _, x, y, z = line.strip().split()
            x, y, z = map(float, [x, y, z])
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_z = min(min_z, z)
            max_z = max(max_z, z)

        min_x -= 5.0
        max_x += 5.0
        min_y -= 5.0
        max_y += 5.0
        min_z -= 5.0
        max_z += 5.0
        min_x = math.floor(min_x / cube_delta) * cube_delta
        min_y = math.floor(min_y / cube_delta) * cube_delta
        min_z = math.floor(min_z / cube_delta) * cube_delta

        num_x = math.ceil((max_x - min_x) / cube_delta)
        num_y = math.ceil((max_y - min_y) / cube_delta)
        num_z = math.ceil((max_z - min_z) / cube_delta)

        with open(f'{self._base_path}_dummy.cube', 'w') as f:
            f.write('\n\n')
            f.write(f'1, {min_x/bohr:.3f}, {min_y/bohr:.3f}, {min_z/bohr:.3f}\n')
            f.write(f'{num_x}, {cube_delta/bohr:.3f}, 0.0, 0.0\n')
            f.write(f'{num_y}, 0.0, {cube_delta/bohr:.3f}, 0.0\n')
            f.write(f'{num_z}, 0.0, 0.0, {cube_delta/bohr:.3f}\n')

    def _execute(
        self,
        command_list: List[str],
        complete_message: str = 'Calculation completed'
    ) -> subprocess.CompletedProcess:
        """Execute command

        Parameters
        ----------
        command_list : list
            A list of space-separated commands.
        complete_message : str
            The message when the calculation is completed., default 'Calculation completed'

        Returns
        -------
        CompletedProcess
            Return value of subprocess.run.
        """
        command = ' '.join(command_list)
        print(f'> {command}')

        res = subprocess.run(command_list, capture_output=True, text=True)

        # check error
        if res.returncode:
            print(f'Failed to execute {command}')
        else:
            print(complete_message)
            base_path = os.path.splitext(command_list[-1])[0]
            print(f' {base_path}{self._extension_log}')

        return res

    def _read_gross_orb_populations(self, reader: TextIO) -> None:
        """Extract atomic orbitals and atomic symbols.

        Parameters
        ----------
        reader : _io.TextIOWrapper
            Return value of open function.
        """
        reader.readline()

        self.atom_index = []
        self.atom_symbol = []
        self.atom_orbital = []

        idx_num = 1
        for i in range(1, self.n_basis_d+1):
            line = reader.readline().strip()
            # remove number of row
            line = line[re.match(f'{i}', line).span()[1]:].strip()
            # check atom
            change_atom = re.match(f'{idx_num} ', line)

            if change_atom:
                self.atom_index.append(idx_num-1)
                line = line[change_atom.span()[1]:].strip()
                elem = line[:2].strip()
                self.atom_symbol.append(elem)
                line = line[re.match(f'{elem}', line).span()[1]:].strip()
                idx_num += 1
            else:
                self.atom_index.append(self.atom_index[-1])
                self.atom_symbol.append(self.atom_symbol[-1])
            self.atom_orbital.append(line[:5].strip())


class PairAnalysis:
    """Analyze atomic pair transfer integrals."""
    def __init__(self, apta: ArrayLike) -> None:
        """Inits PairAnalysisClass.

        Parameters
        ----------
        apta : array_like
            List of atomic pair transfer integrals including labels.
        """
        self._labels = []
        self._a_transfer = []

        for row in apta[1:-1]:
            symbol = re.sub('[0-9]+', '', row[0])
            self._labels.append(symbol)
            self._a_transfer.append(row[1:-1])

        label = apta[0][1:-1]
        label = list(map(lambda x: re.sub('[0-9]+', '', x), label))

        self._labels.extend(label)
        self._a_transfer = np.array(self._a_transfer, dtype=np.float64)
        self.n_atoms1 = self._a_transfer.shape[0]
        self.n_atoms2 = self._a_transfer.shape[1]

    def print_largest_pairs(self) -> None:
        """Print largest pairs."""
        transfer = np.sum(self._a_transfer)
        a_transfer_flat = self._a_transfer.flatten()
        sorted_index = np.argsort(a_transfer_flat)
        print()
        print('---------------')
        print(' Largest Pairs ')
        print('---------------')
        print(f'rank\tpair{" " * 9}\ttransfer (meV)\tratio (%)')

        rank_list = np.arange(1, len(sorted_index)+1)
        if len(sorted_index) <= 20:
            print_index = sorted_index
            ranks = rank_list
        else:
            print_index = np.hstack([sorted_index[:10], sorted_index[-10:]])
            ranks = np.hstack([rank_list[:10], rank_list[-10:]])

        for i, a_i in enumerate(reversed(print_index)):
            row_i = a_i // self.n_atoms2
            col_i = a_i % self.n_atoms2
            pair = f'{row_i+1}{self._labels[row_i]}' + ' - ' + \
                f'{col_i+self.n_atoms2+1}{self._labels[col_i]}'
            ratio = np.divide(a_transfer_flat[a_i], transfer, out=np.array(0.0), where=(transfer!=0)) * 100
            print(f'{ranks[i]:<4}\t{pair:<13}\t{a_transfer_flat[a_i]:>14}\t{ratio:>9.1f}')

    def print_element_pairs(self) -> None:
        """Print element pairs."""
        element_pair = {
            'H-I': 0.0, 'C-C': 0.0, 'C-H': 0.0, 'C-S': 0.0, 'C-Se': 0.0,
            'C-I': 0.0, 'S-S': 0.0, 'Se-Se': 0.0, 'N-S': 0.0, 'I-S': 0.0,
            'I-I': 0.0,
        }
        keys = element_pair.keys()

        for i in range(self.n_atoms1):
            sym1 = self._labels[i]
            for j in range(self.n_atoms2):
                sym2 = self._labels[self.n_atoms1 + j]
                key = '-'.join(sorted([sym1, sym2]))
                if key in keys:
                    element_pair[key] += self._a_transfer[i][j]

        print()
        print('---------------')
        print(' Element Pairs ')
        print('---------------')
        for k, value in element_pair.items():
            if value != 0:
                print(f'{k:<5}\t{value:>9.3f}')


if __name__ == '__main__':
    main()
