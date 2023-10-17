"""Tcal"""
import argparse
import csv
from datetime import datetime
import functools
import math
import os
import platform
import re
import subprocess
from time import time
import traceback

import numpy as np


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
        '-o', '--output', help='output csv file on the result of apta', action='store_true'
    )
    parser.add_argument(
        '-r', '--read', help='read log files without executing Gaussian', action='store_true'
    )
    parser.add_argument('-x', '--xyz', help='convert xyz to gjf', action='store_true')
    args = parser.parse_args()

    print('--------------------------------------')
    print(' tcal 1.0 (2023/10/10) by Matsui Lab. ')
    print('--------------------------------------')
    print(f'\nInput File Name: {args.file}')
    Tcal.print_timestamp()
    before = time()

    tcal = Tcal(args.file)

    # convert xyz to gjf
    if args.xyz:
        tcal.convert_xyz_to_gjf()

    if not args.read:
        tcal.create_monomer_file()

        if args.g09:
            gaussian_command = 'g09'
        else:
            gaussian_command = 'g16'
        tcal.run_gaussian(gaussian_command)

    try:
        tcal.read_monomer1(args.matrix)
        tcal.read_monomer2(args.matrix)
        tcal.read_dimer(args.matrix)

        if args.cube:
            tcal.create_cube_file()

        tcal.print_transfer_integrals()

        if args.apta:
            analyze_orbital = 'HOMO'
        if args.lumo:
            analyze_orbital = 'LUMO'

        if args.apta or args.lumo:
            if args.output:
                output_apta = True
            else:
                output_apta = False

            apta = tcal.atomic_pair_transfer_analysis(analyze_orbital, output_apta=output_apta)
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
    EV = 4.35974417e-18 / 1.60217653e-19 * 1000.0

    def __init__(self, file):
        """Inits TcalClass.

        Parameters
        ----------
        file : str
            A path of gjf file.
        """
        if platform.system() == 'Windows':
            self._extension_log = '.out'
        else:
            self._extension_log = '.log'

        self._base_path = os.path.splitext(file)[0]
        self.n_basis1 = None
        self.n_elect1 = None
        self.n_basis2 = None
        self.n_elect2 = None
        self.n_basis_d = None
        self.n_elect_d = None
        self.mo1 = None
        self.mo2 = None
        self.overlap = None
        self.fock = None
        self.atom_index = None
        self.atom_symbol = None
        self.atom_orbital = None

    @staticmethod
    def cal_transfer_integrals(bra, overlap, fock, ket):
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
        double
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
    def check_normal_termination(reader):
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
    def execute(command_list):
        """Execute command

        Parameters
        ----------
        command_list : list
            A list of space-separated commands.
        """
        command = ' '.join(command_list)
        print(f'> {command}')

        res = subprocess.run(command_list, capture_output=True, text=True)

        # check error
        if res.returncode:
            if platform.system() == 'Windows':
                print(f'Failed to execute {command}')
            else:
                print(res.stderr.strip())

    @staticmethod
    def extract_coordinates(reader):
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
    def extract_num(pattern, line):
        """Extract integer in strings.

        Parameters
        ----------
        pattern : str
            Strings using regular expression.
        line : str
            String of target.

        Returns
        -------
        int or None
            If there is integer, return it.
        """
        res = re.search(pattern, line)
        if res:
            return int(res.group().split()[0])
        return None

    @staticmethod
    def output_csv(file_name, array):
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
    def print_square_matrix(matrix):
        """Print square matrix.

        Parameters
        ----------
        matrix : array_like
        """
        for row in matrix:
            for cell in row[:-1]:
                print(cell, end='\t')
            print(row[-1])

    @staticmethod
    def print_timestamp():
        """Print timestamp."""
        month = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec',
        }
        dt_now = datetime.now()
        print(f"Timestamp: {dt_now.strftime('%a')} {month[dt_now.month]} {dt_now.strftime('%d %H:%M:%S %Y')}")

    @staticmethod
    def read_square_matrix(reader, n_basis):
        """Read square matrix.

        Parameters
        ----------
        reader : _io.TextIOWrapper
            Return value of open function.
        n_basis : int
            The number of column or row.

        Returns
        -------
        numpy.array
            Read square matrix like MO coefficients.
        """
        mat = np.zeros((n_basis, n_basis), dtype=np.float64)
        for i in range(math.ceil(n_basis/5)):
            if 'Alpha density matrix' in reader.readline():
                break
            for j in range(n_basis):
                for k, val in enumerate(reader.readline().split()[1:]):
                    mat[j][i*5+k] = float(val.strip().replace('D', 'E'))

        return mat

    @staticmethod
    def read_symmetric_matrix(reader, n_basis):
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

    def atomic_pair_transfer_analysis(self, analyze_orbital='HOMO', output_apta=False):
        """Calculate atomic pair transfer integrals.

        Parameters
        ----------
        analyze_orbital : str, optional
            Analyze orbital., default 'HOMO'
        output_apta : bool, optional
            If it is True, output csv file of atomic pair transfer integrals., default False
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
            Tcal.output_csv(f'{self._base_path}_apta_{analyze_orbital}.csv', apta)

        return apta

    def convert_xyz_to_gjf(self, function='B3LYP/6-31G(d,p)', nprocshared=4, mem=16, unit='GB'):
        """Convert xyz file to gjf file.

        Parameters
        ----------
        function : str, optional
            _description_, default 'b3lyp/6-31g(d,p)'
        nprocshared : int, optional
            The number of nprocshared., default 4
        mem : int, optional
            The number of memory., default 16
        unit : str, optional
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

    def create_cube_file(self):
        """Create cube file."""
        self.execute(['formchk', f'{self._base_path}.chk', f'{self._base_path}.fchk'])
        self._create_dummy_cube_file()
        self.execute(['formchk', f'{self._base_path}_m1.chk', f'{self._base_path}_m1.fchk'])
        self.execute(['cubegen', '0', f'mo={self.n_elect1-1}', f'{self._base_path}_m1.fchk', f'{self._base_path}_m1_NHOMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        self.execute(['cubegen', '0', f'mo={self.n_elect1}', f'{self._base_path}_m1.fchk', f'{self._base_path}_m1_HOMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        self.execute(['cubegen', '0', f'mo={self.n_elect1+1}', f'{self._base_path}_m1.fchk', f'{self._base_path}_m1_LUMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        self.execute(['cubegen', '0', f'mo={self.n_elect1+2}', f'{self._base_path}_m1.fchk', f'{self._base_path}_m1_NLUMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        print('cube file of the 1st monomer created')
        print(f' {self._base_path}_m1_NHOMO.cube')
        print(f' {self._base_path}_m1_HOMO.cube')
        print(f' {self._base_path}_m1_LUMO.cube')
        print(f' {self._base_path}_m1_NLUMO.cube')

        self.execute(['formchk', f'{self._base_path}_m2.chk', f'{self._base_path}_m2.fchk'])
        self.execute(['cubegen', '0', f'mo={self.n_elect2-1}', f'{self._base_path}_m2.fchk', f'{self._base_path}_m2_NHOMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        self.execute(['cubegen', '0', f'mo={self.n_elect2}', f'{self._base_path}_m2.fchk', f'{self._base_path}_m2_HOMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        self.execute(['cubegen', '0', f'mo={self.n_elect2+1}', f'{self._base_path}_m2.fchk', f'{self._base_path}_m2_LUMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        self.execute(['cubegen', '0', f'mo={self.n_elect2+2}', f'{self._base_path}_m2.fchk', f'{self._base_path}_m2_NLUMO.cube', '-1', 'h', f'{self._base_path}_dummy.cube'])
        print('cube file of the 2nd monomer created')
        print(f' {self._base_path}_m2_NHOMO.cube')
        print(f' {self._base_path}_m2_HOMO.cube')
        print(f' {self._base_path}_m2_LUMO.cube')
        print(f' {self._base_path}_m2_NLUMO.cube')

    def _create_dummy_cube_file(self):
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
                    f, coordinates = Tcal.extract_coordinates(f)
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

    def create_monomer_file(self):
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
                    f, coordinates = Tcal.extract_coordinates(f)
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

        mono_coordinates = coordinates[:len(coordinates)//2]

        for i in (1, 2):
            with open(f'{self._base_path}_m{i}.gjf', 'w', encoding='utf-8') as f:
                f.write(f'%Chk={self._base_path}_m{i}.chk\n')
                f.writelines(link0)

                if i == 2:
                    mono_coordinates = coordinates[len(coordinates)//2:]

                f.writelines(mono_coordinates)
                f.writelines(basis_gem)

                f.write(f'{link1[0]}')
                f.write(f'%Chk={self._base_path}_m{i}.chk\n')
                f.writelines(link1[1:])

        print('monomer gjf file created')
        print(f' {self._base_path}_m1.gjf')
        print(f' {self._base_path}_m2.gjf')

    def custom_atomic_pair_transfer_analysis(self, analyze_orb1=-1, analyze_orb2=-1, output_apta=False):
        """Calculate atomic pair transfer integrals.

        Parameters
        ----------
        analyze_orb1 : int, optional
            Analyze orbital., default -1
        analyze_orb2 : int, optional
            Analyze orbital., default -1
        output_apta : bool, optional
            If it is True, output csv file of atomic pair transfer integrals., default False
        """
        orb1 = self.mo1[self.n_elect1 + analyze_orb1]
        orb2 = self.mo2[self.n_elect2 + analyze_orb2]

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
            Tcal.output_csv(f'{self._base_path}_apta_{analyze_orb1}_{analyze_orb2}.csv', apta)

        return apta

    def print_apta(self, a_transfer):
        """Create list of apta and print it.

        Parameters
        ----------
        n_atoms : int
            The number of atoms.
        labels : list
            The list of index and atom.
        a_transfer : numpy.array
            Result of atomic pair transfer analysis.

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
        print('-------------------------------')
        print(' Atomic Pair Transfer Analysis ')
        print('-------------------------------')
        apta = []
        tmp_list = ['atom']
        for i in range(n_atoms//2, n_atoms):
            tmp_list.append(f'{i+1}{labels[i]}')
        tmp_list.append('sum')
        apta.append(tmp_list)

        for i in range(n_atoms//2):
            tmp_list = []
            tmp_list.append(f'{i+1}{labels[i]}')
            for j in range(n_atoms//2, n_atoms):
                tmp_list.append(f'{a_transfer[i][j] * Tcal.EV:.3f}')
            tmp_list.append(f'{col_sum[i] * Tcal.EV:.3f}')
            apta.append(tmp_list)

        tmp_list = ['sum']
        for j in range(n_atoms//2, n_atoms):
            tmp_list.append(f'{row_sum[j] * Tcal.EV:.3f}')
        tmp_list.append(f'{total_sum * Tcal.EV:.3f}')
        apta.append(tmp_list)

        Tcal.print_square_matrix(apta)

        return apta

    def print_transfer_integrals(self):
        """Print transfer integrals of NLUMO, LUMO, HOMO and NHOMO."""
        print()
        print("--------------------")
        print(" Transfer Integrals ")
        print("--------------------")
        transfer = Tcal.cal_transfer_integrals(
            self.mo1[self.n_elect1 + 1], self.overlap, self.fock, self.mo2[self.n_elect2 + 1]
        )
        print(f'NLUMO\t{transfer:.3f}\tmeV')

        transfer = Tcal.cal_transfer_integrals(
            self.mo1[self.n_elect1], self.overlap, self.fock, self.mo2[self.n_elect2]
        )
        print(f'LUMO\t{transfer:.3f}\tmeV')

        transfer = Tcal.cal_transfer_integrals(
            self.mo1[self.n_elect1 - 1], self.overlap, self.fock, self.mo2[self.n_elect2 - 1]
        )
        print(f'HOMO\t{transfer:.3f}\tmeV')

        transfer = Tcal.cal_transfer_integrals(
            self.mo1[self.n_elect1 - 2], self.overlap, self.fock, self.mo2[self.n_elect2 - 2]
        )
        print(f'NHOMO\t{transfer:.3f}\tmeV')

    def read_monomer1(self, is_matrix=False, output_matrix=False):
        """Extract MO coefficients from log file of monomer.

        Parameters
        ----------
        is_matrix : bool, optional
            If it is True, print MO coefficients., default False
        output_matrix : bool, optional
            If it is True, Output MO coefficients., default False
        """
        print(f'reading {self._base_path}_m1{self._extension_log}')
        with open(f'{self._base_path}_m1{self._extension_log}', 'r', encoding='utf-8') as f:
            f = Tcal.check_normal_termination(f)
            while True:
                line = f.readline()
                if not line:
                    break

                if self.n_basis1 is None:
                    self.n_basis1 = Tcal.extract_num('[0-9]+ basis functions', line)

                if self.n_elect1 is None:
                    self.n_elect1 = Tcal.extract_num('[0-9]+ beta electrons', line)

                if 'Alpha MO coefficients at cycle' in line:
                    self.mo1 = Tcal.read_square_matrix(f, self.n_basis1)
                    break

        if is_matrix:
            print(' *** Alpha MO coefficients *** ')
            Tcal.print_square_matrix(self.mo1)

        if output_matrix:
            Tcal.output_csv(f'{self._base_path}_mo1.csv', self.mo1)

    def read_monomer2(self, is_matrix=False, output_matrix=False):
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
            f = Tcal.check_normal_termination(f)
            while True:
                line = f.readline()
                if not line:
                    break

                if self.n_basis2 is None:
                    self.n_basis2 = Tcal.extract_num('[0-9]+ basis functions', line)

                if self.n_elect2 is None:
                    self.n_elect2 = Tcal.extract_num('[0-9]+ alpha electrons', line)

                if 'Alpha MO coefficients at cycle' in line:
                    self.mo2 = Tcal.read_square_matrix(f, self.n_basis2)
                    break

        if is_matrix:
            print(' *** Alpha MO coefficients *** ')
            Tcal.print_square_matrix(self.mo2)

        if output_matrix:
            Tcal.output_csv(f'{self._base_path}_mo2.csv', self.mo2)

    def read_dimer(self, is_matrix=False, output_matrix=False):
        """Extract overlap and fock matrix from log file of dimer.

        Parameters
        ----------
        is_matrix : bool, optional
            If it is True, print overlap and fock matrix., default False
        output_matrix : bool, optional
            If it is True, Output overlap and fock matrix., default False
        """
        print(f'reading {self._base_path}{self._extension_log}')
        with open(f'{self._base_path}{self._extension_log}', 'r', encoding='utf-8') as f:
            f = Tcal.check_normal_termination(f)
            while True:
                line = f.readline()
                if not line:
                    break

                if self.n_basis_d is None:
                    self.n_basis_d = Tcal.extract_num('[0-9]+ basis functions', line)

                if self.n_elect_d is None:
                    self.n_elect_d = Tcal.extract_num('[0-9]+ beta electrons', line)

                if '*** Overlap ***' in line:
                    self.overlap = Tcal.read_symmetric_matrix(f, self.n_basis_d)

                if 'Fock matrix (alpha):' in line:
                    self.fock = Tcal.read_symmetric_matrix(f, self.n_basis_d)

                if 'Gross orbital populations:' in line:
                    self._read_gross_orb_populations(f)
                    break

        zeros_matrix = np.zeros((self.n_basis1, self.n_basis1))
        self.mo1 = np.hstack([self.mo1.T, zeros_matrix])

        self.mo2 = np.hstack([zeros_matrix, self.mo2.T])

        if is_matrix:
            print()
            print(' *** Overlap *** ')
            Tcal.print_square_matrix(self.overlap)
            print()
            print(' *** Fock matrix (alpha) *** ')
            Tcal.print_square_matrix(self.fock)

        if output_matrix:
            Tcal.output_csv(f'{self._base_path}_overlap.csv', self.overlap)
            Tcal.output_csv(f'{self._base_path}_fock.csv', self.fock)

    def run_gaussian(self, gaussian_command):
        """Execute gjf files using gaussian.

        Parameters
        ----------
        gaussian_command : str
            Command of gaussian.
        """
        Tcal.execute([gaussian_command, f'{self._base_path}_m1.gjf'])
        print('1st monomer calculation completed')
        print(f' {self._base_path}_m1{self._extension_log}')

        Tcal.execute([gaussian_command, f'{self._base_path}_m2.gjf'])
        print('2nd monomer calculation completed')
        print(f' {self._base_path}_m2{self._extension_log}')

        Tcal.execute([gaussian_command, f'{self._base_path}.gjf'])
        print('dimer calculation completed')
        print(f' {self._base_path}{self._extension_log}')

    def _read_gross_orb_populations(self, reader):
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
    def __init__(self, apta):
        """Inits PairAnalysisClass.

        Parameters
        ----------
        apta : list
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

        self._mono_atom_num = len(label)

    def print_largest_pairs(self):
        """Print largest pairs."""
        transfer = np.sum(self._a_transfer)
        a_transfer_flat = self._a_transfer.flatten()
        sorted_index = np.argsort(a_transfer_flat)
        print()
        print('---------------')
        print(' Largest Pairs ')
        print('---------------')
        print('rank\tpair\ttransfer (meV)\tratio (%)')

        rank_list = np.arange(1, len(sorted_index)+1)
        if len(sorted_index) <= 20:
            print_index = sorted_index
            ranks = rank_list
        else:
            print_index = np.hstack([sorted_index[:10], sorted_index[-10:]])
            ranks = np.hstack([rank_list[:10], rank_list[-10:]])

        for i, a_i in enumerate(reversed(print_index)):
            row_i = a_i // self._mono_atom_num
            col_i = a_i % self._mono_atom_num
            pair = f'{row_i+1}{self._labels[row_i]}' + ' - ' + \
                f'{col_i+self._mono_atom_num+1}{self._labels[col_i]}'
            ratio = a_transfer_flat[a_i] / transfer * 100
            print(f'{ranks[i]}\t{pair}\t{a_transfer_flat[a_i]}\t{ratio:.1f}')

    def print_element_pairs(self):
        """Print element pairs."""
        element_pair = {
            'H-I': 0.0, 'C-C': 0.0, 'C-H': 0.0, 'C-S': 0.0, 'C-Se': 0.0,
            'C-I': 0.0, 'S-S': 0.0, 'Se-Se': 0.0, 'N-S': 0.0, 'I-S': 0.0,
            'I-I': 0.0,
        }
        keys = element_pair.keys()

        for i in range(len(self._labels)//2):
            sym1 = self._labels[i]
            for j in range(len(self._labels)//2):
                sym2 = self._labels[self._mono_atom_num + j]
                key = '-'.join(sorted([sym1, sym2]))
                if key in keys:
                    element_pair[key] += self._a_transfer[i][j]

        print()
        print('---------------')
        print(' Element Pairs ')
        print('---------------')
        for k, value in element_pair.items():
            if value != 0:
                print(f'{k}\t{value:.3f}')


if __name__ == '__main__':
    main()
