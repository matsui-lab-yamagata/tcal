===========================================================
tcal Documentation
===========================================================

.. image:: https://img.shields.io/badge/python-3.9%20or%20newer-blue
   :target: https://www.python.org
   :alt: Python

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Requirements
============

* Python 3.9 or newer
* NumPy
* Gaussian 09 or 16

.. important::
   The path of the Gaussian must be set.

Installation
============

.. code-block:: bash

   pip install yu-tcal


Verify Installation
-------------------

After installation, you can verify by running:

.. code-block:: bash

   tcal --help

Options
=======

.. list-table::
   :header-rows: 1
   :widths: 10 15 75

   * - Short
     - Long
     - Explanation
   * - ``-a``
     - ``--apta``
     - Perform atomic pair transfer analysis.
   * - ``-c``
     - ``--cube``
     - Generate cube files.
   * - ``-g``
     - ``--g09``
     - Use Gaussian 09. (default is Gaussian 16)
   * - ``-h``
     - ``--help``
     - Show options description.
   * - ``-l``
     - ``--lumo``
     - Perform atomic pair transfer analysis of LUMO.
   * - ``-m``
     - ``--matrix``
     - Print MO coefficients, overlap matrix and Fock matrix.
   * - ``-o``
     - ``--output``
     - Output csv file on the result of apta.
   * - ``-r``
     - ``--read``
     - Read log files without executing Gaussian.
   * - ``-x``
     - ``--xyz``
     - Convert xyz file to gjf file.
   * -
     - ``--napta N1 N2``
     - Perform atomic pair transfer analysis between different levels. N1 is the number of level in the first monomer. N2 is the number of level in the second monomer.
   * -
     - ``--hetero N``
     - Calculate the transfer integral of heterodimer. N is the number of atoms in the first monomer.
   * -
     - ``--nlevel N``
     - Calculate transfer integrals between different levels. N is the number of levels from HOMO-LUMO. N=0 gives all levels.
   * -
     - ``--skip N...``
     - Skip specified Gaussian calculation. If N is 1, skip 1st monomer calculation. If N is 2, skip 2nd monomer calculation. If N is 3, skip dimer calculation.

How to Use
==========

1. Create gjf file
------------------

First of all, create a gaussian input file as follows:

**Example:** xxx.gjf

.. image:: https://raw.githubusercontent.com/matsui-lab-yamagata/tcal/main/img/gjf_file_example.png
   :alt: gjf file example

The xxx part is an arbitrary string.

Description of link commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**pop=full**
   Required to output coefficients of basis functions, overlap matrix, and Fock matrix.

**iop(3/33=4,5/33=3)**
   Required to output coefficients of basis functions, overlap matrix, and Fock matrix.

How to create a gjf using Mercury
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Open cif file in Mercury.
2. Display the dimer you want to calculate.

   .. image:: https://raw.githubusercontent.com/matsui-lab-yamagata/tcal/main/img/Anthracene_dimer.png
      :alt: Anthracene dimer

3. Save in mol file or mol2 file.
4. Open a mol file or mol2 file in GaussView and save it in gjf format.

2. Execute tcal
---------------

Suppose the directory structure is as follows:

.. code-block:: text

   yyy
   └── xxx.gjf

1. Open a terminal.
2. Go to the directory where the files is located.

   .. code-block:: bash

      cd yyy

3. Execute the following command.

   .. code-block:: bash

      tcal -a xxx.gjf

3. Visualization of molecular orbitals
--------------------------------------

1. Execute the following command.

   .. code-block:: bash

      tcal -cr xxx.gjf

2. Open xxx.fchk in GaussView.
3. [Results] → [Surfaces/Contours...]

   .. image:: https://raw.githubusercontent.com/matsui-lab-yamagata/tcal/main/img/visualize1.png
      :alt: Visualize 1

4. [Cube Actions] → [Load Cube]
5. Open xxx_m1_HOMO.cube and xxx_m2_HOMO.cube.

   .. image:: https://raw.githubusercontent.com/matsui-lab-yamagata/tcal/main/img/visualize2.png
      :alt: Visualize 2

6. Visualize by operating [Surface Actions] → [New Surface].

   .. image:: https://raw.githubusercontent.com/matsui-lab-yamagata/tcal/main/img/visualize3.png
      :alt: Visualize 3

   .. image:: https://raw.githubusercontent.com/matsui-lab-yamagata/tcal/main/img/visualize4.png
      :alt: Visualize 4

Interatomic Transfer Integral
==============================

For calculating the transfer integral between molecule A and molecule B, DFT calculations were performed for monomer A, monomer B, and the dimer AB. The monomer molecular orbitals :math:`\ket{A}` and :math:`\ket{B}` were obtained from the monomer calculations. Fock matrix F was calculated in the dimer system. Finally the intermolecular transfer integral :math:`t^{[1]}` was calculated by using the following equation:

.. math::

   t = \frac{\braket{A|F|B} - \frac{1}{2} (\epsilon_{A}+\epsilon_{B})\braket{A|B}}{1 - \braket{A|B}^2},

where :math:`\epsilon_A \equiv \braket{A|F|A}` and :math:`\epsilon_B \equiv \braket{B|F|B}`.

In addition to the intermolecular transfer integral in general use, we developed an interatomic transfer integral for further analysis :math:`^{[2]}`. By grouping the basis functions :math:`\ket{i}` and :math:`\ket{j}` for each atom, the molecular orbitals can be expressed as

.. math::

   \ket{A} = \sum^A_{\alpha} \sum^{\alpha}_i a_i \ket{i},

.. math::

   \ket{B} = \sum^B_{\beta} \sum^{\beta}_j b_j \ket{j},

where :math:`\alpha` and :math:`\beta` are the indices of atoms, :math:`i` and :math:`j` are indices of basis functions, and :math:`a_i` and :math:`b_j` are the coefficients of basis functions. Substituting this formula into aforementioned equation gives

.. math::

   t = \sum^A_{\alpha} \sum^B_{\beta} \sum^{\alpha}_i \sum^{\beta}_j a^*_i b_j \frac{\braket{i|F|j} - \frac{1}{2} (\epsilon_A + \epsilon_B) \braket{i|j}}{1 - \braket{A|B}^2}

Here we define the interatomic transfer integral :math:`u_{\alpha\beta}` as:

.. math::

   u_{\alpha \beta} \equiv \sum^{\alpha}_i \sum^{\beta}_j a^*_i b_j \frac{\braket{i|F|j} - \frac{1}{2} (\epsilon_A + \epsilon_B) \braket{i|j}}{1 - \braket{A|B}^2}

References
==========

[1] Veaceslav Coropceanu et al., Charge Transport in Organic Semiconductors, *Chem. Rev.* **2007**, *107*, 926-952.

[2] Koki Ozawa et al., Statistical analysis of interatomic transfer integrals for exploring high-mobility organic semiconductors, *Sci. Technol. Adv. Mater.* **2024**, *25*, 2354652.

Citation
========

When publishing works that benefited from tcal, please cite the following article:

   Koki Ozawa, Tomoharu Okada, Hiroyuki Matsui, Statistical analysis of interatomic transfer integrals for exploring high-mobility organic semiconductors, *Sci. Technol. Adv. Mater.*, **2024**, *25*, 2354652.

   `DOI: 10.1080/14686996.2024.2354652 <https://doi.org/10.1080/14686996.2024.2354652>`_

Example of using tcal
======================

1. `Satoru Inoue et al., Regioisomeric control of layered crystallinity in solution-processable organic semiconductors, Chem. Sci. 2020, 11, 12493-12505. <https://pubs.rsc.org/en/content/articlelanding/2020/SC/D0SC04461J>`_

2. `Toshiki Higashino et al., Architecting Layered Crystalline Organic Semiconductors Based on Unsymmetric π-Extended Thienoacenes, Chem. Mater. 2021, 33, 18, 7379-7385. <https://pubs.acs.org/doi/10.1021/acs.chemmater.1c01972>`_

3. `Koki Ozawa et al., Statistical analysis of interatomic transfer integrals for exploring high-mobility organic semiconductors, Sci. Technol. Adv. Mater. 2024, 25, 2354652. <https://doi.org/10.1080/14686996.2024.2354652>`_

API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Authors
=======

`Matsui Laboratory, Research Center for Organic Electronics (ROEL), Yamagata University <https://matsui-lab.yz.yamagata-u.ac.jp/index-e.html>`_

* Hiroyuki Matsui
* Koki Ozawa

Email: h-matsui[at]yz.yamagata-u.ac.jp (Please replace [at] with @)

Acknowledgements
================

This work was supported by JST, CREST, Grand Number JPMJCR18J2.

License
=======

This project is released under the MIT License.

For more details, see the `LICENSE file on GitHub <https://github.com/matsui-lab-yamagata/tcal/blob/main/LICENSE>`_.
