# UFF1 Database Documentation

## Overview

The UFF1 database is a comprehensive collection of geometric and electronic properties of CoII single-ion magnets. It was created as part of our research into the
prediction of magnetic anisotropy values using machine learning techniques.

## Data Structure

The UFF1 database is stored in a pickle (.pkl) file, which is a binary file format used
by Python for serializing and de-serializing Python object structures. The data
is structured as a pandas DataFrame, a two-dimensional, size-mutable,
heterogeneous tabular data structure.

## Data Columns

The DataFrame contains 86 columns, each representing a different parameter or
feature of the CoII single-ion magnets. These parameters are sectionalized into
geometric, DFT, and CASSCF features. For a full list of these parameters,
please refer to List S1 in our paper.

Here are the columns in the DataFrame:

- **Structure name** (dtype: object): Unique identifier for each structure.
- **Cartesian coordinates** (dtype: object): The Cartesian coordinates of the atoms in the structure.
- **Internal coordinates** (dtype: object): The internal coordinates of the atoms in the structure.
- **Coordination number** (dtype: int64): The coordination number of the CoII ion in the structure.
- **Spin deviation** (dtype: float64): The deviation of the spin from the ideal value.
- **SOMO-LUMO gap (a.u)** (dtype: float64): The energy gap between the singly occupied molecular orbital (SOMO) and the lowest unoccupied molecular orbital (LUMO).
- **SOMO-1-LUMO gap** (dtype: float64): The energy gap between the SOMO-1 and the LUMO.
- **SOMO-2-LUMO gap** (dtype: float64): The energy gap between the SOMO-2 and the LUMO.
- **Dipole moment X** (dtype: float64): The X component of the dipole moment of the structure.
- **Dipole moment Y** (dtype: float64): The Y component of the dipole moment of the structure.
- **Dipole moment Z** (dtype: float64): The Z component of the dipole moment of the structure.
- **Bond lengths 1** (dtype: float64): The first bond length in the structure.
- **Bond lengths 2** (dtype: float64): The second bond length in the structure.
- **Bond angles** (dtype: float64): The bond angles in the structure.
- **CAS root Mult** (dtype: float64): The root multiplicity of the complete active space (CAS) calculation.
- **CAS GS energy (Eh)** (dtype: float64): The ground state energy from the CAS calculation.
- **CAS transition energies** (dtype: object): The energies of the transitions from the CAS calculation.
- **CAS 1 el energy (Eh)** (dtype: float64): The one-electron energy from the CAS calculation.
- **CAS 2 el energy (Eh)** (dtype: float64): The two-electron energy from the CAS calculation.
- **CAS nucl. repulsion (Eh)** (dtype: float64): The nuclear repulsion energy from the CAS calculation.
- **Kinetic energy (Eh)** (dtype: float64): The kinetic energy from the CAS calculation.
- **Potential energy (Eh)** (dtype: float64): The potential energy from the CAS calculation.
- **Virial ratio (Eh)** (dtype: float64): The virial ratio from the CAS calculation.
- **Core energy (Eh)** (dtype: float64): The core energy from the CAS calculation.
- **SOC CAS lowest eigenv. (Eh)** (dtype: float64): The lowest eigenvalue from the spin-orbit coupling (SOC) CAS calculation.
- **SOC CAS stab. energy (cm-1)** (dtype: float64): The stabilization energy from the SOC CAS calculation.
- **CAS Kramers** (dtype: object): The Kramers degeneracy from the CAS calculation.
- **CAS Ms states** (dtype: object): The Ms states from the CAS calculation.
- **CAS 2PT D** (dtype: float64): The D value from the two-point CAS calculation.
- **CAS 2PT E/D** (dtype: float64): The E/D value from the two-point CAS calculation.
- **CAS Heff D** (dtype: float64): The D value from the effective Hamiltonian (Heff) CAS calculation.
- **CAS Heff E/D** (dtype: float64): The E/D value from the Heff CAS calculation.
- **CAS gx** (dtype: float64): The X component of the g tensor from the CAS calculation.
- **CAS gy** (dtype: float64): The Y component of the g tensor from the CAS calculation.
- **CAS gz** (dtype: float64): The Z component of the g tensor from the CAS calculation.
- **CAS giso** (dtype: float64): The isotropic g value from the CAS calculation.
- **CAS F0dd** (dtype: float64): The F0dd value from the CAS calculation.
- **CAS F2dd** (dtype: float64): The F2dd value from the CAS calculation.
- **CAS F4dd** (dtype: float64): The F4dd value from the CAS calculation.
- **CAS Racah A** (dtype: float64): The Racah A parameter from the CAS calculation.
- **CAS Racah B** (dtype: float64): The Racah B parameter from the CAS calculation.
- **CAS Racah C** (dtype: float64): The Racah C parameter from the CAS calculation.
- **CAS d-orb1 (eV)** (dtype: float64): The energy of the d orbital 1 from the CAS calculation in eV.
- **CAS d-orb2 (eV)** (dtype: float64): The energy of the d orbital 2 from the CAS calculation in eV.
- **CAS d-orb3 (eV)** (dtype: float64): The energy of the d orbital 3 from the CAS calculation in eV.
- **CAS d-orb4 (eV)** (dtype: float64): The energy of the d orbital 4 from the CAS calculation in eV.
- **CAS d-orb5 (eV)** (dtype: float64): The energy of the d orbital 5 from the CAS calculation in eV.
- **CAS d-orb1 (cm-1)** (dtype: float64): The energy of the d orbital 1 from the CAS calculation in cm-1.
- **CAS d-orb2 (cm-1)** (dtype: float64): The energy of the d orbital 2 from the CAS calculation in cm-1.
- **CAS d-orb3 (cm-1)** (dtype: float64): The energy of the d orbital 3 from the CAS calculation in cm-1.
- **CAS d-orb4 (cm-1)** (dtype: float64): The energy of the d orbital 4 from the CAS calculation in cm-1.
- **CAS d-orb5 (cm-1)** (dtype: float64): The energy of the d orbital 5 from the CAS calculation in cm-1.
- **CAS d-orb1 (xy)** (dtype: float64): The xy component of the d orbital 1 from the CAS calculation.
- **CAS d-orb2 (yz)** (dtype: float64): The yz component of the d orbital 2 from the CAS calculation.
- **CAS d-orb3 (z2)** (dtype: float64): The z2 component of the d orbital 3 from the CAS calculation.
- **CAS d-orb4 (xz)** (dtype: float64): The xz component of the d orbital 4 from the CAS calculation.
- **CAS d-orb5 (x2y2)** (dtype: float64): The x2y2 component of the d orbital 5 from the CAS calculation.
- **CAS SOC a** (dtype: float64): The a value from the spin-orbit coupling (SOC) CAS calculation.
- **CAS SOC b** (dtype: float64): The b value from the SOC CAS calculation.
- **CAS SOC Zeta** (dtype: float64): The Zeta value from the SOC CAS calculation.
- **Number of elements** (dtype: int64): The number of different elements in the structure.
- **Elements** (dtype: object): The types of elements in the structure.
- **Coordinates** (dtype: object): The coordinates of the atoms in the structure.
- **Z** (dtype: object): The atomic numbers of the atoms in the structure.
- **Coulumb matrix** (dtype: object): The Coulomb matrix of the structure.
- **Padded coulumb matrix** (dtype: object): The padded Coulomb matrix of the structure.

## Usage

To load the UFF1 database into a pandas DataFrame in Python, use the following code:

```python
import pandas as pd

df = pd.read_pickle('UFF1_DataFram.pkl')
```

You can then access the data in the DataFrame using standard pandas DataFrame operations. For example, to access the Cartesian coordinates of the first structure in the database, you would use:

```python
coords = df.loc[0, 'Cartesian coordinates']
```

Please refer to the pandas documentation for more information on working with DataFrames.
