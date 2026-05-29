import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem.rdMolTransforms import *
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from pysoftk.tools.utils_func import *
from pysoftk.tools.utils_ob import *
from pysoftk.tools.utils_rdkit import *

from openbabel import openbabel as ob
from openbabel import pybel as pb


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _normalize_force_field(force_field):
    ff = str(force_field)
    return "MMFF94" if ff == "MMFF" else ff


def _rdkit_optimize_conformers(mol, force_field, relax_iterations, num_confs, num_threads):
    work = Chem.Mol(mol)
    while work.GetNumConformers():
        work.RemoveConformer(0)

    ps = AllChem.ETKDGv3()
    ps.randomSeed = 0xF00D
    ps.useSmallRingTorsions = True
    ps.useMacrocycleTorsions = True
    if hasattr(ps, "numThreads"):
        ps.numThreads = int(max(1, num_threads))

    cids = list(AllChem.EmbedMultipleConfs(work, int(max(1, num_confs)), ps))
    if not cids:
        raise RuntimeError("RDKit EmbedMultipleConfs failed for polymer candidate generation")

    ff_name = _normalize_force_field(force_field)
    if ff_name == "UFF":
        results = AllChem.UFFOptimizeMoleculeConfs(
            work,
            numThreads=int(max(1, num_threads)),
            maxIters=int(max(1, relax_iterations)),
        )
        energies = [float(energy) for _, energy in results]
    else:
        variant = "MMFF94s" if ff_name == "MMFF94" else ff_name
        mp = AllChem.MMFFGetMoleculeProperties(work, mmffVariant=variant)
        if mp is None:
            raise RuntimeError(f"RDKit MMFF properties unavailable for force field {ff_name}")
        AllChem.MMFFOptimizeMoleculeConfs(
            work,
            numThreads=int(max(1, num_threads)),
            maxIters=int(max(1, relax_iterations)),
            mmffVariant=variant,
        )
        energies = []
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(work, mp, confId=int(cid))
            energies.append(float(ff.CalcEnergy()) if ff is not None else float("inf"))

    ranked = sorted(zip(cids, energies), key=lambda item: item[1])
    return work, ranked


def _openbabel_energy(mol, force_field):
    ff = ob.OBForceField.FindForceField(str(force_field))
    if ff is None:
        return float("inf")
    if not ff.Setup(mol.OBMol):
        return float("inf")
    return float(ff.Energy())


def _refine_pdb_candidate(task):
    pdb_block, force_field, relax_iterations, rot_steps = task
    ff_name = _normalize_force_field(force_field)
    mol_new = pb.readstring("pdb", pdb_block)
    last_mol = ff_ob_relaxation(mol_new, ff_name, int(relax_iterations))
    rot_mol = rotor_opt(last_mol, ff_name, int(rot_steps))
    energy = _openbabel_energy(rot_mol, ff_name)
    return rot_mol.write("pdb"), energy


class Lp:
    """
    A class for constructing linear polymers from individual molecular units (monomers)
    using RDKit functionalities.

    This class facilitates the creation of polymeric structures by taking a monomer
    RDKit molecule, replicating it a specified number of times, and then combining
    these copies into a linear chain. It handles the spatial arrangement and bonding
    of the monomer units, including the removal of placeholder atoms and subsequent
    geometry optimization using force fields.

    Examples
    ---------
    >>> from rdkit import Chem
    >>> from pysoftk.linear_polymers.linear_polymer import Lp
    >>> # Example: Create a simple monomer with a placeholder atom (e.g., [Pt])
    >>> monomer_smiles = "C([Pt])C"
    >>> monomer = Chem.MolFromSmiles(monomer_smiles)
    >>> # Initialize Lp with the monomer, placeholder atom, number of copies, and shift
    >>> linear_poly_builder = Lp(mol=monomer, atom="Pt", n_copies=3, shift=1.5)
    >>> # Generate the linear polymer with MMFF force field
    >>> polymer_mol = linear_poly_builder.linear_polymer(force_field="MMFF")
    >>> print(f"Polymer atoms: {polymer_mol.GetNumAtoms()}")
    >>> print(f"Polymer bonds: {polymer_mol.GetNumBonds()}")

    Note
    -----
    The RDKit and OpenBabel Python packages must be installed for this class to function
    correctly. Ensure that any placeholder atoms used in the monomer SMILES
    (e.g., "[Pt]") are consistent with the `atom` parameter.
    """

    __slots__ = ['mol', 'atom', 'n_copies', 'shift']

    def __init__(self, mol, atom, n_copies, shift):
        """
        Initializes the Lp class with the monomer molecule, placeholder atom,
        number of copies, and spatial shift.

        Parameters
        -----------
        mol : rdkit.Chem.rdchem.Mol
            The RDKit Mol object representing the monomer unit. This molecule
            should contain a designated placeholder atom that will be used for
            connecting the monomer copies.

        atom : str
            The atomic symbol of the placeholder atom in the `mol` that will be
            removed and replaced with a bond to form the polymer chain (e.g., "Pt", "Au").

        n_copies : int
            The total number of monomer units to be linked together to form the
            linear polymer.

        shift : float
            The initial X-axis translation distance (in Ångstroms) applied to
            each subsequent monomer copy before bond formation. This value
            influences the initial spacing between monomer units. If `None`,
            a default value of 1.25 Å is used.
        """

        self.mol = mol
        self.atom = atom
        self.n_copies = n_copies
        self.shift = float(1.25) if shift is None else float(shift)

    def max_dist_mol(self):
        """
        Calculates and returns the maximum interatomic distance within the
        initial RDKit monomer molecule (`self.mol`).

        This distance is crucial for determining the appropriate spacing
        between repeated monomer units to avoid steric clashes during the
        initial polymer assembly.

        Return
        -------
        np.float
            The largest distance (in Ångstroms) found between any two atoms
            in the monomer's conformational representation.
        """
        
        mol = self.mol
        bm = molDG.GetMoleculeBoundsMatrix(mol)
        return np.amax(bm)

    def x_shift(self):
        """
        Computes the refined X-axis translation value for positioning
        subsequent monomer units during polymer construction.

        This method adjusts the initial `shift` value based on the maximum
        distance between atoms in the monomer, aiming to create a more
        realistic and physically sound initial arrangement of the polymer chain.

        Return
        -------
        shift_final : float
            The calculated X-axis translation value (in Ångstroms) that will be
            applied to each repeating unit to properly space them in the linear polymer.
        """

        shift = self.shift
        shift_final = float(self.max_dist_mol()) - float(shift)

        return shift_final

    def copy_mol(self):
        """
        Generates a list of identical RDKit molecular objects, each representing
        a copy of the initial monomer.

        Before replication, the conformer of the original molecule is canonicalized
        to ensure consistent orientation for all copies.

        Parameters
        -----------
        mol : rdkit.Chem.rdchem.Mol
            The RDKit Mol object (self.mol) that will be replicated.

        Return
        --------
        fragments : list[rdkit.Chem.rdchem.Mol]
            A list containing `n_copies` identical RDKit Mol objects, each
            ready to be incorporated into the polymer chain.
        """

        mol = self.mol
        CanonicalizeConformer(mol.GetConformer())

        n_copies = self.n_copies

        fragments = [mol for _ in range(int(n_copies))]

        return fragments

    def polimerisation(self, fragments):
        """
        Recursively combines a list of RDKit monomer fragments into a single
        linear polymer RDKit molecule.

        Each subsequent fragment is combined with the growing polymer chain,
        with a calculated X-axis offset to maintain linearity and avoid overlap.
        The atoms in the resulting molecule are then renumbered for canonical ordering.

        Parameters
        -----------
        fragments : list[rdkit.Chem.rdchem.Mol]
            A list of RDKit Mol objects, each representing a monomer unit
            to be joined to form the polymer.

        Return
        --------
        outmol : rdkit.Chem.rdchem.Mol
            A single RDKit Mol object representing the combined linear polymer
            with all monomer units spatially arranged and canonically ordered.
        """

        x_offset = self.x_shift()

        outmol = fragments[0]
        for idx, values in enumerate(fragments[1:]):
            outmol = Chem.CombineMols(outmol, values,
                                      offset=Point3D(x_offset * (idx + 1), 0.0, 0.0))

        order = Chem.CanonicalRankAtoms(outmol, includeChirality=True)
        mol_ordered = Chem.RenumberAtoms(outmol, list(order))

        return outmol

    def bond_conn(self, outmol):
        """
        Identifies and processes the bonding information related to the placeholder
        atom within the combined super-monomer structure (`outmol`).

        This method extracts information about the atoms that are bonded to the
        placeholder atom, which is crucial for forming new bonds between monomers
        and subsequently removing the placeholder.

        Parameters
        -----------
        outmol : rdkit.Chem.rdchem.Mol
            The RDKit Mol object representing the partially formed polymer
            (combined super-monomer), containing the placeholder atoms.

        Return
        --------
        Tuple : (list[tuple], list[int])
            A tuple containing two lists:
            - The first list (`all_conn`) contains tuples of atom indices
              that represent the connections to be made between the monomers
              after the placeholder atoms are removed.
            - The second list (`erase_br`) contains the indices of the
              placeholder atoms that need to be removed from the molecule.
        """
        
        atom = self.atom

        bonds = atom_neigh(outmol, str(atom))
        conn_bonds = [b for a, b in bonds][1:-1]

        erase_br = [a for a, b in bonds]
        all_conn = list(zip(conn_bonds[::2], conn_bonds[1::2]))

        return all_conn, erase_br

    def proto_polymer(self):
        """
        Constructs the initial "proto-polymer" by combining monomer units,
        forming new bonds, and removing the placeholder atoms.

        This method orchestrates the steps of copying monomers, spatially
        arranging them, identifying and creating new bonds between them,
        and finally removing the temporary placeholder atoms used for connection.
        Hydrogen atoms are also added to the resulting molecule.

        Returns
        --------
        newMol_H : rdkit.Chem.rdchem.Mol
            A new RDKit Mol object representing the raw linear polymer structure,
            with placeholder atoms removed and new bonds formed, and explicit
            hydrogen atoms added with their coordinates.
        """
        
        atom = self.atom
        mol = self.mol
        n_copies = self.n_copies

        fragments = self.copy_mol()
        outmol = self.polimerisation(fragments)
        all_conn, erase_br = self.bond_conn(outmol)

        rwmol = Chem.RWMol(outmol)
        for ini, fin in all_conn:
            rwmol.AddBond(ini, fin, Chem.BondType.SINGLE)

        for i in sorted(erase_br[1:-1], key=None, reverse=True):
            rwmol.RemoveAtom(i)

        mol3 = rwmol.GetMol()
        Chem.SanitizeMol(mol3)

        mol4 = Chem.AddHs(mol3, addCoords=True)

        return mol4

    def linear_polymer(self, force_field="MMFF", relax_iterations=350, rot_steps=125, no_att=True):
        """
        Generates and optimizes the 3D structure of a linear polymer from the
        provided monomer unit.

        This is the main method for creating the final polymer. It first constructs
        a "proto-polymer" by combining monomers and forming initial bonds. Then,
        it can optionally remove any remaining placeholder atoms. Finally, it
        applies a specified force field to relax the geometry and performs
        rotor optimization to refine the polymer's conformation.

        Parameters
        -----------
        force_field : str, optional
            The name of the force field to use for geometry optimization.
            Accepted values are "MMFF", "UFF", or "MMFF94". "MMFF" will
            automatically default to "MMFF94". Defaults to "MMFF".
        relax_iterations : int, optional
            The maximum number of iterations to use during the force field
            relaxation step. A higher number generally leads to better
            minimization but takes longer. Defaults to 350.
        rot_steps : int, optional
            The number of steps for rotor optimization, which involves
            optimizing dihedral angles to find low-energy conformers.
            Defaults to 125.
        no_att : bool, optional
            If `True`, any remaining placeholder atoms (as defined by `self.atom`)
            will be removed from the polymer structure before force field
            optimization. If `False`, placeholder atoms will be retained.
            Defaults to `True`.

        Returns
        --------
        newMol_H : openbabel.pybel.Molecule
            An OpenBabel Molecule object representing the optimized 3D structure
            of the linear polymer.

        Raises
        ------
        ValueError
            If an invalid `force_field` is provided or if `relax_iterations`
            or `rot_steps` are not integers.
        """

        mol = self.proto_polymer()
        atom = self.atom

        if no_att:
            mol1 = remove_plcholder(mol, atom)
        else:
            mol1 = mol  # Use the original mol if mol1 is not provided

        # Validate force field:
        valid_force_fields = ("MMFF", "UFF", "MMFF94")
        if force_field not in valid_force_fields:
            raise ValueError(f"Invalid force field: {force_field}. Valid options are: {valid_force_fields}")

        # Automatically change ff if necessary:
        if force_field == "MMFF":
            force_field = "MMFF94"  # Change to default MMFF94 for MMFF

        # Validate and convert iterations and steps to integers:
        try:
            relax_iterations = int(relax_iterations)
            rot_steps = int(rot_steps)
        except ValueError:
            raise ValueError("relax_iterations and rot_steps must be integers.")

        # Prefer a multicore path when requested: generate multiple RDKit conformers,
        # then refine the best candidates with OpenBabel in parallel processes.
        use_multicore = _env_bool("GROMACS_PYSOFTK_MULTICORE", True)
        internal_threads = max(1, _env_int("GROMACS_PYSOFTK_INTERNAL_THREADS", 1))
        num_confs = max(1, _env_int("GROMACS_PYSOFTK_NUM_CONFS", max(2, internal_threads)))
        ob_workers = max(1, _env_int("GROMACS_PYSOFTK_OB_WORKERS", internal_threads))

        if use_multicore and (internal_threads > 1 or num_confs > 1 or ob_workers > 1):
            try:
                rdkit_mol, ranked = _rdkit_optimize_conformers(
                    mol1,
                    force_field=force_field,
                    relax_iterations=relax_iterations,
                    num_confs=num_confs,
                    num_threads=internal_threads,
                )

                candidate_count = max(1, min(len(ranked), ob_workers))
                candidate_tasks = []
                for cid, _energy in ranked[:candidate_count]:
                    pdb_block = Chem.MolToPDBBlock(rdkit_mol, confId=int(cid))
                    candidate_tasks.append((pdb_block, force_field, relax_iterations, rot_steps))

                best_pdb = None
                best_energy = float("inf")

                if len(candidate_tasks) == 1:
                    best_pdb, best_energy = _refine_pdb_candidate(candidate_tasks[0])
                else:
                    max_workers = min(len(candidate_tasks), ob_workers)
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        futs = [executor.submit(_refine_pdb_candidate, task) for task in candidate_tasks]
                        for fut in as_completed(futs):
                            pdb_block, energy = fut.result()
                            if float(energy) < float(best_energy):
                                best_energy = float(energy)
                                best_pdb = pdb_block

                if best_pdb:
                    return pb.readstring("pdb", best_pdb)
            except Exception:
                # Fall back to the original serial path for robustness.
                pass

        # Original serial path.
        last_rdkit = Chem.MolToPDBBlock(mol1)
        mol_new = pb.readstring('pdb', last_rdkit)
        last_mol = ff_ob_relaxation(mol_new, force_field, relax_iterations)
        rot_mol = rotor_opt(last_mol, force_field, rot_steps)
        return rot_mol

class Lpr:
    """
    Facilitates the generation of complex chemical structures by recursively
    substituting placeholders within a SMILES string, with a defined stopping
    condition and a final replacement for any remaining placeholders.

    This class is particularly useful for building polymeric or dendritic
    structures where a repeating unit needs to be expanded multiple times.
    It integrates with OpenBabel for 3D structure generation and force field
    optimization of the final molecule.
    """
    
    __slots__ = ['mol', 'replacements', 'max_repetitions', 'final_replacement']

    def __init__(self, mol, replacements, max_repetitions=10, final_replacement="*"):
        """
        Initializes the Lpr object for recursive SMILES string generation and
        subsequent 3D structure creation and optimization.

        Args:
            mol : str
                The base SMILES string which may contain one or more placeholders
                (e.g., "{R}") that will be recursively substituted.
            replacements : dict
                A dictionary where keys are the placeholders (e.g., "R") and values
                are the SMILES strings to substitute them with. These replacement
                strings can themselves contain placeholders for recursive expansion.
            max_repetitions : int, optional
                The maximum number of times the recursive substitution process will
                be attempted. The process stops earlier if no further substitutions
                occur. Defaults to 10.
            final_replacement : str, optional
                The SMILES string or SMARTS pattern to replace any remaining
                placeholders in the SMILES string after the recursive substitutions
                have completed. Defaults to "*", which represents an atom of any type.
        """
        
        self.mol = mol
        self.replacements = replacements
        self.max_repetitions = max_repetitions
        self.final_replacement = final_replacement

    def generate_recursive_smiles(self, force_field="MMFF", relax_iterations=350, rot_steps=125):
        """
        Executes the recursive SMILES generation process, followed by 3D structure
        generation and optimization using OpenBabel.

        The method iteratively substitutes placeholders in the SMILES string until
        either no more substitutions can be made or `max_repetitions` is reached.
        After generating the final SMILES string, it converts it to a 3D molecule,
        applies a force field for geometry relaxation, and performs rotor optimization.

        Parameters
        ----------
        force_field : str, optional
            The name of the force field to use for geometry optimization.
            Accepted values are "MMFF", "UFF", or "MMFF94". "MMFF" will
            automatically default to "MMFF94". Defaults to "MMFF".
        relax_iterations : int, optional
            The maximum number of iterations to use during the force field
            relaxation step. A higher number generally leads to better
            minimization but takes longer. Defaults to 350.
        rot_steps : int, optional
            The number of steps for rotor optimization, which involves
            optimizing dihedral angles to find low-energy conformers.
            Defaults to 125.

        Returns
        -------
        openbabel.pybel.Molecule
            An OpenBabel Molecule object representing the optimized 3D structure
            of the generated molecule.

        Raises
        ------
        KeyError
            If a placeholder specified in the SMILES string is not found in the
            `replacements` dictionary.
        ValueError
            If an invalid `force_field` is provided or if `relax_iterations`
            or `rot_steps` are not integers.
        """
        
        smiles = self.mol
        for _ in range(self.max_repetitions-1):
            try:
                new_smiles = smiles.format(**self.replacements)
            except KeyError as e:
                print(f"KeyError: {e} in replacements. Please check your replacements dictionary and input SMILES string.")
                return smiles # Return the most recent processed string, or handle as needed.

            if new_smiles == smiles:  # Stop if the SMILES string doesn't change
                break
            smiles = new_smiles

        # Replace the placeholder with the final_replacement
        smiles = smiles.replace("{R}", self.final_replacement)

        mol_new = pb.readstring('smiles', smiles)
        mol_new.make3D()

        # Validate force field:
        valid_force_fields = ("MMFF", "UFF", "MMFF94")
        if force_field not in valid_force_fields:
            raise ValueError(f"Invalid force field: {force_field}. Valid options are: {valid_force_fields}")

        # Automatically change ff if necessary:
        if force_field == "MMFF":
            force_field = "MMFF94"  # Change to default MMFF94 for MMFF

        # Validate and convert iterations and steps to integers:
        try:
            relax_iterations = int(relax_iterations)
            rot_steps = int(rot_steps)
        except ValueError:
            raise ValueError("relax_iterations and rot_steps must be integers.")

        # Relaxation and optimization:
        last_mol = ff_ob_relaxation(mol_new, force_field, relax_iterations)
        rot_mol = rotor_opt(last_mol, force_field, rot_steps)
        rel_mol = ff_ob_relaxation(rot_mol, force_field, relax_iterations)

        return rel_mol
