# Representative trajectory deposition plan

## Deposit contents

Create one public record containing two 50 ns production trajectories and their companion run inputs:

1. Generated candidate `Traj_912118`, replica 3.
2. Reference reassessment `Traj_13430`.

Upload each `.xtc` together with the matching `.tpr`, final `.gro`, production `.mdp`, and `index.ndx`. Use the archive filenames and SHA256 checksums in `representative_trajectory_files.csv`. The two XTC files total 4,675,157,648 bytes (about 4.68 GB decimal).

## Publication sequence

1. Verify that redistribution of the reference `Traj_13430` trajectory and topology is permitted by its upstream data license.
2. Upload the ten files listed in `representative_trajectory_files.csv` to Zenodo or Figshare.
3. Verify every uploaded file against the recorded SHA256 checksum.
4. Add the issued DOI and citation to this README and the repository citation metadata.
5. Link the DOI from the manuscript Data Availability statement.

## Boundary

Do not upload all 180 generated production trajectories to GitHub. GitHub contains the complete compact numerical outputs, representative structures/topologies, MDPs, Packmol inputs, construction workflow, and analysis scripts. The external record contains only the two designated representative trajectories.
