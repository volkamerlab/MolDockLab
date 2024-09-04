# MolDockLab Workflow

MolDockLab is a **data-driven workflow** designed to identify the best balanced consensus **Structure-Based Virtual Screening (SBVS)** workflow for a target of interest. The workflow integrates various **docking tools**, **scoring functions**, and **consensus methods** to achieve optimal screening performance.

As a validation case study, the workflow was applied to the **EGFR target** (Epidermal Growth Factor Receptor) to compare different SBVS pipelines and assess their relative performance. Results can be found in `egfr_data/`.

MolDockLab was used in a real case study to find potential hits for an antibacterial target called **Energy Coupling Factor (ECF) Transporters**, in a collaboration work with Helmholtz-Institut für Pharmazeutische Forschung Saarland (HIPS). By screening the in-house library of around 6.6K compounds, it resulted in the identification of **two new antibacterial classes** for the target, which were validated experimentally.


<p align="center">
  <img src="moldocklab_fig.png" alt="MolDockLab Workflow Diagram">
</p>

---

## Repository Structure

- **`egfr_data/`**: Contains the EGFR-specific data used in the MolDockLab workflow, along with the results generated from running the pipeline.
- **`src/`**: The source code directory containing the core implementation of the MolDockLab workflow.
- **`test_data/`**: A small subset of test data (3 compounds for pipeline selection and 10 compounds for SBVS) used to test the workflow on a smaller scale.
- **`test_output/`**: Output generated from running the workflow on the test data.
- **`EGFR_pipeline.ipynb`**: A Jupyter notebook that reproduces the EGFR target results. Use this to follow along with the pipeline's analysis for EGFR.
- **`moldocklab.py`**: The main script that orchestrates the execution of the MolDockLab workflow.
- **`setup_py310.sh`**: Installation script to set up the Python 3.10 environment and dependencies required to run the workflow.

---

## Installation

Thanks to the [Installation guide of DockM8](https://github.com/DrugBud-Suite/DockM8/blob/main/DockM8_Installation_Guide.pdf), the environment can be installed using the provided `setup_py310.sh` installation script, after adding the needed packages and removing the unused ones.

## Installation

Thanks to the [Installation guide of DockM8](https://github.com/DrugBud-Suite/DockM8/blob/main/DockM8_Installation_Guide.pdf), the environment was adapted to meet the needs of **MolDockLab**. It can be installed using the provided `setup_py310.sh` after opening the terminal in the `MolDockLab` directory, through installation script:

```
bash ./setup_py310.sh
```
For more details or for running on windows, please refer to the the [installation guide of DockM8](https://github.com/DrugBud-Suite/DockM8/blob/main/DockM8_Installation_Guide.pdf)

---
## Usage

### Running the Workflow

To execute the workflow, run the `moldocklab.py` script with your desired parameters. Make sure your input data is ready. You can also use the provided test data in `test_data/` for smaller-scale testing. For reproducibility, the following command can be used with the test data:

```
python moldocklab.py --protein_path test_data/5UG9_noligand_protoss.pdb --ref_ligand_path test_data/ref_ligand.pdb --known_ligands_path test_data/test_three_cpds.sdf --sbvs_ligands_path test_data/test_library_10_cpds.sdf --out_dir test_output --true_value_col true_value --docking_programs gnina smina diffdock plants --pose_quality_checker True 
```
For step-by-step tutorial, the steps in `test_run.ipynb` can be followed.

### Command-Line Options


All arguments:
```
  -h, --help                    Show this help message and exit.

**Required Inputs:**
  --protein_path arg            Path to the protein file (required).
  --ref_ligand_path arg         Path to the reference ligand file (required).
  --known_ligands_path arg
                                         Path to the experimentally validated ligands library (required).
  --sbvs_ligands_path arg       Path to the large ligand library for SBVS (required).
  --true_value_col arg          Column name of the true activity values in the experimentally validated ligands library (required).

**Optional Arguments:**
  --activity_col arg           Column name for the activity class (default: "activity_class").
  --id_col arg                 Column name for the ligand ID (default: "ID").
  --protein_name arg           Protein name for documentation (optional).
  --n_cpus arg                 Number of CPUs to use for rescoring and ranking (default: 1).
  --out_dir arg                Directory to save results (default: "output").

**Docking Options:**
  --docking_programs args [DOCKING_PROGRAMS ...]
                                         Docking programs to use (default: gnina, smina, diffdock, plants, flexx).
                                         Example: --docking_programs gnina smina diffdock
  --n_poses arg                     Number of poses to generate per docking tool (default: 10).
  --exhaustiveness arg       Exhaustiveness for SMINA/GNINA docking tools (default: 8).
  --local_diffdock                      Use local DiffDock for predictions (default: False).

**Rescoring Options:**
  --rescoring args [RESCORING ...] Rescoring functions to use (default: cnnscore, ad4, linf9, rtmscore, vinardo, chemplp, rfscore_v1, rfscore_v3, vina_hydrophobic, vina_intra_hydrophobic).
                                         Example: --rescoring cnnscore vinardo
  --corr_threshold arg       Spearman correlation threshold to filter scores (default: 0.9).

**Ranking Methods:**
  --ranking_method args [RANKING_METHOD ...]
                                         Ranking methods to use (default: best_ECR, rank_by_rank, best_Zscore, weighted_ECR).
                                         Example: --ranking_method best_ECR weighted_ECR
  --runtime_reg arg             Regularization parameter for runtime cost in score optimization (default: 0.1).

**Pipeline Selection:**
  --corr_range arg               Allowed range of Spearman correlation for selecting a pipeline with the lowest runtime cost (default: 0.1).
  --ef_range arg                   Enrichment factor range for selecting the best pipeline (default: 0.5).

**Interaction Analysis:**
  --interacting_chains args [INTERACTING_CHAINS ...]
                                         Chains to include in protein-ligand interactions (default: X).
  --key_residues args [KEY_RESIDUES ...]
                                         Key residues for interaction analysis (e.g., "123A 124B"). If None, the top four frequent interacting residues from active compounds will be used (default: None).

**Diversity Selection:**
  --n_clusters arg               Number of clusters/compounds to select in diversity selection (default: 5).

**Quality Checking:**
  --pose_quality_checker                Enable pose quality checker using PoseBusters (default: False).
  --versatility_analysis                Enable versatility analysis to evaluate MolDockLab workflow performance (default: False).
```