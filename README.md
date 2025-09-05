# palaeoPSM

Python package to read PSM data from Fragpipe Open search runs into amino acid position specific identifications and PTMs.

It reads the `psm.tsv` file or `interact.pep.xml` (pepXML format) with each PSM.
If mzML files are provided, they are parsed to count spectra and find the experimental RT from each PSM spectrum.

It calculates the global protein position of each amino acid from each PSM peptide and uses the PTM location information from MSFragger to calculate the position of ech PTM in the protein sequence.
This creates a dataframe with standard column names that then is pivoted to long form so each amino acid position is in one row.

Basic use:
```python
# Set up the fragpipe run object
fp_run = '../fp_results/milk_st_db1'
fp_run_milk = FragPipeRun(
    path=fp_run,
    run_id='milk_test_db1',
    format='tsv',
    decoy_tag='rev_',
    # We can provide a table with the # of spectra per file, instead of the mzML
    n_scans_path='../test_data/n_scans.csv'
)
# Read PSM data
psm_data = fp_run_milk.read(
    n_procs=1,
    save_path=None,
    remove_contams=False,
    remove_decoy=True,
    fdr_threshold=0.01)
# Pivot to long format
psm_long = psm_to_long(psm_data)
```


TODO:
- Add support for Metamorpheus, Mascot, MaxQuant and pFind
- Add position specific weights to PTM counts when position is ambiguous