import sys

import pandas as pd
import numpy as np
from pyteomics import pepxml
import os
from pyteomics.mass import std_aa_mass
# from pyteomics import mzml
# from pyteomics import mzid
# from pyteomics import parser
# from pyteomics.mass import unimod
# from lxml import etree
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib_venn import venn3
# from Bio import SeqIO
# import re


def load_psm_table(file, column_data, sep):
    col_rename = {f[0]: f[1] for f in column_data}
    usecols = [f[0] for f in column_data]
    colnames = [f[1] for f in column_data]
    table = (
        pd.read_csv(file, sep=sep, usecols=usecols)
        .rename(columns=col_rename)
    )
    if table.empty:
        table = pd.DataFrame(
            [[f[2] for f in column_data]],
            columns=colnames
        )
    return table


def get_prot_info(pept_seq, proteins, fdb, field='protein'):
    """
    Return the start and end of a peptide sequence within a protein, using a list of identified proteins
    It needs a dictionary of protein ID keys to Bio.SeqRecord objects
    :param pept_seq: Peptide sequence
    :param proteins: List of strings or dicts. Strings are protein IDs, if dicts they contain protein ID in field
    :param fdb: Dictionary of Bio.SeqRecords as returned by SeqIO.to_dict
    :param field: f proteins is list of dicts, name of field that contains the ID
    :return: protein id, start and end (0-based)
    """

    # See if peptide matches to any LACB
    prot_id = proteins[0][field] if field != '' else proteins[0]
    # for p in proteins:
    #     p = p[field] if field != '' else p
    #     if p in bovin_lacb:
    #         prot_id = p
    #         break
    #     if p in other_lacb:
    #         prot_id = p
    # Get protein sequence using prot_id and get peptide start and end
    seq_rec = fdb.get(prot_id, None)
    if seq_rec is None:
        return '', -1, -1
    prot_seq = seq_rec.seq
    pstart = prot_seq.find(pept_seq)
    pend = pstart + len(pept_seq) - 1
    return prot_id, pstart, pend


def get_rt(scan_ns, mzml_spectra):
    """
    Given a list of scan numbers, it retrieves the RT from mzML pyteomics object
    :param scan_ns: List of scan numbers
    :param mzml_spectra: mzML object
    :return: DataFrame with columns scan index and RTsec
    """
    scan_idx = scan_ns - 1
    sel_mzml = mzml_spectra[scan_idx.tolist()]
    rt_list = []
    for i in range(len(scan_idx)):
        rt = sel_mzml[i]['scanList']['scan'][0]['scan start time'] * 60
        rt_list.append([scan_idx[i], rt])
    return pd.DataFrame(rt_list, columns=['Scan_idx', 'RTsec'])


class PepXMLdataExtractor:
    """
    Extracts data from pyteomics pepxml object. The object is callable and can be passed to
    the pepxml map method to extractdata from PSMs.
    It works with Mascot and FragPipe pepxml flavours
    """
    def __init__(self, flavour, decoy_tag='rev_'):
        self.empty_pept_seq = ''
        self.empty_prot_id = ''
        self.empty_pstart = -1
        self.empty_pend = -1
        self.empty_isdecoy = None
        self.empty_delta_mass = np.nan
        self.empty_calc_mass = np.nan
        self.empty_qval_key = np.nan
        self.empty_scan_no = -1
        self.empty_rt = np.nan
        self.decoy_tag = decoy_tag
        if flavour == 'mascot':
            self.qval_key_f = self.mascot_expectation
            self.scan_no_f = self.mascot_scan_no
            self.rt_f = self.mascot_rt
            self.qval_key = 'expectation'
        elif flavour == 'fragpipe':
            self.qval_key_f = self.fragpipe_probability
            self.scan_no_f = self.fragpipe_scan_no
            self.rt_f = self.fragpipe_rt
            self.qval_key = 'probability'

    def empty_sample(self):
        psm_data = [self.empty_scan_no, self.empty_rt, self.empty_pept_seq,
                    self.empty_prot_id, self.empty_pstart, self.empty_pend,
                    self.empty_isdecoy,  self.empty_calc_mass, self.empty_delta_mass,
                    self.empty_qval_key]
        return psm_data

    @staticmethod
    def mascot_expectation(psm):
        return psm['search_hit'][0]['search_score']['expect']

    @staticmethod
    def mascot_scan_no(psm):
        return int(psm['spectrum'].split(' ')[2].split('=')[1])

    @staticmethod
    def mascot_rt(psm):
        return float(psm['search_specification'].split(' ')[2].split('(')[1].rstrip(')'))

    @staticmethod
    def fragpipe_probability(psm):
        return psm['search_hit'][0]['analysis_result'][0]['peptideprophet_result']['probability']

    @staticmethod
    def fragpipe_scan_no(psm):
        return psm['start_scan']

    @staticmethod
    def fragpipe_rt(psm):
        return psm['retention_time_sec']

    def get_pepxml_data(self, psm, fdb):
        rt = self.rt_f(psm)
        scan_no = self.scan_no_f(psm)
        if 'search_hit' not in psm:
            psm_data = [scan_no, rt, self.empty_pept_seq,
                        self.empty_prot_id, self.empty_pstart, self.empty_pend,
                        self.empty_isdecoy,  self.empty_calc_mass, self.empty_delta_mass,
                        self.empty_qval_key]
        else:
            pept_seq = psm['search_hit'][0]['peptide']
            prot_id, pstart, pend = get_prot_info(
                pept_seq, psm['search_hit'][0]['proteins'],
                fdb, field='protein')
            proteins = psm['search_hit'][0]['proteins']
            if len(proteins) > 1:
                other_prot_ids = [p['protein'] for p in proteins[1:]]
            else:
                other_prot_ids = []
            isdecoy = pepxml.is_decoy(psm, prefix=self.decoy_tag)
            delta_mass = psm['search_hit'][0]['massdiff']
            calc_mass = psm['search_hit'][0]['calc_neutral_pep_mass']
            qval_key = self.qval_key_f(psm)

            # extract Variable modification information
            var_mods = psm['search_hit'][0]['modifications']
            var_mods_pos = [0] * len(pept_seq)
            for m in var_mods:
                mod_aa = pept_seq[m['position']-1]
                mass = m['mass'] - np.round(std_aa_mass[mod_aa])
                var_mods_pos[m['position']-1] = mass

            # Extract delta_mass mods information
            ptm_result = psm['ptm_result']
            delta_mass_mods_pos = [0] * len(pept_seq)
            # Loop through positions and insert mass into delta_mass_mods_pos
            for p in ptm_result['localization'].split('_'):
                if p == '':
                    break
                p = int(p)


            psm_data = {
                'Scan_No': scan_no,
                'RTsec': rt,
                'Seq': pept_seq,
                'prot_id': prot_id,
                'other_prot_ids': other_prot_ids,
                'start': pstart,
                'end': pend,
                'is_decoy': isdecoy,
                'calc_mass': calc_mass,
                'delta_mass': delta_mass,
                self.qval_key: qval_key,
                'var_mods_pos': var_mods_pos
            }

        return psm_data

    def __call__(self, *args, **kwargs):
        return self.get_pepxml_data(*args, **kwargs)


class FragPipeRun:

    def __init__(self, path, db, run_id, format='tsv', decoy_tag='rev_'):
        """
        :param path: Path to results
        :param db: List of SeqRecords
        :param run_id: Identification of the run
        :param format: Format of the FragPipe output to read. Either "tsv" or "pepXML".
               "tsv" will read the psm.tsv file from each experiment, while "pepXML" will read the interact.pep.xml.
        """
        self.path = path
        self.db = db
        self.ptm_shepher_folder = 'ptm-shepherd-output'
        self.run_id = run_id
        self.format = format
        self.decoy_tag = decoy_tag
        experiments = []
        files = []
        for f in os.listdir(self.path):
            if f == self.ptm_shepher_folder or not os.path.isdir(os.path.join(self.path, f)):
                continue
            experiments.append(f)
            exp_path = os.path.join(self.path, f)
            if self.format == 'pepXML':
                file = os.path.join(exp_path, 'interact.pep.xml')
            elif self.format == 'tsv':
                file = os.path.join(exp_path, 'psm.tsv')
            else:
                sys.exit("Unknown format: " + self.format)
            files.append(file)
        self.files = files
        self.column_data = [
            ('Spectrum', 'Scan_No', -1),
            ('Peptide', 'Seq', ''),
            ('Retention', 'RTsec', np.nan),
            ('Calculated Peptide Mass', 'calc_mass', np.nan),
            ('Delta Mass', 'delta_mass', np.nan),
            ('Hyperscore', 'hyperscore', np.nan),
            ('PeptideProphet Probability', 'probability', np.nan),
            ('Protein Start', 'start', -1),
            ('Protein End', 'end', -1),
            ('Intensity', 'intensity', np.nan),
            ('Protein', 'prot_id', ''),
            ('Mapped Proteins', 'other_prot_ids', []),
            ('Assigned Modifications', 'var_mods_pos', []),
            ('Observed Modifications', 'delta_mass_mods', []),
            ('MSFragger Localization', 'delta_mass_mods_pos', []),
            (None, 'is_decoy', None),
            (None, '')
        ]
        self.experiments = experiments
        if self.format == 'tsv':
            self.fragpipe_tsv_target = load_psm_table
        elif self.format == 'pepXML':
            self.fragpipe_pepxml_target = PepXMLdataExtractor(flavour='fragpipe')
        else:
            sys.exit('Unknown format: ' + self.format)
        self.exp_psms = []
        self.__load_psms()

    def __load_psms(self):
        for f in self.files:
            if self.format == 'pepXML':
                self.exp_psms.append(pepxml.PepXML(f))
            elif self.format == 'tsv':
                self.exp_psms.append(load_psm_table(f, column_data=self.column_data, sep='\t'))

    def read_pepxml(self, n_procs):
        frag_psm_data = []
        for i in range(len(self.files)):
            print(f'\tReading sample {self.experiments[i]} ... ', end='')
            exp_pepxml = pepxml.PepXML(self.files[i])
            if len(exp_pepxml) == 0:
                psm_data = [self.fragpipe_pepxml_target.empty_sample()]
            else:
                psm_data = exp_pepxml.map(
                    self.fragpipe_pepxml_target, processes=n_procs, fdb=self.db)
            psm_data = pd.DataFrame(psm_data)
            psm_data.columns = [
                'Scan_No', 'RTsec', 'Seq',
                'prot_id', 'start', 'end',
                'is_decoy', 'calc_mass', 'delta_mass',
                'probability'
            ]
            psm_data['Run_id'] = self.run_id
            psm_data['Sample'] = self.experiments[i]
            psm_data = pepxml.qvalues(
                psm_data, key='probability', reverse=True, correction=0,
                is_decoy='is_decoy', full_output=True)
            frag_psm_data.append(psm_data)
        frag_psm_data = pd.concat(frag_psm_data)
        return frag_psm_data

    def read_fp_tsv(self):
        frag_psm_data = []
        for i in range(len(self.files)):
            print(f'\tProcessing experiment {self.experiments[i]} ... ', end='')
            psm_data = load_psm_table(self.files[i], column_data=self.column_data, sep='\t')
            psm_data['Run_id'] = self.run_id
            psm_data['Sample'] = self.experiments[i]
            psm_data['start'] = psm_data['start'] - 1
            psm_data['end'] = psm_data['end'] - 1
            frag_psm_data.append(psm_data)
        frag_psm_data = pd.concat(frag_psm_data)
        return frag_psm_data



    def read(self, n_procs=6):
        if self.format == 'pepXML':
            frag_psm_data = self.read_pepxml(n_procs=n_procs)
        elif self.format == 'tsv':
            frag_psm_data = self.read_fp_tsv()
        return frag_psm_data

