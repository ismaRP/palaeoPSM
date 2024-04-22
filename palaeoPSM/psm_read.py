import sys

import pandas as pd
import numpy as np
from pyteomics import pepxml
import os
from pyteomics.mass import std_aa_mass
import warnings
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
    def __init__(self, flavour, fields, fdb, decoy_tag='rev_'):
        self.decoy_tag = decoy_tag
        self.fdb = fdb
        self.fields = fields

        # These retrieve data from the scan, like spectrum id or RT
        self.scan_functions = {}
        # These retrieve data from the search_hit
        self.hit_functions = {
            'Seq': [self.get_seq, ''],
            'prot_id': [self.get_prot_id, ''],
            'calc_mass': [self.get_calc_mass, np.nan],
            'var_mods_pos': [self.get_var_mods_pos, [], 'Seq'],
            'delta_mass': [self.get_delta_mass, np.nan],
            'delta_mass_mods_pos': [self.get_delta_mass_mods_pos, [], 'Seq'],
            # 'delta_mass_mods_weights': [lambda _, x: x[1], [], 'delta_mass_mods_pos'],
            'is_decoy': [lambda x: self.is_decoy(x, self.decoy_tag), None],
            'other_prot_ids': [self.get_other_prot_ids, []],
            'start': [lambda _, x, y: self.get_prot_start(x, y, self.fdb), '-1', 'Seq', 'prot_id']
        }

        if flavour == 'mascot':
            self.scan_functions['Scan_No'] = [self.mascot_scan_no, -1]
            self.scan_functions['RTsec'] = [self.mascot_rt, np.nan]
            self.hit_functions['expectation'] = [self.mascot_expectation, np.nan]
            self.hit_functions['score'] = [lambda x: self.get_score(x, 'ionscore'), np.nan]
        elif flavour == 'fragpipe':
            self.scan_functions['Scan_No'] = [self.fragpipe_scan_no, -1]
            self.scan_functions['RTsec'] = [self.fragpipe_rt, np.nan]
            self.hit_functions['probability'] = [self.fragpipe_probability, np.nan]
            self.hit_functions['score'] = [lambda x: self.get_score(x, 'hyperscore'), np.nan]

        scan_fields = []
        hit_fields = []
        for f in fields:
            if f in self.scan_functions:
                scan_fields.append(f)
            elif f in self.hit_functions:
                hit_fields.append(f)
            else:
                warnings.warn(f'Field {f} not recognized. It will be ignored.')
        self.scan_fields = scan_fields
        self.hit_fields = hit_fields

        empty_sample = {}
        for field in self.scan_fields:
            empty_sample[field] = self.scan_functions[field][1]
        for field in self.hit_fields:
            empty_sample[field] = self.hit_functions[field][1]
        self.empty_sample = empty_sample

    @staticmethod
    def get_prot_id(search_hit):
        return search_hit['proteins'][0]['protein']

    @staticmethod
    def get_seq(search_hit):
        return search_hit['peptide']

    @staticmethod
    def get_other_prot_ids(search_hit):
        return [p['protein'] for p in search_hit['proteins'][1:]]

    @staticmethod
    def get_calc_mass(search_hit):
        return search_hit['calc_neutral_pep_mass']

    @staticmethod
    def get_delta_mass(search_hit):
        return search_hit['massdiff']

    @staticmethod
    def get_score(search_hit, score):
        return search_hit['search_score'][score]

    @staticmethod
    def is_decoy(search_hit, decoy_tag):
        return all(protein['protein'].startswith(decoy_tag) for protein in search_hit['proteins'])

    @staticmethod
    def get_prot_start(pept_seq, prot_id, fdb):
        seq_rec = fdb.get(prot_id, None)
        if seq_rec is None:
            prot_start = -1
        else:
            prot_seq = seq_rec.seq
            prot_start = prot_seq.find(pept_seq)
        return prot_start

    @staticmethod
    def get_var_mods_pos(search_hit, pept_seq):
        var_mods = search_hit.get('modifications')
        if var_mods is None:
            return []
        var_mods_pos = [0] * len(pept_seq)
        for m in var_mods:
            mod_aa = pept_seq[m['position']-1]
            mass = np.round(m['mass'] - std_aa_mass[mod_aa], 3)
            var_mods_pos[m['position']-1] = mass
        return var_mods_pos

    @staticmethod
    def get_delta_mass_mods_pos(search_hit, pept_seq):
        ptm_result = search_hit.get('ptm_result')
        if ptm_result is None:
            return []
        delta_mass_mods_pos = [0] * len(pept_seq)
        # delta_mass_mods_weights = [0] * len(pept_seq)
        # Loop through positions and insert mass into delta_mass_mods_pos
        pos = ptm_result['localization'].split('_')
        # w = 1/len(pos)
        for p in pos:
            if p == '':
                break
            p = int(p)
            if p > len(pept_seq):
                continue
            delta_mass_mods_pos[p-1] = ptm_result['ptm_mass']
            # delta_mass_mods_weights[p-1] = w
        return delta_mass_mods_pos  # , delta_mass_mods_weights

    @staticmethod
    def mascot_expectation(search_hit):
        return search_hit['search_score']['expect']

    @staticmethod
    def mascot_scan_no(psm):
        return int(psm['spectrum'].split(' ')[2].split('=')[1])

    @staticmethod
    def mascot_rt(psm):
        return float(psm['search_specification'].split(' ')[2].split('(')[1].rstrip(')'))

    @staticmethod
    def fragpipe_probability(search_hit):
        return search_hit['analysis_result'][0]['peptideprophet_result']['probability']

    @staticmethod
    def fragpipe_scan_no(psm):
        return psm['start_scan']

    @staticmethod
    def fragpipe_rt(psm):
        return psm['retention_time_sec']

    def get_pepxml_data(self, psm):
        data = {}
        for field in self.scan_fields:
            f = self.scan_functions[field][0](psm)
            data[field] = f
        search_hit = psm.get('search_hit')
        if search_hit is None:
            for field in self.hit_fields:
                data[field] = self.hit_functions[field][1]
        else:
            for field in self.hit_fields:
                args = [data[k] for k in self.hit_functions[field][2:]]
                data[field] = self.hit_functions[field][0](search_hit[0], *args)
        return data

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
        self.experiments = experiments

        if self.format == 'tsv':
            self.fragpipe_tsv_target = load_psm_table
            self.column_data = [
                ('Scan_No', 'Spectrum', -1),
                ('Seq', 'Peptide', ''),
                ('RTsec', 'Retention', np.nan),
                ('calc_mass', 'Calculated Peptide Mass', np.nan),
                ('delta_mass', 'Delta Mass', np.nan),
                ('hyperscore', 'Hyperscore', np.nan),
                ('probability', 'PeptideProphet Probability', np.nan),
                ('start', 'Protein Start', -1),
                ('intensity', 'Intensity', np.nan),
                ('prot_id', 'Protein', ''),
                ('other_prot_ids', 'Mapped Proteins', []),
                ('var_mods_pos', 'Assigned Modifications', []),
                ('delta_mass_mods', 'Observed Modifications', []),
                ('delta_mass_mods_pos', 'MSFragger Localization', []),
                ('is_decoy', None, None),
                ('', None, None)
            ]
        elif self.format == 'pepXML':
            self.fields = [
                'Scan_No',
                'RTsec',
                'Seq',
                'calc_mass',
                'delta_mass',
                'score',
                'probability',
                'prot_id',
                'start',
                'other_prot_ids',
                'var_mods_pos',
                'delta_mass_mods_pos',
                'is_decoy',
            ]
            self.fragpipe_pepxml_target = PepXMLdataExtractor(
                flavour='fragpipe', fields=self.fields, fdb=self.db)
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
                psm_data = [self.fragpipe_pepxml_target.empty_sample]
            else:
                psm_data = exp_pepxml.map(
                    self.fragpipe_pepxml_target, processes=n_procs)
            psm_data = pd.DataFrame(psm_data)
            psm_data['Run_id'] = self.run_id
            psm_data['Sample'] = self.experiments[i]
            psm_data = pepxml.qvalues(
                psm_data, key='probability', reverse=True, correction=0,
                is_decoy='is_decoy', full_output=True)
            frag_psm_data.append(psm_data)
            print('Done')
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

