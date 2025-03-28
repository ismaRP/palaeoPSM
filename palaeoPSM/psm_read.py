import sys

import pandas as pd
import numpy as np
from pyteomics import pepxml
from lxml import etree
import os
from pyteomics.mass import std_aa_mass
import warnings
import re


def load_psm_table(file, column_data, sep):
    col_rename = {f[1]: f[0] for f in column_data}
    colnames = [f[0] for f in column_data]
    usecols = [f[1] for f in column_data]
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


def filter_psms(psm_data, remove_contams=None, whitelist=None, remove_decoy=True, fdr_threshold=0.05):
    if remove_contams:
        psm_data = psm_data[~psm_data['prot_id'].isin(remove_contams)]
    if whitelist:
        psm_data = psm_data[psm_data['prot_id'].isin(whitelist)]
    if remove_decoy:
        psm_data = psm_data[~psm_data['is_decoy']]
    if fdr_threshold is not None:
        psm_data = psm_data[psm_data['q'] < fdr_threshold]
    return psm_data


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
        return float(search_hit.get('ptm_result', {}).get('ptm_mass', 0))
        # return search_hit['massdiff']

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
        var_mods_pos = [0] * len(pept_seq)
        if var_mods is not None:
            for m in var_mods:
                mod_aa = pept_seq[m['position']-1]
                mass = np.round(m['mass'] - std_aa_mass[mod_aa], 4)
                var_mods_pos[m['position']-1] = mass
        return var_mods_pos

    @staticmethod
    def get_delta_mass_mods_pos(search_hit, pept_seq):
        ptm_result = search_hit.get('ptm_result')
        delta_mass_mods_pos = [0] * len(pept_seq)
        if ptm_result is not None:
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
                delta_mass_mods_pos[p-1] = float(ptm_result['ptm_mass'])
                # delta_mass_mods_weights[p-1] = w
        return np.array(delta_mass_mods_pos)  # , delta_mass_mods_weights

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

    def __init__(
            self, path, db, run_id, contams, format='tsv', decoy_tag='rev_', n_scans_path=None,
            prob_column='PeptideProphet Probability'):
        """
        :param path: Path to results
        :param db: List of SeqRecords
        :param run_id: Identification of the run
        :param format: Format of the FragPipe output to read. Either "tsv" or "pepXML".
               "tsv" will read the psm.tsv file from each experiment, while "pepXML" will read the interact.pep.xml.
        :param decoy_tag: Decoy tag
        :param n_scans_path: Path to csv files with MS1 and MS2 scans per LC-MSMS run
        :param prob_column: Some Fragpipe versions us "Proability", others "PeptideProphet Probability".
               adjust accordingly.
        """
        self.path = path
        self.db = db
        self.ptm_shepher_folder = 'ptm-shepherd-output'
        self.run_id = run_id
        self.format = format
        self.decoy_tag = decoy_tag
        self.contams = contams
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

        self.n_scans_path = n_scans_path
        self.n_scans = self.count_n_scans()

        if self.format == 'tsv':
            self.fragpipe_tsv_target = load_psm_table
            self.column_data = [
                ('Scan_No', 'Spectrum', -1),
                ('Seq', 'Peptide', ''),
                ('RTsec', 'Retention', np.nan),
                ('calc_mass', 'Calculated Peptide Mass', np.nan),
                ('delta_mass', 'Delta Mass', np.nan),
                ('hyperscore', 'Hyperscore', np.nan),
                ('probability', prob_column, np.nan),
                ('start', 'Protein Start', -1),
                ('intensity', 'Intensity', np.nan),
                ('prot_id', 'Protein', ''),
                ('other_prot_ids', 'Mapped Proteins', []),
                ('var_mods_pos', 'Assigned Modifications', []),
                ('delta_mass_mods', 'Observed Modifications', []),
                ('delta_mass_mods_pos', 'MSFragger Localization', []),
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

    def count_n_scans(self):
        if os.path.isfile(self.n_scans_path):
            print(f'Reading # of scans from {self.n_scans_path}')
            ext = os.path.splitext(self.n_scans_path)[1]
            if ext == '.csv':
                sep = ','
            elif ext == '.tsv':
                sep = '\t'
            else:
                sys.exit(f'{self.n_scans_path} is not recognised as a csv or tsv file')
            n_scans = pd.read_csv(self.n_scans_path, sep=sep)
        elif os.path.isdir(self.n_scans_path):
            print(f'Reading mzML files from {self.n_scans_path} and counting # of scans')
            mzml_files = [f for f in os.listdir(self.n_scans_path) if f.endswith('.mzML')]
            scan_counts = []
            for f in mzml_files:
                sample = f.rstrip('.mzML')
                print(f'\tReading sample {sample} ... ', end='')
                path = os.path.join(self.n_scans_path, f)
                tree = etree.parse(path)
                root = tree.getroot()
                ms1spectra = root.xpath(
                    "mzml:mzML/mzml:run/mzml:spectrumList/mzml:spectrum"
                    "[mzml:cvParam/@name='ms level' and mzml:cvParam/@value='1']",
                    namespaces={'mzml': 'http://psi.hupo.org/ms/mzml'})
                ms2spectra = root.xpath(
                    "mzml:mzML/mzml:run/mzml:spectrumList/mzml:spectrum"
                    "[mzml:cvParam/@name='ms level' and mzml:cvParam/@value='2']",
                    namespaces={'mzml': 'http://psi.hupo.org/ms/mzml'})
                scan_counts.append([sample, len(ms1spectra), len(ms2spectra)])
                print('Done')
            n_scans = pd.DataFrame(scan_counts, columns=['sample', 'n_ms1scans', 'n_ms2scans'])
            n_scans_file = os.path.join(self.n_scans_path, 'n_scans.csv')
            print(f'Writing # of scans on {n_scans_file}')
            n_scans.to_csv(n_scans_file, index=False)
        else:
            warnings.warn(f'{self.n_scans_path} is not recognised as a csv or tsv file, or a directory to mzML files.\n'
                          f'n_scans set to None.')
            n_scans = None
        return n_scans

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
            psm_data['sample'] = self.experiments[i]
            psm_data = pepxml.qvalues(
                psm_data, key='probability', reverse=True, correction=0,
                is_decoy='is_decoy', full_output=True)
            frag_psm_data.append(psm_data)
            print('Done')
        frag_psm_data = pd.concat(frag_psm_data)
        return frag_psm_data

    @staticmethod
    def get_var_mods_pos(vms, pept_seq):
        var_mods_pos = [0] * len(pept_seq)
        if not pd.isna(vms):
            vms = vms.split(',')
            for m in vms:
                m = m.split('(')
                if m[0] == 'N-term':
                    pos = 1
                elif m[0] == 'C-term':
                    pos = len(pept_seq)
                else:
                    aa = m[0][-1]
                    pos = int(m[0][:-1])
                mass = float(m[1].rstrip(')'))
                var_mods_pos[pos-1] = mass
        return var_mods_pos

    @staticmethod
    def get_delta_mass_mods_pos(delta_mass_mods, pept_seq, msfragger_loc, delta_mass, mods_regex):
        mod1_pos = [0] * len(pept_seq)
        mod2_pos = [0] * len(pept_seq)
        delta_mass_pos = [0] * len(pept_seq)
        if not pd.isna(delta_mass_mods) and not pd.isna(msfragger_loc) and delta_mass > 0.5:
            msfragger_loc = [i for i, c in enumerate(msfragger_loc) if c.islower()]
            mods = delta_mass_mods.split(' + ')
            if len(mods) > 0: #  This just checks if a PTM has been actually found in the string
                for p in msfragger_loc:
                    mod1_pos[p] = mods[0]
                    if len(mods) == 2:
                        mod2_pos[p] = mods[1]
                    delta_mass_pos[p] = delta_mass
        return delta_mass_pos, mod1_pos, mod2_pos

    @staticmethod
    def get_delta_mass_mods(delta_mass_mods, mods_regex):
        mods = None
        if not pd.isna(delta_mass_mods):
            mods = delta_mass_mods.split("; ")[0]
            mods = mods_regex.findall(mods)
            if len(mods) > 0:
                mods = ' + '.join(mods)
            else:
                print(f'Could not extract PTM from {delta_mass_mods}')
                return delta_mass_mods
        return mods

    @staticmethod
    def tsv_is_decoy(prot_id, other_prot_ids, decoy_tag):
        if not pd.isna(other_prot_ids):
            other_prot_ids = other_prot_ids.split(', ')
            other_prot_ids.append(prot_id)
            return all(p.startswith(decoy_tag) for p in other_prot_ids)
        else:
            return prot_id.startswith(decoy_tag)

    def read_fp_tsv(self):
        frag_psm_data = []
        for i in range(len(self.files)):
            print(f'\tReading sample {self.experiments[i]} ... ', end='')
            psm_data = load_psm_table(self.files[i], column_data=self.column_data, sep='\t')
            psm_data['sample'] = self.experiments[i]
            psm_data['is_decoy'] = (
                psm_data
                .apply(lambda x: self.tsv_is_decoy(x['prot_id'], x['other_prot_ids'], 'rev_'), axis=1))
            psm_data = pepxml.qvalues(
                psm_data, key='probability', reverse=True, correction=0,
                is_decoy='is_decoy', full_output=True)
            frag_psm_data.append(psm_data)
            print('Done')
        frag_psm_data = pd.concat(frag_psm_data)
        frag_psm_data['Run_id'] = self.run_id
        frag_psm_data['start'] = frag_psm_data['start'] - 1
        frag_psm_data['var_mods_pos'] = (
            frag_psm_data
            .apply(lambda x: self.get_var_mods_pos(x['var_mods_pos'], x['Seq']), axis=1))

        # mods_regex = re.compile(r'Mod\d: (.+?)[,(]')
        mods_regex = re.compile(r'Mod\d: (.+?)(?:, | \((?:Theoretical|PeakApex))')
        frag_psm_data['delta_mass_mods'] = (
            frag_psm_data
            .apply(lambda x: self.get_delta_mass_mods(x['delta_mass_mods'], mods_regex), axis=1)
        )
        frag_psm_data['other_prot_ids'] = frag_psm_data['other_prot_ids'].fillna('')
        frag_psm_data['other_prot_ids'] = frag_psm_data['other_prot_ids'].str.split(', ')
        frag_psm_data[['delta_mass_pos', 'delta_mass_mod1_pos', 'delta_mass_mod2_pos']] = (
            frag_psm_data
            .apply(lambda x: self.get_delta_mass_mods_pos(
                x['delta_mass_mods'], x['Seq'], x['delta_mass_mods_pos'], x['delta_mass'], mods_regex),
                result_type='expand', axis=1)
        )

        return frag_psm_data

    def read(self, n_procs=1, save_path=None, **kwargs):
        if save_path is None or not os.path.exists(save_path):
            print(f'Collecting PSMs from {self.format} files')
            if self.format == 'pepXML':
                frag_psm_data = self.read_pepxml(n_procs=n_procs)
            elif self.format == 'tsv':
                frag_psm_data = self.read_fp_tsv()
            if save_path is not None:
                print(f'Saving PSMs to {save_path}')
                frag_psm_data.to_csv(save_path, index=False)
        else:
            print(f'Loading PSMs from {save_path}')
            frag_psm_data = pd.read_csv(save_path)

        frag_psm_data = filter_psms(frag_psm_data, **kwargs)

        return frag_psm_data

