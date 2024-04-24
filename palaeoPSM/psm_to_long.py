import pandas as pd
import numpy as np


def merge_true_seq(position_counts, seq_records):
    seq_list = []
    pos_list = []
    id_list = []
    for rec in seq_records:
        seq = str(rec.seq)
        seq_list.extend(seq)
        pos_list.extend(list(range(1, len(seq) + 1)))
        id_list.extend([rec.id] * len(seq))
    seqs_df = pd.DataFrame({
        'pep_letters': seq_list, 'pep_positions': pos_list, 'prot_id': id_list})
    seqs_df = seqs_df.drop_duplicates().sort_values('pep_positions')
    seqs_df['correct'] = True

    # Here we merge with the actual BLG positions and amino acids
    # To spot where there are wrong positions
    position_counts = position_counts.merge(
        seqs_df, how='left', on=['pep_letters', 'pep_positions', 'prot_id'])
    position_counts.loc[position_counts['correct'].isna(), 'correct'] = False

    # Mirror incorrect or coming from q_val > 0.05 PSMs
    # mirror = (~position_counts['correct']) & (position_counts['gt_005fdr'])

    position_counts['count_mirror'] = position_counts['count']
    position_counts['rel_count_mirror'] = position_counts['rel_count']

    position_counts.loc[~position_counts['correct'], 'count_mirror'] = (
            position_counts.loc[~position_counts['correct'], 'count_mirror'] * -1)
    position_counts.loc[~position_counts['correct'], 'rel_count_mirror'] = (
            position_counts.loc[~position_counts['correct'], 'rel_count_mirror'] * -1)

    return position_counts


def psm_to_long(psm_data_df, seq_records=None, q_val_threshold=0.05):
    psm_data_df['pep_positions'] = (
            psm_data_df['start'] +
            psm_data_df['Seq'].str.len().apply(lambda x: np.arange(1, x + 1, 1)))
    psm_data_df['pep_letters'] = psm_data_df['Seq'].str.split('').str[1:-1]

    psm_data_df['gt_005fdr'] = psm_data_df['q'] > q_val_threshold

    position_counts = (
        psm_data_df
        .explode(['pep_positions', 'pep_letters'], ignore_index=True)
        .groupby(['Run_id', 'Sample', 'gt_005fdr', 'pep_positions', 'pep_letters'])
        .agg('size')
        .reset_index().rename(columns={0: 'count'}))

    position_counts['total_counts'] = (
        position_counts
        .groupby(['Run_id', 'Sample', 'prot_id', 'gt_005fdr', 'pep_positions'])['count']
        .transform('sum'))
    position_counts['rel_count'] = position_counts['count'] / position_counts['total_counts']

    return position_counts

