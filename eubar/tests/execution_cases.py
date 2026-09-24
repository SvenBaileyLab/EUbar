"""Small jobs shared by output regression and adapter tests."""
from pathlib import Path

FIXTURES = Path(__file__).parent / 'fixtures'


def arguments(name, output, fixtures=FIXTURES):
    f = lambda name: str(fixtures / name)
    common = ['--intensities', f('intensities.tsv'), '--array', f('array.tsv'),
              '--genome', f('genome.fa'), '--kmer-size', '4']
    if name.startswith('snv'):
        args = ['snv', *common, '--snv-list-file', f('variants.txt'), '--rand-n', '30']
        if 'mask' in name:
            args += ['--mask', '11010' if 'asymmetric' in name else '11011']
        if 'best' in name:
            args += ['--best-pval', '--diagnostics']
        if 'holm' in name:
            args += ['--holm']
        if 'cap' in name:
            args += ['--max-probes', '30', '--seed', '7']
        return args
    if name.startswith('scan'):
        args = ['scan', *common, '--region', 'chr1:101-108']
        if 'holm' in name:
            args += ['--best-pval', '--holm', '--rand-n', '30', '--diagnostics']
        if 'reverse' in name:
            args += ['--reverse']
        if 'pooled' in name:
            args += ['--pooled']
        return args
    if name.startswith('motifs'):
        args = ['motifs', *common, '--outdir', str(output), '--prefix', 'motif',
                '--min-F', '5', '--min-per-base', '2', '--min-support', '2',
                '--extend-left', '0', '--extend-right', '0', '--max-gaps', '0']
        if 'extend' in name:
            args += ['--extend-left', '2', '--extend-right', '2']
        if 'mask' in name:
            args += ['--mask', '11011', '--mask-candidate-probes', '50', '--mask-max-candidates', '30']
        return args
    if name == 'max_probes':
        return ['calibrate', 'max-probes', '--intensities', f('intensities.tsv'),
                '--array', f('array.tsv'), '--kmer-size', '4', '--out-prefix', str(output/'cal'),
                '--n-families', '4', '--repeats', '3', '--caps', '20,40', '--min-probes', '15', '--quiet']
    if name == 'mask_calibration':
        return ['calibrate', 'mask', '--intensities', f('intensities.tsv'),
                '--genome', f('genome.fa'), '--out-prefix', str(output/'cal'),
                '--masks', '11011', '--random-controls', '1', '--n-families', '3',
                '--stability-probes', '15', '--min-probes', '10', '--repeats', '3', '--quiet']
    raise ValueError(name)


CASES = ['snv', 'snv_best', 'snv_best_holm_cap', 'snv_mask', 'snv_mask_best_holm_cap',
         'scan', 'scan_holm', 'scan_holm_reverse', 'scan_pooled',
         'motifs', 'motifs_mask', 'motifs_extend', 'snv_mask_asymmetric_best_holm_cap', 'max_probes', 'mask_calibration']
