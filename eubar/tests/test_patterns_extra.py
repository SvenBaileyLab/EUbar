import pytest
from eubar.core.patterns import SequenceMask
from eubar.core.masking import _signature_codes
from eubar.core.sequence import reverse_complement


@pytest.mark.parametrize('k', [1, 8, 40])
def test_contiguous_metadata_keeps_legacy_span(k):
    pattern = SequenceMask.from_options(k)
    assert pattern.span == k
    assert pattern.informative == tuple(range(k))
    assert pattern.reverse_informative == tuple(reversed(range(k)))
    assert pattern.is_contiguous


@pytest.mark.parametrize('mask', ['11010', '001101', '11111'])
def test_mask_signatures_and_reverse_positions(mask):
    pattern = SequenceMask.from_options(8, mask)
    assert pattern.span == len(mask)
    seq = 'ACGTACGTACGT'
    forward, reverse, vf, vr = _signature_codes(seq, pattern)
    for i in range(len(forward)):
        window = seq[i:i+pattern.span]
        assert vf[i] and vr[i]
        assert int(forward[i]) == pattern.signature_code(window)
        assert int(reverse[i]) == pattern.signature_code(reverse_complement(window))


def test_ignored_bases_do_not_change_signature():
    pattern = SequenceMask.parse('11010')
    assert pattern.signature_code('ACGTG') == pattern.signature_code('ACNTN')
    assert pattern.signature_code('ANGTG') is None


def test_array_skips_ambiguous_bases_without_shifting_offsets(tmp_path, capsys):
    from eubar.array import run_array
    from eubar.task_config import ArrayConfig
    genome = tmp_path/'genome.fa'; genome.write_text('>chrN\nACNTAACG\n')
    bed = tmp_path/'probes.bed'; bed.write_text('chrN\t0\t8\nchrN\t0\t8\nmissing\t0\t5\n')
    output = tmp_path/'array.tsv'
    run_array(ArrayConfig(bed=str(bed), genome=str(genome), output=str(output), kmer_size=2))
    assert output.read_text() == 'AA\tchrN:0-8;1;5\nAC\tchrN:0-8;2;1 6\nCG\tchrN:0-8;1;7\nTA\tchrN:0-8;1;4\n'
    assert 'Chromosome missing not found' in capsys.readouterr().out
