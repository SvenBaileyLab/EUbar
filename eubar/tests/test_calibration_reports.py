import pandas as pd
import pytest

from execution_cases import FIXTURES
from eubar.core.calibration_reporting import _write_max_plot, _write_mask_plot


@pytest.mark.parametrize('name', ['max_probes', 'mask_calibration'])
def test_calibration_pdf(name, tmp_path):
    root = FIXTURES/'expected'/name
    summary = pd.read_csv(root/'cal.summary.tsv', sep='\t')
    recommendation = pd.read_csv(root/'cal.recommendation.tsv', sep='\t')
    output = tmp_path/'diagnostics.pdf'
    if name == 'max_probes':
        quartiles = pd.read_csv(root/'cal.pvalue_quartiles.tsv', sep='\t')
        assert _write_max_plot(summary, recommendation, quartiles, str(output))
    else:
        assert _write_mask_plot(summary, recommendation, str(output))
    data = output.read_bytes()
    assert data.startswith(b'%PDF-')
    assert b'%%EOF' in data[-100:]
    assert len(data) > 1000
