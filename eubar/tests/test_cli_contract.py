import pytest
from execution_cases import FIXTURES
from test_execution import run_cli


@pytest.mark.parametrize('command', ['array','intensities','snv','scan','motifs','calibrate','run','template'])
def test_public_help_matches_reference(command):
    result = run_cli([command, '--help'])
    assert result.returncode == 0
    assert result.stdout == (FIXTURES/'help'/f'{command}.txt').read_text()
