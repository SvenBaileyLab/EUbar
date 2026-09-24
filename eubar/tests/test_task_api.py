import importlib

import pytest
import yaml

from eubar import config, run
from eubar.api import TASKS
from eubar.task_config import ArrayConfig, SnvConfig, TaskConfigError
from execution_cases import CASES, FIXTURES, arguments
from output_comparison import assert_output_equal


def yaml_from_cli(name, out):
    argv = arguments(name, out)
    module = importlib.import_module('eubar.' + argv[0])
    parser = module.build_parser()
    parsed = vars(parser.parse_args(argv[1:]))
    task = {'max_probes': 'calibrate_max_probes', 'mask_calibration': 'calibrate_mask'}.get(name, argv[0])
    spec = config.TASK_SPECS[task]
    raw = {'task': task}
    used = set()
    for section in ('inputs', 'parameters', 'output'):
        values = {}
        for key, option in getattr(spec, section).items():
            if not option.cli:
                continue
            action = parser._option_string_actions[option.cli]
            if action.dest in used:
                continue
            used.add(action.dest)
            value = parsed[action.dest]
            if value is None:
                continue
            if action.dest == 'combine_revcomp':
                value = not value
            if isinstance(value, tuple):
                value = list(value)
            values[key] = value
        raw[section] = values
    raw['output']['stdout'] = str(out/'stdout.txt')
    return raw, TASKS[task][2].from_values(parsed), module


# Sparse masked families can have linearly dependent model columns. The pooled
# scan also uses synthetic intensities below -1, outside log1p's domain. Keep
# these baseline edge cases, but silence only their known warnings here.
def yaml_case(name):
    marks = []
    if name in {'snv_mask', 'snv_mask_best_holm_cap',
                'snv_mask_asymmetric_best_holm_cap'}:
        marks.append(pytest.mark.filterwarnings(
            r'ignore:^The design matrix is rank-deficient\.:statsmodels.tools.sm_exceptions.SingularMatrixWarning:eubar\.core\.regression$'
        ))
    if name == 'scan_pooled':
        marks.append(pytest.mark.filterwarnings(
            r'ignore:^invalid value encountered in log1p$:RuntimeWarning:pandas\.core\.arraylike$'
        ))
    return pytest.param(name, marks=marks, id=name)


@pytest.mark.parametrize('name', [yaml_case(name) for name in CASES])
def test_yaml_calls_api_without_reparsing(name, tmp_path, monkeypatch):
    raw, expected_options, module = yaml_from_cli(name, tmp_path)
    path = tmp_path/'task.yaml'
    path.write_text(yaml.safe_dump(raw))
    cfg = config.load_yaml(path)
    actual_options, _ = config.task_options(cfg)
    assert actual_options == expected_options

    def unexpected(*args, **kwargs):
        pytest.fail('YAML execution must not build or parse CLI arguments')

    monkeypatch.setattr(module, 'build_parser', unexpected)
    monkeypatch.setattr(run, 'build_argv', unexpected)
    assert run.main([str(path)]) == 0
    expected = FIXTURES/'expected'/name
    files = [p for p in expected.iterdir() if p.suffix in {'.tsv', '.meme'} or p.name == 'stdout.txt']
    for file in files:
        assert_output_equal(tmp_path/file.name, file)


def test_array_api_and_pipeline(tmp_path, monkeypatch):
    from eubar.array import run_array
    api_out = tmp_path/'api.tsv'
    assert run_array(ArrayConfig(bed=str(FIXTURES/'probes.bed'), genome=str(FIXTURES/'genome.fa'), kmer_size=4, output=str(api_out))) == 0
    assert api_out.read_bytes() == (FIXTURES/'array.tsv').read_bytes()
    first = {'task':'array', 'inputs':{'bed':str(FIXTURES/'probes.bed'), 'genome':str(FIXTURES/'genome.fa')},
             'parameters':{'kmer_size':4}, 'output':{'file':'array.tsv'}}
    second = {'task':'snv', 'inputs':{'array':'array.tsv', 'intensities':str(FIXTURES/'intensities.tsv'),
              'genome':str(FIXTURES/'genome.fa'), 'variants':str(FIXTURES/'variants.txt')},
              'parameters':{'kmer_size':4, 'rand_n':30}, 'output':{'stdout':'snv.tsv'}}
    for name, raw in [('array.yaml',first),('snv.yaml',second),('pipeline.yaml',{'task':'pipeline','steps':['array.yaml','snv.yaml']})]:
        (tmp_path/name).write_text(yaml.safe_dump(raw))
    monkeypatch.chdir(tmp_path.parent)
    assert run.main([str(tmp_path/'pipeline.yaml')]) == 0
    assert_output_equal(tmp_path/'snv.tsv', FIXTURES/'expected/snv/stdout.txt')


def test_override_keeps_yaml_values(tmp_path):
    raw, _, _ = yaml_from_cli('snv_best_holm_cap', tmp_path)
    raw['parameters']['jobs'] = 7
    path = tmp_path/'task.yaml'; path.write_text(yaml.safe_dump(raw))
    assert run.main([str(path),'--jobs','1']) == 0
    assert_output_equal(tmp_path/'stdout.txt', FIXTURES/'expected/snv_best_holm_cap/stdout.txt')


def test_pipeline_cycle_and_first_failure(tmp_path, monkeypatch):
    (tmp_path/'cycle.yaml').write_text('task: pipeline\nsteps: [cycle.yaml]\n')
    assert run.main([str(tmp_path/'cycle.yaml')]) == 2
    for name in ['one','two']:
        (tmp_path/f'{name}.yaml').write_text('task: array\ninputs: {bed: probes.bed, genome: genome.fa}\noutput: {file: array.tsv}\n')
    (tmp_path/'pipeline.yaml').write_text('task: pipeline\nsteps: [one.yaml, two.yaml]\n')
    calls = []
    def fail(*args):
        calls.append(args)
        return 17
    monkeypatch.setattr(run, '_dispatch', fail)
    assert run.main([str(tmp_path/'pipeline.yaml')]) == 17
    assert len(calls) == 1


def test_invalid_api_config_fails_before_io():
    from eubar.snv import run_snv
    with pytest.raises(TaskConfigError, match='jobs'):
        run_snv(SnvConfig(intensities='missing', genome='missing', array='missing', snv_list='chr1:1:A>C', jobs=0))


def test_defaults_and_false_flags(tmp_path):
    path = tmp_path/'motifs.yaml'
    path.write_text('task: motifs\ninputs: {intensities: TF.int, array: array.tsv}\nparameters: {no_combine_revcomp: false}\n')
    options, _ = config.task_options(config.load_yaml(path))
    assert options.kmer_size == 8
    assert options.combine_revcomp is True
    path.write_text(path.read_text().replace('false', 'true'))
    options, _ = config.task_options(config.load_yaml(path))
    assert options.combine_revcomp is False


@pytest.mark.parametrize('raw', [
    {'task':'calibrate_max_probes','inputs':{'intensities':str(FIXTURES/'intensities.tsv')},'output':{'prefix':'cal'}},
    {'task':'snv','inputs':{'intensities':str(FIXTURES/'intensities.tsv'),'genome':str(FIXTURES/'genome.fa'),'array':str(FIXTURES/'array.tsv'),'variants':'empty.txt'}},
])
def test_yaml_task_errors_return_usage_status(raw, tmp_path, capsys):
    path = tmp_path/'task.yaml'; path.write_text(yaml.safe_dump(raw))
    (tmp_path/'empty.txt').write_text('')
    assert run.main([str(path)]) == 2
    assert 'error:' in capsys.readouterr().err


def test_bad_numeric_options_rejected_before_io(tmp_path):
    from eubar.array import run_array
    with pytest.raises(TaskConfigError, match='kmer_size must be int'):
        run_array(ArrayConfig(bed='missing', genome='missing', output='missing', kmer_size='4'))
    from eubar.task_config import MotifsConfig
    with pytest.raises(TaskConfigError, match='beta'):
        MotifsConfig.from_values({'beta':True})


def test_dry_run_does_not_execute_or_create_output(tmp_path, monkeypatch, capsys):
    raw, _, module = yaml_from_cli('snv_best', tmp_path)
    path = tmp_path/'job.yaml'; path.write_text(yaml.safe_dump(raw))
    def unexpected(*args):
        pytest.fail('dry run executed a task')
    monkeypatch.setattr(run, '_dispatch', unexpected)
    assert run.main([str(path),'--dry-run','--jobs','2']) == 0
    assert '--jobs 2' in capsys.readouterr().out
    assert not (tmp_path/'stdout.txt').exists()
