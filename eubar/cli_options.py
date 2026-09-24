"""Parse explicit CLI overrides without supplying task defaults a second time."""
import argparse


def parse_overrides(parser, argv):
    # The task parser is a fresh instance. Required inputs come from YAML;
    # omitted flags must not overwrite those values with parser defaults.
    parser._defaults.clear()
    for action in list(parser._actions):
        if not action.option_strings:
            parser._remove_action(action)
            continue
        action.required = False
        action.default = argparse.SUPPRESS
    for group in parser._mutually_exclusive_groups:
        group.required = False
    return vars(parser.parse_args(argv))
