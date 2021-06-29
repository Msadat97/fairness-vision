from argparse import ArgumentParser, ArgumentTypeError
from dl2 import dl2lib

def str2bool(v):
    "taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_arguments():
    parser = ArgumentParser()
    dl2lib.add_default_parser_args(parser)
    parser.add_argument('--robust', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--base', type=str2bool, default=False, const=True, nargs="?")
    args = parser.parse_args()

    return args
