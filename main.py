""" Transpile a Keras model to a CIRCOM circuit.

Usage:
    main.py <model.h5> [-o <output>] [--raw]
    main.py (-h | --help)

Options:
    -h --help                       Show this screen.
    -o <output> --output=<output>   Output directory [default: output].
    --raw                           Output raw model outputs instead of the argmax of outputs [default: False].

"""
from docopt import docopt

from torch2circom import circom, transpiler


def main():
    """Main entry point of the app"""
    args = docopt(__doc__)
    circom.dir_parse("circomlib-ml/circuits/", skips=["util.circom", "circomlib-matrix", "circomlib", "crypto"])
    transpiler.transpile(args["--output"])


if __name__ == "__main__":
    """This is executed when run from the command line"""
    main()
