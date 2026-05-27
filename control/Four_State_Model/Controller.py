from . import Model
from .. import Vehicle_Params

import numpy as np


def main():
    A, B, E = Model.build_state_matrices()


if __name__ == "__main__":
    main()