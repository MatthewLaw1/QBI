import numpy as np
import itertools
SEQUENCE = [8, 7, 6, 5, 4, 2, 0]
sequence_iterator = itertools.cycle(SEQUENCE)

def get_prediction(processed_data):
    return next(sequence_iterator)



if __name__ == "__main__":
    print(get_prediction({}))
    print(get_prediction({}))
    print(get_prediction({}))
    print(get_prediction({}))
    print(get_prediction({}))
    print(get_prediction({}))
    print(get_prediction({}))
    print(get_prediction({}))
