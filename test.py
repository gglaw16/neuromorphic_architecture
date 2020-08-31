import numpy as np
import pickle
import ipdb


def compare_results_with_truth(results, truth):
    """ results and truth are both dictionaries witht he same keys.
    Values should be the same for the test to pass.
    """
    success = True
    for key in truth.keys():
        if not key in results:
            print(f"Failed: {key} missing from results");
            success = False
            continue
        if type(truth[key]) != type(results[key]):
            print(f"Failed: {key} type mismatch, {type(results[key])} != {type(truth[key])}");
            success = False
            continue
        if isinstance(truth[key], float):
            # Do we need an epsilon/fuzzy comparison?
            if truth[key] != results[key]:
                print(f"Failed: {key} value mismatch, {results[key]} != {truth[key]}")
                success = False
            continue
        if isinstance(truth[key], (np.ndarray, np.generic)):
            # Do we need an epsilon/fuzzy comparison?
            if not np.array_equal(truth[key], results[key]):
                print(f"Failed: {key} array mismatch")
                success = False
            continue
    return success


def load_truth(rootpath):
    """ Just use pickle for now. (even though it is no secure).
    rootpath:  filepath without an extension.
    """
    try:
        filepath = f"{rootpath}.pkl"
        with open(filepath, 'rb') as fp:
            return pickle.load(fp)
    except:
        print("Error loading truth.")
        return None

            
def save_truth(results, rootpath):
    """ Just use pickle for now. (even though it is no secure).
    result: dictionary of values (including numpy arrays).
    rootpath:  filepath without an extension.
    """
    filepath = f"{rootpath}.pkl"
    with open(filepath, 'wb') as fp:
        pickle.dump(results, fp)
    


