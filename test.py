import numpy as np
import torch
import pickle
import ipdb



def copy(in_value):
    if isinstance(in_value, dict):
        out_value = {}
        for key in in_value.keys():
            out_value[key] = copy(in_value[key])
        return out_value
    if isinstance(in_value, (np.ndarray, np.generic)):
        return np.copy(in_value)
    
    return in_value
    



def compare_results_with_truth(results, truth, label=""):
    """ results and truth are both dictionaries witht he same keys.
    Values should be the same for the test to pass.
    """
    if type(results) != type(truth):
        print(f"Failed {label}: type mismatch {type(results)} {type(truth)}")
        return False
 
    if isinstance(results, dict):
        success = True
        for key in truth.keys():
            next_label = f"{label}['{key}']"
            if not key in results:
                print(f"Failed {label}: missing from results");
                success = False
                continue
            if not compare_results_with_truth(results[key], truth[key], next_label):
                success = False
                continue
        return success
    
    if isinstance(results, list):
        success = True
        if len(results) != len(truth):
            print(f"Failed {label}: array length mismatch {len(results)} {len(truth)}");
            return False
        for idx in range(len(results)):
            next_label = f"{label}[{idx}]"
            if not compare_results_with_truth(results[idx], truth[idx], next_label):
                return False
        return True

    if isinstance(results, (np.ndarray, np.generic)):
        # Do we need an epsilon/fuzzy comparison?
        if not np.array_equal(truth, results):
            print(f"Failed {label}: numpy array mismatch")
            compare_numpy_arrays(results, truth, label)
            return False
        return True
    
    if isinstance(result, float):
        # Do we need an epsilon/fuzzy comparison?
        if truth != results:
            print(f"Failed {label}: float value mismatch, {results} != {truth}")
            return False
        return True

    # Do we need an epsilon/fuzzy comparison?
    if truth != results:
        print(f"Failed {label}: value mismatch, {results} != {truth}")
        return False
    return True
    


def compare_numpy_arrays(results, truth, label = ""):
    if results.shape != truth.shape:
        print(f"{label} shape mismatch {results.shape} != {truth.shape}") 
        return False
    if results.dtype != truth.dtype:
        print(f"{label} type mismatch {results.dtype} != {truth.dtype}") 
        return False
    if len(results.shape) == 0:
        if results != truth:
            print(f"{label} value mismatch {results} != {truth} (truth)") 
            return False
        return True
    for idx in range(results.shape[0]):
        next_label = f"{label}[{idx}]"
        if not compare_numpy_arrays(results[idx], truth[idx], next_label):    
            return False
    return True


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
    


def convert_to_numpy(in_val):
    """ Change tensors to numpy ararys in the dict.
    """
    if torch.is_tensor(in_val):
        return in_val.detach().cpu().numpy()
    if isinstance(in_val, dict):
        out_val = {}
        for key, val in in_val.items():
            out_val[key] = convert_to_numpy(val)
        return out_val
    if isinstance(in_val, list):
        out_val = []
        for val in in_val:
            out_val.append(convert_to_numpy(val))
        return out_val

    return in_val

    
        

def evaluate_results(results, tag):
    truth = load_truth(f"test_data/{tag}")
    if truth is None:
        print(f"Iniitialize truth for {tag}")
        save_truth(results, f"test_data/{tag}")
        return True

    if compare_results_with_truth(results, truth, tag):
        print(f"{tag} test passed ")
        return True
    else:
        print(f"{tag} test failed")
        return False

    
        
