import h5py
import numpy as np


def read_dataset(paths, dataset_name, max_rows=None):

    n = []
    for i, path in enumerate(paths):
        with h5py.File(path, "r") as file:
            arr = np.array(file[dataset_name])
            n.append(arr.shape[0])
            if i==0: shape = arr.shape[1:]
            else: shape = tuple(max(dims) for dims in zip(shape, arr.shape[1:]))
            if max_rows is not None and sum(n) >= max_rows: 
                n[-1] = max_rows - sum(n[:-1])
                break
    
    dataset = np.full((sum(n), *shape), np.nan)
    for i, _ in enumerate(n):
        with h5py.File(paths[i], "r") as file:
            data = np.array(file[dataset_name])[:n[i]]
            slices = tuple(slice(0, dim) for dim in data.shape[1:])
            dataset[(slice(sum(n[:i]),sum(n[:i+1])),)+slices] = data

    return dataset  


def topk(data, topk=16):
    temp = np.where(np.isnan(data), -np.inf, data)
    idx = np.argpartition(temp, -topk, axis=-1)[:,-topk:] 
    topk_data = data[np.arange(len(idx))[:,None], idx]
    return topk_data, idx


def compute_far(y, y_pred, number_of_signals, total_sample_length, desired_far):
    num_points = 1000
    thresholds = 1 - np.logspace(-6, 0, num_points, endpoint=True)
    y = y.flatten()
    y_pred = y_pred.flatten()
    mask = np.isfinite(y) & np.isfinite(y_pred)
    y = y[mask].astype(bool)
    y_pred = y_pred[mask]
    order = np.argsort(-y_pred)
    y_sorted = y[order]
    y_pred_sorted = y_pred[order]
    tp_cumsum = np.cumsum(y_sorted)
    fp_cumsum = np.cumsum(~y_sorted)
    k = np.searchsorted(-y_pred_sorted, -thresholds, side="left")
    tp = np.zeros_like(thresholds, dtype=float)
    fp = np.zeros_like(thresholds, dtype=float)
    mask = k > 0
    tp[mask] = tp_cumsum[k[mask] - 1]
    fp[mask] = fp_cumsum[k[mask] - 1]
    detection_ratio = tp / number_of_signals
    far = fp / total_sample_length
    idx = (np.abs(far - desired_far)).argmin()
    far_opt, detection_ratio_opt = far[idx], detection_ratio[idx]
    threshold_opt = thresholds[idx]
    print(threshold_opt, detection_ratio_opt)
    return far, detection_ratio, threshold_opt, far_opt, detection_ratio_opt


def greedy_match(y_pred, t_pred, t, sigma):
    t = t.copy()
    n = t.shape[0]
    y = np.full_like(y_pred, 0)
    y[np.isnan(y_pred)] = np.nan
    pred_idx = np.full_like(t, np.nan)
    signal_idx = np.full_like(t, np.nan)
    order = np.argsort(-y_pred, axis=-1)
    for best in order.T:
        best_t_pred = t_pred[np.arange(n),best]
        # only consider samples with any signals and finite t prediction
        mask_samples = np.any(np.isfinite(t), axis=-1) & np.isfinite(best_t_pred)
        # find closest signal to prediction
        delta = np.abs(t[mask_samples]-best_t_pred[mask_samples,None])
        match = np.nanargmin(delta, axis=-1)
        delta_match = delta[np.arange(match.shape[0]),match]
        # y is only true if delta <= sigma
        cond = delta_match <= sigma
        samples = np.arange(n)[mask_samples]
        pred = best[mask_samples]
        y[samples, pred] = cond
        t[samples, match] = np.where(cond, np.nan, t[samples, match])
        pred_idx[samples[cond], match[cond]] = pred[cond]
        signal_idx[samples[cond], match[cond]] = match[cond]
    return y, signal_idx, pred_idx


def match(arr, idx): 
    mask = np.isnan(idx)
    idx = np.where(mask, 0, idx).astype(np.int64)
    samples = np.arange(arr.shape[0])[:, None]
    arr = arr[samples, idx]
    arr[mask] = np.nan
    return arr