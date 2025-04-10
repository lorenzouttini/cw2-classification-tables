
import numpy as np

# ----------- Compute Depth Metrics ------------ #
def compute_depth_metrics(pred_depth, gt_depth, valid_mask):
    pred_depth = pred_depth[valid_mask]
    gt_depth = gt_depth[valid_mask]

    # Base errors
    abs_rel = np.mean(np.abs(pred_depth - gt_depth) / gt_depth)
    rmse = np.sqrt(np.mean((pred_depth - gt_depth) ** 2))
    mae = np.mean(np.abs(pred_depth - gt_depth))
    rmse_log = np.sqrt(np.mean((np.log(pred_depth + 1e-8) - np.log(gt_depth + 1e-8))**2))
    log10 = np.mean(np.abs(np.log10(pred_depth / gt_depth)))

    # Silog
    log_diff = np.log(pred_depth + 1e-8) - np.log(gt_depth + 1e-8)
    silog = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))

    # Threshold accuracies
    ratio = np.maximum(pred_depth / gt_depth, gt_depth / pred_depth)
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25 ** 2)
    delta3 = np.mean(ratio < 1.25 ** 3)

    return {
        "AbsRel": abs_rel,
        "RMSE": rmse,
        "MAE": mae,
        "RMSE_log": rmse_log,
        "Log10": log10,
        "Silog": silog,
        "δ1 (<1.25)": delta1,
        "δ2 (<1.25^2)": delta2,
        "δ3 (<1.25^3)": delta3
    }
