import torch


def max_of_two(y_over_z, z_over_y):
    return torch.max(y_over_z, z_over_y)


def evaluate_error(pred_depth, gt_depth, depth_mask):

    error = {'loss': 0, 'MAE': 0, 'MAE_XYZ': 0, 'REL': 0, 'BAD1SCORE': 0, 'BAD2SCORE': 0, 'BAD5SCORE': 0, 'BAD10SCORE': 0}
    error['MAE_XYZ'] = torch.sum(torch.abs(pred_depth[depth_mask] - gt_depth[depth_mask]))
    pred_depth = pred_depth[:, 2, :, :]
    gt_depth = gt_depth[:, 2, :, :]
    depth_mask = depth_mask[:, 2, :, :]
    _pred_depth = pred_depth[depth_mask]
    _gt_depth = gt_depth[depth_mask]
    n_valid_element = _gt_depth.size(0)
    error['MAE_XYZ'] = error['MAE_XYZ'] / n_valid_element / 3
    if n_valid_element > 0:
        diff_mat = torch.abs(_gt_depth-_pred_depth)
        rel_mat = torch.div(diff_mat, _gt_depth)

        error['MAE'] = torch.sum(diff_mat)/n_valid_element
        error['REL'] = torch.sum(rel_mat)/n_valid_element

        error['BAD1SCORE'] = torch.sum(diff_mat > 1).numpy()/float(n_valid_element)
        error['BAD2SCORE'] = torch.sum(diff_mat > 2).numpy()/float(n_valid_element)
        error['BAD5SCORE'] = torch.sum(diff_mat > 5).numpy()/float(n_valid_element)
        error['BAD10SCORE'] = torch.sum(diff_mat > 10).numpy()/float(n_valid_element)

    return error


# avg the error
def avg_error(error_sum, error_cur, total_count, count):
    error_avg = {'loss': 0, 'MAE': 0, 'MAE_XYZ': 0, 'REL': 0, 'BAD1SCORE': 0, 'BAD2SCORE': 0, 'BAD5SCORE': 0, 'BAD10SCORE': 0}
    for item, value in error_cur.items():
        error_sum[item] += error_cur[item] * count
        error_avg[item] = error_sum[item]/float(total_count)
    return error_avg


def log_error(error, logger):
    for item, value in error.items():
        logger.info(f'{item} = {value}')