import numpy as np


def hubbard_discrete_transform(aux_config, time_step_size, uval):
    lda = (uval * time_step_size / 2) + np.log(1 + np.sqrt(1 - np.exp(-uval * time_step_size)))
    expterm = np.exp(lda * aux_config)
    return np.diag(expterm)


def hubbard_discrete_transform_inverse(aux_config, time_step_size, uval):
    lda = (uval * time_step_size / 2) + np.log(1 + np.sqrt(1 - np.exp(-uval * time_step_size)))
    expterm = np.exp(-1 * lda * aux_config)
    return np.diag(expterm)


def afm_measure(params):
    num_sites = params.G_up.shape[0]
    sublattice_vals = np.concat((np.repeat(1, int(num_sites / 2)), np.repeat(-1, int(num_sites / 2))))

    # i not equal to j case
    i = np.repeat(np.arange(num_sites), num_sites)
    j = np.tile(np.arange(num_sites), num_sites)
    same_mask = i == j
    i = i[~same_mask]
    j = j[~same_mask]
    final_result = 0.25 * np.sum((sublattice_vals[i] * sublattice_vals[j]) * (
        params.G_up[i, i] * params.G_up[j, j]
        + params.G_down[i, i] * params.G_down[j, j]
        - params.G_up[i, i] * params.G_down[j, j]
        - params.G_down[i, i] * params.G_up[j, j]
        - params.G_up[j, i] * params.G_up[i, j]
        - params.G_down[j, i] * params.G_down[i, j]
    ))

    # i=j case
    i = np.repeat(np.arange(num_sites), num_sites)
    j = np.tile(np.arange(num_sites), num_sites)
    same_mask = i == j
    i = i[same_mask]
    j = j[same_mask]
    final_result += 0.25 * np.sum((sublattice_vals[i] * sublattice_vals[j]) * (
            params.G_up[i, i] * params.G_up[j, j]
            + params.G_down[i, i] * params.G_down[j, j]
            - params.G_up[i, i] * params.G_down[j, j]
            - params.G_down[i, i] * params.G_up[j, j]
            - params.G_up[j, i] * params.G_up[i, j]
            - params.G_down[j, i] * params.G_down[i, j]
            + params.G_up[i, j] + params.G_down[i, j]  # Extra bit for i=j case
    ))

    final_result /= num_sites ** 2

    return final_result
