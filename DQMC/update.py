import numpy as np
from numba import njit


@njit
def get_rank_one_update_vecs_legacy(auxiliary_field, time_slice_size, interaction_strength,
                                    trial_auxiliary_field, trial_time_slice, trial_site):
    colvecUpdateU = np.zeros((auxiliary_field.shape[1], 1), dtype=np.float64)
    colvecUpdateU[trial_site, 0] = 1.0

    rowvecUpdateVT = np.zeros((1, auxiliary_field.shape[1]), dtype=np.float64)
    lmda = np.log(np.exp(0.5 * time_slice_size * interaction_strength) + np.sqrt(np.exp(time_slice_size * interaction_strength) - 1))
    rowvecUpdateVT[0, trial_site] = np.exp(lmda * (trial_auxiliary_field[trial_time_slice, trial_site]
                                                   - auxiliary_field[trial_time_slice, trial_site])) - 1.0

    return colvecUpdateU, rowvecUpdateVT


@njit
def get_rank_one_update_vecs(auxiliary_field, time_slice_size, interaction_strength,
                             trial_time_slice, trial_site):
    colvecUpdateU = np.zeros((auxiliary_field.shape[1], 1), dtype=np.float64)
    colvecUpdateU[trial_site, 0] = 1.0

    rowvecUpdateVT = np.zeros((1, auxiliary_field.shape[1]), dtype=np.float64)
    lmda = np.log(np.exp(0.5 * time_slice_size * interaction_strength) + np.sqrt(np.exp(time_slice_size * interaction_strength) - 1))
    rowvecUpdateVT[0, trial_site] = np.exp(lmda * (-1.0 * auxiliary_field[trial_time_slice, trial_site]
                                                   - auxiliary_field[trial_time_slice, trial_site])) - 1.0

    return colvecUpdateU, rowvecUpdateVT


@njit
def rank_one_update_G(arrG, colvecUpdateU, rowvecUpdateVT):
    numerator = (arrG @ colvecUpdateU) @ (rowvecUpdateVT @ (np.eye(arrG.shape[0]) - arrG))
    denominator = 1.0 + (rowvecUpdateVT @ (np.eye(arrG.shape[0]) - arrG) @ colvecUpdateU)[0, 0]
    if np.abs(denominator) < 0.001:
        print(denominator)
    return arrG - (numerator / denominator)


# @njit
# def metropolis_criteria_rank_one_update(arrG, colvecUpdateU, rowvecUpdateVT):
#     return 1.0 + (rowvecUpdateVT @ (np.eye(arrG.shape[0]) - arrG) @ colvecUpdateU)[0, 0]
@njit
def metropolis_criteria_rank_one_update(arrG, trial_ind, delta_val):
    return 1.0 + ((1 - arrG[trial_ind, trial_ind]) * delta_val)


@njit
def low_rank_update_G(arrG, cached_u, cached_v):
    return arrG + cached_u @ cached_v


@njit
def submatrix_update_form_new_gamma(prev_gamma, original_g, accepted_trial_sites, new_trial_site, new_delta_value):
    if np.any(accepted_trial_sites == -1):
        raise ValueError
    arr_w = -1.0 * original_g[accepted_trial_sites, new_trial_site]
    arr_q = -1.0 * original_g[new_trial_site, accepted_trial_sites]
    arr_sigma = 1.0 + (1 / new_delta_value) - original_g[new_trial_site, new_trial_site]

    new_gamma = np.zeros((prev_gamma.shape[0] + 1, prev_gamma.shape[1] + 1), dtype=np.float64)
    new_gamma[:-1, :-1] = prev_gamma
    new_gamma[:-1, -1] = arr_w
    new_gamma[-1, :-1] = arr_q
    new_gamma[-1, -1] = arr_sigma

    return new_gamma


@njit
def submatrix_update_invert_new_gamma(prev_gamma, prev_gamma_inv, original_g,
                                      accepted_trial_sites, new_trial_site, new_delta_value):
    if np.any(accepted_trial_sites == -1):
        raise ValueError
    arr_w = -1.0 * original_g[accepted_trial_sites, new_trial_site].reshape(-1, 1)
    arr_q = -1.0 * original_g[new_trial_site, accepted_trial_sites].reshape(1, -1)
    arr_sigma = 1.0 + (1 / new_delta_value) - original_g[new_trial_site, new_trial_site]

    new_gamma_matrix_inv = np.zeros((prev_gamma.shape[0] + 1, prev_gamma.shape[1] + 1),
                                    dtype=np.float64)

    inverse_val = arr_sigma - (arr_q @ prev_gamma_inv @ arr_w)[0, 0]

    new_gamma_matrix_inv[:prev_gamma.shape[0], :prev_gamma.shape[1]]\
        = (prev_gamma_inv + (prev_gamma_inv @ arr_w @ arr_q @ prev_gamma_inv)
           / inverse_val)

    new_gamma_matrix_inv[:-1, -1] = (-1.0 * (prev_gamma_inv @ arr_w) / inverse_val).reshape(-1)
    new_gamma_matrix_inv[-1, :-1] = (-1.0 * (arr_q @ prev_gamma_inv) / inverse_val).reshape(-1)
    new_gamma_matrix_inv[-1, -1] = 1 / inverse_val

    return new_gamma_matrix_inv


@njit
def metropolis_criteria_submatrix_update(prev_gamma_inv, original_g, accepted_trial_sites, new_trial_site, new_delta_value):
    if np.any(accepted_trial_sites == -1):
        raise ValueError
    sylvester_mat = (-1.0 * (original_g[new_trial_site, new_trial_site] +
                             original_g[new_trial_site, accepted_trial_sites]
                             @ prev_gamma_inv
                             @ original_g[accepted_trial_sites, new_trial_site]) + 1.0) * new_delta_value + 1.0

    return sylvester_mat


@njit
def submatrix_update_update_G(original_g, accepted_trial_sites, gamma_mat_inv, delta_values):
    if np.any(accepted_trial_sites == -1):
        raise ValueError
    arr_g = original_g[:, accepted_trial_sites] @ gamma_mat_inv @ original_g[accepted_trial_sites, :]
    arr_g += original_g
    # arr_g = arr_g @ (np.eye(original_g.shape[0]) -
    #                  np.eye(original_g.shape[0])[:, accepted_trial_sites]
    #                  @ np.diag(delta_values / (1 + delta_values))
    #                  @ np.eye(original_g.shape[0])[accepted_trial_sites, :])
    other_arr = np.ones(original_g.shape[0])
    other_arr[accepted_trial_sites] -= delta_values / (1 + delta_values)
    arr_g = arr_g * other_arr
    return arr_g

