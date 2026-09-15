import time
from DQMC.update import *
from DQMC.stabilization import *
from numba import njit


# @njit
# def time_slice_fast_update_sweep(G, auxiliary_field, time_slice_size, interaction_strength, trial_time_slice):
#     for i in range(auxiliary_field.shape[1]):
#         trial_auxiliary_field = auxiliary_field.copy()
#         trial_auxiliary_field[trial_time_slice, i] = -1 * auxiliary_field[trial_time_slice, i]
#
#         colvecUpdateU, rowvecUpdateVT = get_rank_one_update_vecs(auxiliary_field, time_slice_size, interaction_strength,
#                                                                  trial_auxiliary_field, trial_time_slice, i)
#
#         acceptProbability = metropolis_criteria_rank_one_update(G, colvecUpdateU, rowvecUpdateVT)
#
#         if acceptProbability >= 1:
#             G = rank_one_update_G(G, colvecUpdateU, rowvecUpdateVT)
#             auxiliary_field = trial_auxiliary_field.copy()
#         else:
#             diceRoll = np.random.uniform(0, 1)
#             if diceRoll <= acceptProbability:
#                 G = rank_one_update_G(G, colvecUpdateU, rowvecUpdateVT)
#                 auxiliary_field = trial_auxiliary_field.copy()
#
#     return G, auxiliary_field
#
#
# def full_sweep(params, restablization_frequency, isForwardSweep, measurement_func, measurement_freq):
#     measurements = []
#     num_time_slices = params.auxiliary_field.shape[0]
#     if isForwardSweep:
#         for it in range(num_time_slices - 1, -1, -1):
#             params.G, params.auxiliary_field = time_slice_fast_update_sweep(params.G,
#                                                                             params.auxiliary_field,
#                                                                             params.time_slice_size,
#                                                                             params.interaction_strength,
#                                                                             trial_time_slice=it)
#             if ((it - num_time_slices + 2) % restablization_frequency) == 0:
#                 params.G = quick_stabilization_computation(params, it, isForwardSweep)
#             else:
#                 params.G = params.G @ params.expInteractionTermFunc(params.auxiliary_field[it, :],
#                                                                     params.time_slice_size,
#                                                                     params.interaction_strength) @ params.expKineticTerm
#                 params.G = params.expKineticInvTerm @ params.expInteractionInvTermFunc(params.auxiliary_field[it, :],
#                                                                                        params.time_slice_size,
#                                                                                        params.interaction_strength) @ params.G
#
#             if ((it - num_time_slices + 2) % measurement_freq) == 0:
#                 measurements.append(measurement_func(params))
#
#     else:
#         for it in range(num_time_slices):
#             params.G, params.auxiliary_field = time_slice_fast_update_sweep(params.G,
#                                                                             params.auxiliary_field,
#                                                                             params.time_slice_size,
#                                                                             params.interaction_strength,
#                                                                             trial_time_slice=it)
#             if ((it + 1) % restablization_frequency) == 0:
#                 params.G = quick_stabilization_computation(params, it, isForwardSweep)
#             else:
#                 params.G = params.G @ params.expKineticInvTerm @ params.expInteractionInvTermFunc(params.auxiliary_field[it, :],
#                                                                                                   params.time_slice_size,
#                                                                                                   params.interaction_strength)
#                 params.G = params.expInteractionTermFunc(params.auxiliary_field[it, :],
#                                                          params.time_slice_size,
#                                                          params.interaction_strength) @ params.expKineticTerm @ params.G
#
#             if ((it + 1) % measurement_freq) == 0:
#                 measurements.append(measurement_func(params))
#
#     return measurements


@njit
def time_slice_fast_update_sweep_spin_factorize(G_up, G_down, auxiliary_field, time_slice_size,
                                                interaction_strength, trial_time_slice):
    for i in range(auxiliary_field.shape[1]):
        trial_auxiliary_field = auxiliary_field.copy()
        trial_auxiliary_field[trial_time_slice, i] = -1 * auxiliary_field[trial_time_slice, i]

        colvecUpdateU_up, rowvecUpdateVT_up = get_rank_one_update_vecs_legacy(auxiliary_field, time_slice_size,
                                                                              interaction_strength,
                                                                              trial_auxiliary_field,
                                                                              trial_time_slice, i)
        colvecUpdateU_down, rowvecUpdateVT_down = get_rank_one_update_vecs_legacy(-1.0 * auxiliary_field, time_slice_size,
                                                                                  interaction_strength,
                                                                                  -1.0 * trial_auxiliary_field,
                                                                                  trial_time_slice, i)

        acceptProbability = (metropolis_criteria_rank_one_update(G_up, colvecUpdateU_up, rowvecUpdateVT_up)
                             * metropolis_criteria_rank_one_update(G_down, colvecUpdateU_down, rowvecUpdateVT_down))
        if acceptProbability < 0.0:
            print(trial_time_slice, acceptProbability)
            raise ValueError

        if acceptProbability >= 1:
            G_up = rank_one_update_G(G_up, colvecUpdateU_up, rowvecUpdateVT_up)
            G_down = rank_one_update_G(G_down, colvecUpdateU_down, rowvecUpdateVT_down)
            auxiliary_field = trial_auxiliary_field.copy()

        else:
            diceRoll = np.random.uniform(0, 1)
            if diceRoll <= acceptProbability:
                G_up = rank_one_update_G(G_up, colvecUpdateU_up, rowvecUpdateVT_up)
                G_down = rank_one_update_G(G_down, colvecUpdateU_down, rowvecUpdateVT_down)
                auxiliary_field = trial_auxiliary_field.copy()

    return G_up, G_down, auxiliary_field


@njit
def time_slice_fast_update_sweep_spin_factorize_delayed_update(G_up, G_down, auxiliary_field, time_slice_size,
                                                               interaction_strength, trial_time_slice,
                                                               delayed_update_freq):
    cached_col_vec_u_up = np.zeros((G_up.shape[0], delayed_update_freq), dtype=np.float64)
    cached_col_vec_u_down = np.zeros((G_down.shape[0], delayed_update_freq), dtype=np.float64)
    cached_row_vec_vt_up = np.zeros((delayed_update_freq, G_up.shape[0]), dtype=np.float64)
    cached_row_vec_vt_down = np.zeros((delayed_update_freq, G_down.shape[0]), dtype=np.float64)
    number_accepted_moves = 0
    on_fly_G_up = G_up.copy()
    on_fly_G_down = G_down.copy()
    ###
    # test_G_up = G_up.copy()
    # test_G_down = G_down.copy()
    ###
    for i in range(auxiliary_field.shape[1]):
        trial_auxiliary_field = auxiliary_field.copy()
        trial_auxiliary_field[trial_time_slice, i] = -1 * auxiliary_field[trial_time_slice, i]

        colvecUpdateU_up, rowvecUpdateVT_up = get_rank_one_update_vecs_legacy(auxiliary_field, time_slice_size,
                                                                              interaction_strength,
                                                                              trial_auxiliary_field,
                                                                              trial_time_slice, i)
        colvecUpdateU_down, rowvecUpdateVT_down = get_rank_one_update_vecs_legacy(-1.0 * auxiliary_field, time_slice_size,
                                                                                  interaction_strength,
                                                                                  -1.0 * trial_auxiliary_field,
                                                                                  trial_time_slice, i)

        acceptProbability = (metropolis_criteria_rank_one_update(on_fly_G_up, colvecUpdateU_up, rowvecUpdateVT_up)
                             * metropolis_criteria_rank_one_update(on_fly_G_down, colvecUpdateU_down, rowvecUpdateVT_down))
        ###
        # test_accept_prob = (metropolis_criteria_rank_one_update(test_G_up, colvecUpdateU_up, rowvecUpdateVT_up)
        #                     * metropolis_criteria_rank_one_update(test_G_down, colvecUpdateU_down, rowvecUpdateVT_down))
        # if np.abs(acceptProbability - test_accept_prob) > 10**(-14):
        #     print('Accept Probability Difference: ', np.abs(acceptProbability - test_accept_prob))
        ###
        if acceptProbability < 0.0:
            print(trial_time_slice, acceptProbability)
            raise ValueError

        elif acceptProbability >= 1 and number_accepted_moves + 1 != delayed_update_freq:
            cached_col_vec_u_up[:, number_accepted_moves] = on_fly_G_up[:, i]
            cached_col_vec_u_down[:, number_accepted_moves] = on_fly_G_down[:, i]
            cached_row_vec_vt_up[number_accepted_moves, :] = on_fly_G_up[i, :]
            cached_row_vec_vt_down[number_accepted_moves, :] = on_fly_G_down[i, :]
            cached_row_vec_vt_up[number_accepted_moves, i] -= 1
            cached_row_vec_vt_down[number_accepted_moves, i] -= 1
            cached_row_vec_vt_up[number_accepted_moves, :] *= ((rowvecUpdateVT_up @ colvecUpdateU_up)[0, 0]
                                                                   / (1 - (on_fly_G_up[i, i] - 1)
                                                                   * (rowvecUpdateVT_up @ colvecUpdateU_up)[0, 0]))
            cached_row_vec_vt_down[number_accepted_moves, :] *= ((rowvecUpdateVT_down @ colvecUpdateU_down)[0, 0]
                                                                 / (1 - (on_fly_G_down[i, i] - 1)
                                                                 * (rowvecUpdateVT_down @ colvecUpdateU_down)[0, 0]))

            if i != auxiliary_field.shape[1] - 1:
                on_fly_G_up[:, i + 1] = G_up[:, i + 1] + cached_col_vec_u_up @ cached_row_vec_vt_up[:, i + 1]
                on_fly_G_up[i + 1, :] = G_up[i + 1, :] + cached_col_vec_u_up[i + 1, :] @ cached_row_vec_vt_up
                on_fly_G_down[:, i + 1] = G_down[:, i + 1] + cached_col_vec_u_down @ cached_row_vec_vt_down[:, i + 1]
                on_fly_G_down[i + 1, :] = G_down[i + 1, :] + cached_col_vec_u_down[i + 1, :] @ cached_row_vec_vt_down

            auxiliary_field = trial_auxiliary_field.copy()
            number_accepted_moves += 1

            ###
            # test_G_up = rank_one_update_G(test_G_up, colvecUpdateU_up, rowvecUpdateVT_up)
            # test_G_down = rank_one_update_G(test_G_down, colvecUpdateU_down, rowvecUpdateVT_down)
            # other_test_G_up = low_rank_update_G(G_up, cached_col_vec_u_up, cached_row_vec_vt_up)
            # other_test_G_down = low_rank_update_G(G_down, cached_col_vec_u_down, cached_row_vec_vt_down)
            # if np.max(np.abs(test_G_up - other_test_G_up)) > 10**(-14):
            #     print('Max difference intermediate G_up: ', np.max(np.abs(test_G_up - other_test_G_up)))
            # if np.max(np.abs(test_G_down - other_test_G_down)) > 10 ** (-14):
            #     print('Max difference intermediate G_down: ', np.max(np.abs(test_G_down - other_test_G_down)))
            ###

        elif acceptProbability >= 1 and number_accepted_moves + 1 == delayed_update_freq:
            cached_col_vec_u_up[:, number_accepted_moves] = on_fly_G_up[:, i]
            cached_col_vec_u_down[:, number_accepted_moves] = on_fly_G_down[:, i]
            cached_row_vec_vt_up[number_accepted_moves, :] = on_fly_G_up[i, :]
            cached_row_vec_vt_down[number_accepted_moves, :] = on_fly_G_down[i, :]
            cached_row_vec_vt_up[number_accepted_moves, i] -= 1
            cached_row_vec_vt_down[number_accepted_moves, i] -= 1
            cached_row_vec_vt_up[number_accepted_moves, :] *= ((rowvecUpdateVT_up @ colvecUpdateU_up)[0, 0]
                                                                   / (1 - (on_fly_G_up[i, i] - 1)
                                                                   * (rowvecUpdateVT_up @ colvecUpdateU_up)[0, 0]))
            cached_row_vec_vt_down[number_accepted_moves, :] *= ((rowvecUpdateVT_down @ colvecUpdateU_down)[0, 0]
                                                                 / (1 - (on_fly_G_down[i, i] - 1)
                                                                 * (rowvecUpdateVT_down @ colvecUpdateU_down)[0, 0]))

            G_up = low_rank_update_G(G_up, cached_col_vec_u_up, cached_row_vec_vt_up)
            G_down = low_rank_update_G(G_down, cached_col_vec_u_down, cached_row_vec_vt_down)

            auxiliary_field = trial_auxiliary_field.copy()
            cached_col_vec_u_up = np.zeros((G_up.shape[0], delayed_update_freq), dtype=np.float64)
            cached_col_vec_u_down = np.zeros((G_down.shape[0], delayed_update_freq), dtype=np.float64)
            cached_row_vec_vt_up = np.zeros((delayed_update_freq, G_up.shape[0]), dtype=np.float64)
            cached_row_vec_vt_down = np.zeros((delayed_update_freq, G_down.shape[0]), dtype=np.float64)
            number_accepted_moves = 0
            on_fly_G_up = G_up.copy()
            on_fly_G_down = G_down.copy()

            ###
            # test_G_up = rank_one_update_G(test_G_up, colvecUpdateU_up, rowvecUpdateVT_up)
            # test_G_down = rank_one_update_G(test_G_down, colvecUpdateU_down, rowvecUpdateVT_down)
            #
            # if np.max(np.abs(G_up - test_G_up)) > 10**(-14):
            #     print('Max G_up difference: ', np.max(np.abs(G_up - test_G_up)))
            # if np.max(np.abs(G_down - test_G_down)) > 10**(-14):
            #     print('Max G_down difference: ', np.max(np.abs(G_down - test_G_down)))
            ###

        else:
            diceRoll = np.random.uniform(0, 1)
            if (diceRoll <= acceptProbability) and (number_accepted_moves + 1 != delayed_update_freq):
                cached_col_vec_u_up[:, number_accepted_moves] = on_fly_G_up[:, i]
                cached_col_vec_u_down[:, number_accepted_moves] = on_fly_G_down[:, i]
                cached_row_vec_vt_up[number_accepted_moves, :] = on_fly_G_up[i, :]
                cached_row_vec_vt_down[number_accepted_moves, :] = on_fly_G_down[i, :]
                cached_row_vec_vt_up[number_accepted_moves, i] -= 1
                cached_row_vec_vt_down[number_accepted_moves, i] -= 1
                cached_row_vec_vt_up[number_accepted_moves, :] *= ((rowvecUpdateVT_up @ colvecUpdateU_up)[0, 0]
                                                                   / (1 - (on_fly_G_up[i, i] - 1)
                                                                   * (rowvecUpdateVT_up @ colvecUpdateU_up)[0, 0]))
                cached_row_vec_vt_down[number_accepted_moves, :] *= ((rowvecUpdateVT_down @ colvecUpdateU_down)[0, 0]
                                                                     / (1 - (on_fly_G_down[i, i] - 1)
                                                                     * (rowvecUpdateVT_down @ colvecUpdateU_down)[0, 0]))

                if i != auxiliary_field.shape[1] - 1:
                    on_fly_G_up[:, i + 1] = G_up[:, i + 1] + cached_col_vec_u_up @ cached_row_vec_vt_up[:, i + 1]
                    on_fly_G_up[i + 1, :] = G_up[i + 1, :] + cached_col_vec_u_up[i + 1, :] @ cached_row_vec_vt_up
                    on_fly_G_down[:, i + 1] = G_down[:, i + 1] + cached_col_vec_u_down @ cached_row_vec_vt_down[:, i + 1]
                    on_fly_G_down[i + 1, :] = G_down[i + 1, :] + cached_col_vec_u_down[i + 1, :] @ cached_row_vec_vt_down

                auxiliary_field = trial_auxiliary_field.copy()
                number_accepted_moves += 1

                ###
                # test_G_up = rank_one_update_G(test_G_up, colvecUpdateU_up, rowvecUpdateVT_up)
                # test_G_down = rank_one_update_G(test_G_down, colvecUpdateU_down, rowvecUpdateVT_down)
                # other_test_G_up = low_rank_update_G(G_up, cached_col_vec_u_up, cached_row_vec_vt_up)
                # other_test_G_down = low_rank_update_G(G_down, cached_col_vec_u_down, cached_row_vec_vt_down)
                # if np.max(np.abs(test_G_up - other_test_G_up)) > 10 ** (-14):
                #     print('Max difference intermediate G_up: ', np.max(np.abs(test_G_up - other_test_G_up)))
                # if np.max(np.abs(test_G_down - other_test_G_down)) > 10 ** (-14):
                #     print('Max difference intermediate G_down: ', np.max(np.abs(test_G_down - other_test_G_down)))
                ###

            elif (diceRoll <= acceptProbability) and (number_accepted_moves + 1 == delayed_update_freq):
                cached_col_vec_u_up[:, number_accepted_moves] = on_fly_G_up[:, i]
                cached_col_vec_u_down[:, number_accepted_moves] = on_fly_G_down[:, i]
                cached_row_vec_vt_up[number_accepted_moves, :] = on_fly_G_up[i, :]
                cached_row_vec_vt_down[number_accepted_moves, :] = on_fly_G_down[i, :]
                cached_row_vec_vt_up[number_accepted_moves, i] -= 1
                cached_row_vec_vt_down[number_accepted_moves, i] -= 1
                cached_row_vec_vt_up[number_accepted_moves, :] *= ((rowvecUpdateVT_up @ colvecUpdateU_up)[0, 0]
                                                                   / (1 - (on_fly_G_up[i, i] - 1)
                                                                   * (rowvecUpdateVT_up @ colvecUpdateU_up)[0, 0]))
                cached_row_vec_vt_down[number_accepted_moves, :] *= ((rowvecUpdateVT_down @ colvecUpdateU_down)[0, 0]
                                                                     / (1 - (on_fly_G_down[i, i] - 1)
                                                                     * (rowvecUpdateVT_down @ colvecUpdateU_down)[0, 0]))

                G_up = low_rank_update_G(G_up, cached_col_vec_u_up, cached_row_vec_vt_up)
                G_down = low_rank_update_G(G_down, cached_col_vec_u_down, cached_row_vec_vt_down)

                auxiliary_field = trial_auxiliary_field.copy()
                cached_col_vec_u_up = np.zeros((G_up.shape[0], delayed_update_freq), dtype=np.float64)
                cached_col_vec_u_down = np.zeros((G_down.shape[0], delayed_update_freq), dtype=np.float64)
                cached_row_vec_vt_up = np.zeros((delayed_update_freq, G_up.shape[0]), dtype=np.float64)
                cached_row_vec_vt_down = np.zeros((delayed_update_freq, G_down.shape[0]), dtype=np.float64)
                number_accepted_moves = 0
                on_fly_G_up = G_up.copy()
                on_fly_G_down = G_down.copy()

                ###
                # test_G_up = rank_one_update_G(test_G_up, colvecUpdateU_up, rowvecUpdateVT_up)
                # test_G_down = rank_one_update_G(test_G_down, colvecUpdateU_down, rowvecUpdateVT_down)
                #
                # if np.max(np.abs(G_up - test_G_up)) > 10 ** (-14):
                #     print('Max G_up difference: ', np.max(np.abs(G_up - test_G_up)))
                # if np.max(np.abs(G_down - test_G_down)) > 10 ** (-14):
                #     print('Max G_down difference: ', np.max(np.abs(G_down - test_G_down)))
                ###

            else:
                if i != auxiliary_field.shape[1] - 1:
                    on_fly_G_up[:, i + 1] = G_up[:, i + 1] + cached_col_vec_u_up @ cached_row_vec_vt_up[:, i + 1]
                    on_fly_G_up[i + 1, :] = G_up[i + 1, :] + cached_col_vec_u_up[i + 1, :] @ cached_row_vec_vt_up
                    on_fly_G_down[:, i + 1] = G_down[:, i + 1] + cached_col_vec_u_down @ cached_row_vec_vt_down[:, i + 1]
                    on_fly_G_down[i + 1, :] = G_down[i + 1, :] + cached_col_vec_u_down[i + 1, :] @ cached_row_vec_vt_down

    if number_accepted_moves != 0:
        G_up = low_rank_update_G(G_up, cached_col_vec_u_up, cached_row_vec_vt_up)
        G_down = low_rank_update_G(G_down, cached_col_vec_u_down, cached_row_vec_vt_down)

    return G_up, G_down, auxiliary_field


@njit
def time_slice_fast_update_sweep_spin_factorize_submatrix_update(G_up, G_down, auxiliary_field, time_slice_size,
                                                                 interaction_strength, trial_time_slice,
                                                                 flush_update_freq):
    number_accepted_moves = 0
    accepted_trial_sites = np.full(flush_update_freq, -1, dtype=np.int32)
    delta_values_up = np.zeros(flush_update_freq, dtype=np.float64)
    delta_values_down = np.zeros(flush_update_freq, dtype=np.float64)
    gamma_up = None
    prev_gamma_inv_up = None
    gamma_down = None
    prev_gamma_inv_down = None
    ###
    # test_G_up = G_up.copy()
    # test_G_down = G_down.copy()
    ###

    colvecUpdateU_up = np.zeros((auxiliary_field.shape[1], 1), dtype=np.float64)
    colvecUpdateU_down = np.zeros((auxiliary_field.shape[1], 1), dtype=np.float64)
    rowvecUpdateVT_up = np.zeros((1, auxiliary_field.shape[1]), dtype=np.float64)
    rowvecUpdateVT_down = np.zeros((1, auxiliary_field.shape[1]), dtype=np.float64)
    for i in range(auxiliary_field.shape[1]):
        # trial_auxiliary_field = auxiliary_field.copy()
        # trial_auxiliary_field[trial_time_slice, i] = -1 * auxiliary_field[trial_time_slice, i]
        #
        # colvecUpdateU_up, rowvecUpdateVT_up = get_rank_one_update_vecs(auxiliary_field, time_slice_size,
        #                                                                interaction_strength, trial_auxiliary_field,
        #                                                                trial_time_slice, i)
        # colvecUpdateU_down, rowvecUpdateVT_down = get_rank_one_update_vecs(-1.0 * auxiliary_field, time_slice_size,
        #                                                                    interaction_strength,
        #                                                                    -1.0 * trial_auxiliary_field,
        #                                                                    trial_time_slice, i)

        # colvecUpdateU_up, rowvecUpdateVT_up = get_rank_one_update_vecs(auxiliary_field, time_slice_size,
        #                                                                interaction_strength,
        #                                                                trial_time_slice, i)
        # colvecUpdateU_down, rowvecUpdateVT_down = get_rank_one_update_vecs(-1.0 * auxiliary_field, time_slice_size,
        #                                                                    interaction_strength,
        #                                                                    trial_time_slice, i)

        colvecUpdateU_up[:, :] = 0.0
        colvecUpdateU_up[i, 0] = 1.0

        rowvecUpdateVT_up[:, :] = 0.0
        rowvecUpdateVT_up[0, i] = np.exp(np.log(np.exp(0.5 * time_slice_size * interaction_strength)
                                                + np.sqrt(np.exp(time_slice_size * interaction_strength) - 1))
                                         * (-1.0 * auxiliary_field[trial_time_slice, i]
                                            - auxiliary_field[trial_time_slice, i])) - 1.0

        colvecUpdateU_down[:, :] = 0.0
        colvecUpdateU_down[i, 0] = 1.0

        rowvecUpdateVT_down[:, :] = 0.0
        rowvecUpdateVT_down[0, i] = np.exp(np.log(np.exp(0.5 * time_slice_size * interaction_strength)
                                                  + np.sqrt(np.exp(time_slice_size * interaction_strength) - 1))
                                           * (-1.0 * (-1.0 * auxiliary_field[trial_time_slice, i])
                                              - (-1.0 * auxiliary_field[trial_time_slice, i]))) - 1.0

        if gamma_up is None:
            # acceptProbability = (metropolis_criteria_rank_one_update(G_up, colvecUpdateU_up, rowvecUpdateVT_up)
            #                      * metropolis_criteria_rank_one_update(G_down, colvecUpdateU_down, rowvecUpdateVT_down))

            acceptProbability = (metropolis_criteria_rank_one_update(G_up, i, rowvecUpdateVT_up[0, i])
                                 * metropolis_criteria_rank_one_update(G_down, i, rowvecUpdateVT_down[0, i]))
        else:
            acceptProbability = (
                    metropolis_criteria_submatrix_update(prev_gamma_inv_up,
                                                         G_up,
                                                         accepted_trial_sites[:number_accepted_moves],
                                                         i,
                                                         rowvecUpdateVT_up[0, i])
                    * metropolis_criteria_submatrix_update(prev_gamma_inv_down,
                                                           G_down,
                                                           accepted_trial_sites[:number_accepted_moves],
                                                           i,
                                                           rowvecUpdateVT_down[0, i])
            )

        ###
        # test_accept_prob = (metropolis_criteria_rank_one_update(test_G_up, colvecUpdateU_up, rowvecUpdateVT_up)
        #                     * metropolis_criteria_rank_one_update(test_G_down, colvecUpdateU_down, rowvecUpdateVT_down))
        # if np.abs(acceptProbability - test_accept_prob) > 10**(-14):
        #     print('Accept Probability Difference: ', np.abs(acceptProbability - test_accept_prob))
        ###
        if acceptProbability < 0.0:
            print(trial_time_slice, acceptProbability)
            raise ValueError

        elif acceptProbability >= 1 and number_accepted_moves + 1 != flush_update_freq:

            # auxiliary_field = trial_auxiliary_field.copy()
            auxiliary_field[trial_time_slice, i] = -1.0 * auxiliary_field[trial_time_slice, i]
            accepted_trial_sites[number_accepted_moves] = i
            delta_values_up[number_accepted_moves] = rowvecUpdateVT_up[0, i]
            delta_values_down[number_accepted_moves] = rowvecUpdateVT_down[0, i]

            if number_accepted_moves == 0:
                gamma_up = np.array([1 + (1 / rowvecUpdateVT_up[0, i]) - G_up[i, i]]).reshape(1, 1)
                gamma_down = np.array([1 + (1 / rowvecUpdateVT_down[0, i]) - G_down[i, i]]).reshape(1, 1)
                prev_gamma_inv_up = np.array([1 / (1 + (1 / rowvecUpdateVT_up[0, i]) - G_up[i, i])]).reshape(1, 1)
                prev_gamma_inv_down = np.array([1 / (1 + (1 / rowvecUpdateVT_down[0, i]) - G_down[i, i])]).reshape(1, 1)
            else:
                prev_gamma_inv_up = submatrix_update_invert_new_gamma(gamma_up,
                                                                      prev_gamma_inv_up,
                                                                      G_up,
                                                                      accepted_trial_sites[:number_accepted_moves],
                                                                      i,
                                                                      delta_values_up[number_accepted_moves])
                prev_gamma_inv_down = submatrix_update_invert_new_gamma(gamma_down,
                                                                        prev_gamma_inv_down,
                                                                        G_down,
                                                                        accepted_trial_sites[:number_accepted_moves],
                                                                        i,
                                                                        delta_values_down[number_accepted_moves])
                gamma_up = submatrix_update_form_new_gamma(gamma_up,
                                                           G_up,
                                                           accepted_trial_sites[:number_accepted_moves],
                                                           i,
                                                           delta_values_up[number_accepted_moves])
                gamma_down = submatrix_update_form_new_gamma(gamma_down,
                                                             G_down,
                                                             accepted_trial_sites[:number_accepted_moves],
                                                             i,
                                                             delta_values_down[number_accepted_moves])
            number_accepted_moves += 1

            ###
            # test_G_up = rank_one_update_G(test_G_up, colvecUpdateU_up, rowvecUpdateVT_up)
            # test_G_down = rank_one_update_G(test_G_down, colvecUpdateU_down, rowvecUpdateVT_down)
            # other_test_G_up = submatrix_update_update_G(G_up, accepted_trial_sites[:number_accepted_moves], prev_gamma_inv_up, delta_values_up[:number_accepted_moves])
            # other_test_G_down = submatrix_update_update_G(G_down, accepted_trial_sites[:number_accepted_moves], prev_gamma_inv_down, delta_values_down[:number_accepted_moves])
            # if np.max(np.abs(test_G_up - other_test_G_up)) > 10**(-14):
            #     print('Max difference intermediate G_up: ', np.max(np.abs(test_G_up - other_test_G_up)))
            # if np.max(np.abs(test_G_down - other_test_G_down)) > 10 ** (-14):
            #     print('Max difference intermediate G_down: ', np.max(np.abs(test_G_down - other_test_G_down)))
            ###

        elif acceptProbability >= 1 and number_accepted_moves + 1 == flush_update_freq:

            # auxiliary_field = trial_auxiliary_field.copy()
            auxiliary_field[trial_time_slice, i] = -1.0 * auxiliary_field[trial_time_slice, i]
            accepted_trial_sites[number_accepted_moves] = i
            delta_values_up[number_accepted_moves] = rowvecUpdateVT_up[0, i]
            delta_values_down[number_accepted_moves] = rowvecUpdateVT_down[0, i]

            if number_accepted_moves == 0:
                prev_gamma_inv_up = np.array([1 / (1 + (1 / rowvecUpdateVT_up[0, i]) - G_up[i, i])]).reshape(1, 1)
                prev_gamma_inv_down = np.array([1 / (1 + (1 / rowvecUpdateVT_down[0, i]) - G_down[i, i])]).reshape(1, 1)
            else:
                prev_gamma_inv_up = submatrix_update_invert_new_gamma(gamma_up,
                                                                      prev_gamma_inv_up,
                                                                      G_up,
                                                                      accepted_trial_sites[:number_accepted_moves],
                                                                      i,
                                                                      delta_values_up[number_accepted_moves])
                prev_gamma_inv_down = submatrix_update_invert_new_gamma(gamma_down,
                                                                        prev_gamma_inv_down,
                                                                        G_down,
                                                                        accepted_trial_sites[:number_accepted_moves],
                                                                        i,
                                                                        delta_values_down[number_accepted_moves])

            G_up = submatrix_update_update_G(G_up, accepted_trial_sites, prev_gamma_inv_up, delta_values_up)
            G_down = submatrix_update_update_G(G_down, accepted_trial_sites, prev_gamma_inv_down, delta_values_down)

            number_accepted_moves = 0
            accepted_trial_sites = np.full(flush_update_freq, -1, dtype=np.int32)
            delta_values_up = np.zeros(flush_update_freq, dtype=np.float64)
            delta_values_down = np.zeros(flush_update_freq, dtype=np.float64)
            gamma_up = None
            prev_gamma_inv_up = None
            gamma_down = None
            prev_gamma_inv_down = None

            ###
            # test_G_up = rank_one_update_G(test_G_up, colvecUpdateU_up, rowvecUpdateVT_up)
            # test_G_down = rank_one_update_G(test_G_down, colvecUpdateU_down, rowvecUpdateVT_down)
            #
            # if np.max(np.abs(G_up - test_G_up)) > 10**(-14):
            #     print('Max G_up difference: ', np.max(np.abs(G_up - test_G_up)))
            # if np.max(np.abs(G_down - test_G_down)) > 10**(-14):
            #     print('Max G_down difference: ', np.max(np.abs(G_down - test_G_down)))
            ###

        else:
            diceRoll = np.random.uniform(0, 1)
            if (diceRoll <= acceptProbability) and (number_accepted_moves + 1 != flush_update_freq):
                # auxiliary_field = trial_auxiliary_field.copy()
                auxiliary_field[trial_time_slice, i] = -1.0 * auxiliary_field[trial_time_slice, i]
                accepted_trial_sites[number_accepted_moves] = i
                delta_values_up[number_accepted_moves] = rowvecUpdateVT_up[0, i]
                delta_values_down[number_accepted_moves] = rowvecUpdateVT_down[0, i]

                if number_accepted_moves == 0:
                    gamma_up = np.array([1 + (1 / rowvecUpdateVT_up[0, i]) - G_up[i, i]]).reshape(1, 1)
                    gamma_down = np.array([1 + (1 / rowvecUpdateVT_down[0, i]) - G_down[i, i]]).reshape(1, 1)
                    prev_gamma_inv_up = np.array([1 / (1 + (1 / rowvecUpdateVT_up[0, i]) - G_up[i, i])]).reshape(1, 1)
                    prev_gamma_inv_down = np.array([1 / (1 + (1 / rowvecUpdateVT_down[0, i]) - G_down[i, i])]).reshape(1, 1)
                else:
                    prev_gamma_inv_up = submatrix_update_invert_new_gamma(gamma_up,
                                                                          prev_gamma_inv_up,
                                                                          G_up,
                                                                          accepted_trial_sites[:number_accepted_moves],
                                                                          i,
                                                                          delta_values_up[number_accepted_moves])
                    prev_gamma_inv_down = submatrix_update_invert_new_gamma(gamma_down,
                                                                            prev_gamma_inv_down,
                                                                            G_down,
                                                                            accepted_trial_sites[:number_accepted_moves],
                                                                            i,
                                                                            delta_values_down[number_accepted_moves])
                    gamma_up = submatrix_update_form_new_gamma(gamma_up,
                                                               G_up,
                                                               accepted_trial_sites[:number_accepted_moves],
                                                               i,
                                                               delta_values_up[number_accepted_moves])
                    gamma_down = submatrix_update_form_new_gamma(gamma_down,
                                                                 G_down,
                                                                 accepted_trial_sites[:number_accepted_moves],
                                                                 i,
                                                                 delta_values_down[number_accepted_moves])
                number_accepted_moves += 1

                ###
                # test_G_up = rank_one_update_G(test_G_up, colvecUpdateU_up, rowvecUpdateVT_up)
                # test_G_down = rank_one_update_G(test_G_down, colvecUpdateU_down, rowvecUpdateVT_down)
                # other_test_G_up = submatrix_update_update_G(G_up, accepted_trial_sites[:number_accepted_moves],
                #                                             prev_gamma_inv_up, delta_values_up[:number_accepted_moves])
                # other_test_G_down = submatrix_update_update_G(G_down, accepted_trial_sites[:number_accepted_moves],
                #                                               prev_gamma_inv_down, delta_values_down[:number_accepted_moves])
                # if np.max(np.abs(test_G_up - other_test_G_up)) > 10 ** (-14):
                #     print('Max difference intermediate G_up: ', np.max(np.abs(test_G_up - other_test_G_up)))
                # if np.max(np.abs(test_G_down - other_test_G_down)) > 10 ** (-14):
                #     print('Max difference intermediate G_down: ', np.max(np.abs(test_G_down - other_test_G_down)))
                ###

            elif (diceRoll <= acceptProbability) and (number_accepted_moves + 1 == flush_update_freq):
                # auxiliary_field = trial_auxiliary_field.copy()
                auxiliary_field[trial_time_slice, i] = -1.0 * auxiliary_field[trial_time_slice, i]
                accepted_trial_sites[number_accepted_moves] = i
                delta_values_up[number_accepted_moves] = rowvecUpdateVT_up[0, i]
                delta_values_down[number_accepted_moves] = rowvecUpdateVT_down[0, i]

                if number_accepted_moves == 0:
                    prev_gamma_inv_up = np.array([1 / (1 + (1 / rowvecUpdateVT_up[0, i]) - G_up[i, i])]).reshape(1, 1)
                    prev_gamma_inv_down = np.array([1 / (1 + (1 / rowvecUpdateVT_down[0, i]) - G_down[i, i])]).reshape(1, 1)
                else:
                    prev_gamma_inv_up = submatrix_update_invert_new_gamma(gamma_up,
                                                                          prev_gamma_inv_up,
                                                                          G_up,
                                                                          accepted_trial_sites[:number_accepted_moves],
                                                                          i,
                                                                          delta_values_up[number_accepted_moves])
                    prev_gamma_inv_down = submatrix_update_invert_new_gamma(gamma_down,
                                                                            prev_gamma_inv_down,
                                                                            G_down,
                                                                            accepted_trial_sites[:number_accepted_moves],
                                                                            i,
                                                                            delta_values_down[number_accepted_moves])

                G_up = submatrix_update_update_G(G_up, accepted_trial_sites, prev_gamma_inv_up, delta_values_up)
                G_down = submatrix_update_update_G(G_down, accepted_trial_sites, prev_gamma_inv_down, delta_values_down)

                number_accepted_moves = 0
                accepted_trial_sites = np.full(flush_update_freq, -1, dtype=np.int32)
                delta_values_up = np.zeros(flush_update_freq, dtype=np.float64)
                delta_values_down = np.zeros(flush_update_freq, dtype=np.float64)
                gamma_up = None
                prev_gamma_inv_up = None
                gamma_down = None
                prev_gamma_inv_down = None

                ###
                # test_G_up = rank_one_update_G(test_G_up, colvecUpdateU_up, rowvecUpdateVT_up)
                # test_G_down = rank_one_update_G(test_G_down, colvecUpdateU_down, rowvecUpdateVT_down)
                #
                # if np.max(np.abs(G_up - test_G_up)) > 10**(-14):
                #     print('Max G_up difference: ', np.max(np.abs(G_up - test_G_up)))
                # if np.max(np.abs(G_down - test_G_down)) > 10**(-14):
                #     print('Max G_down difference: ', np.max(np.abs(G_down - test_G_down)))
                ###

    if number_accepted_moves != 0:
        G_up = submatrix_update_update_G(G_up, accepted_trial_sites[:number_accepted_moves],
                                         prev_gamma_inv_up, delta_values_up[:number_accepted_moves])
        G_down = submatrix_update_update_G(G_down, accepted_trial_sites[:number_accepted_moves],
                                           prev_gamma_inv_down, delta_values_down[:number_accepted_moves])

    return G_up, G_down, auxiliary_field


def full_sweep_spin_factorize(params, restablization_frequency, measurement_func, measurement_freq):
    measurements = []
    num_time_slices = params.auxiliary_field.shape[0]

    for it in range(num_time_slices - 1, -1, -1):
        params.G_up, params.G_down, params.auxiliary_field = time_slice_fast_update_sweep_spin_factorize(params.G_up,
                                                                                                         params.G_down,
                                                                                                         params.auxiliary_field,
                                                                                                         params.time_slice_size,
                                                                                                         params.interaction_strength,
                                                                                                         trial_time_slice=-1)

        params.G_up = params.G_up @ params.expInteractionTermFunc(params.auxiliary_field[-1, :],
                                                                  params.time_slice_size,
                                                                  params.interaction_strength) @ params.expKineticTerm
        params.G_up = params.expKineticInvTerm @ params.expInteractionInvTermFunc(params.auxiliary_field[-1, :],
                                                                                  params.time_slice_size,
                                                                                  params.interaction_strength) @ params.G_up

        params.G_down = params.G_down @ params.expInteractionTermFunc(-1.0 * params.auxiliary_field[-1, :],
                                                                      params.time_slice_size,
                                                                      params.interaction_strength) @ params.expKineticTerm
        params.G_down = params.expKineticInvTerm @ params.expInteractionInvTermFunc(-1.0 * params.auxiliary_field[-1, :],
                                                                                    params.time_slice_size,
                                                                                    params.interaction_strength) @ params.G_down

        params.auxiliary_field = np.roll(params.auxiliary_field, shift=1, axis=0)

        if (num_time_slices - it) % restablization_frequency == 0:
            propagators_up = form_propagators(params.auxiliary_field,
                                              params.expKineticTerm,
                                              params.expInteractionTermFunc,
                                              params.time_slice_size,
                                              params.interaction_strength)
            params.G_up = full_stabilization_computation(propagators_up)

            propagators_down = form_propagators(-1.0 * params.auxiliary_field,
                                                params.expKineticTerm,
                                                params.expInteractionTermFunc,
                                                params.time_slice_size,
                                                params.interaction_strength)
            params.G_down = full_stabilization_computation(propagators_down)

        if ((it - num_time_slices + 2) % measurement_freq) == 0:
            measurements.append(measurement_func(params))

    return measurements


def full_sweep_spin_factorize_with_checkpoints(params, restabilization_frequency, measurement_func, measurement_freq,
                                               do_delayed_update=False, delayed_update_freq=None,
                                               do_submatrix_update=False, flush_update_freq=None):
    measurements = []
    num_time_slices = params.auxiliary_field.shape[0]

    for it in range(num_time_slices - 1, -1, -1):
        # st = time.time()
        if do_submatrix_update:
            params.G_up, params.G_down, params.auxiliary_field = time_slice_fast_update_sweep_spin_factorize_submatrix_update(
                params.G_up,
                params.G_down,
                params.auxiliary_field,
                params.time_slice_size,
                params.interaction_strength,
                trial_time_slice=-1,
                flush_update_freq=flush_update_freq
            )
        elif do_delayed_update:
            params.G_up, params.G_down, params.auxiliary_field = time_slice_fast_update_sweep_spin_factorize_delayed_update(
                params.G_up,
                params.G_down,
                params.auxiliary_field,
                params.time_slice_size,
                params.interaction_strength,
                trial_time_slice=-1,
                delayed_update_freq=delayed_update_freq
            )
        else:
            params.G_up, params.G_down, params.auxiliary_field = time_slice_fast_update_sweep_spin_factorize(params.G_up,
                                                                                                             params.G_down,
                                                                                                             params.auxiliary_field,
                                                                                                             params.time_slice_size,
                                                                                                             params.interaction_strength,
                                                                                                             trial_time_slice=-1)

        # print(it, time.time() - st)

        params.G_up = params.G_up @ params.expInteractionTermFunc(params.auxiliary_field[-1, :],
                                                                  params.time_slice_size,
                                                                  params.interaction_strength) @ params.expKineticTerm
        params.G_up = params.expKineticInvTerm @ params.expInteractionInvTermFunc(params.auxiliary_field[-1, :],
                                                                                  params.time_slice_size,
                                                                                  params.interaction_strength) @ params.G_up

        params.G_down = params.G_down @ params.expInteractionTermFunc(-1.0 * params.auxiliary_field[-1, :],
                                                                      params.time_slice_size,
                                                                      params.interaction_strength) @ params.expKineticTerm
        params.G_down = params.expKineticInvTerm @ params.expInteractionInvTermFunc(-1.0 * params.auxiliary_field[-1, :],
                                                                                    params.time_slice_size,
                                                                                    params.interaction_strength) @ params.G_down

        params.auxiliary_field = np.roll(params.auxiliary_field, shift=1, axis=0)

        if (num_time_slices % restabilization_frequency) != 0:
            restabilization_cond = ((num_time_slices - it) % restabilization_frequency == 0)
        else:
            restabilization_cond = ((num_time_slices - it) % restabilization_frequency == 0) and (it != 0)

        if restabilization_cond:

            ###
            # propagators_up = form_propagators(params.auxiliary_field,
            #                                   params.expKineticTerm,
            #                                   params.expInteractionTermFunc,
            #                                   params.time_slice_size,
            #                                   params.interaction_strength)
            # original_G_up = full_stabilization_computation(propagators_up)
            #
            # propagators_down = form_propagators(-1.0 * params.auxiliary_field,
            #                                     params.expKineticTerm,
            #                                     params.expInteractionTermFunc,
            #                                     params.time_slice_size,
            #                                     params.interaction_strength)
            # original_G_down = full_stabilization_computation(propagators_down)
            #
            # print('hi')
            ###

            # relevant_right_Q = params.right_checkpoints_up.Q[-(num_time_slices - it) // restabilization_frequency]
            # relevant_right_D = params.right_checkpoints_up.D[-(num_time_slices - it) // restabilization_frequency]
            # relevant_right_T = params.right_checkpoints_up.T[-(num_time_slices - it) // restabilization_frequency]
            relevant_right_Q = params.right_checkpoints_up.Q[(num_time_slices - it) // restabilization_frequency - 1]
            relevant_right_D = params.right_checkpoints_up.D[(num_time_slices - it) // restabilization_frequency - 1]
            relevant_right_T = params.right_checkpoints_up.T[(num_time_slices - it) // restabilization_frequency - 1]
            # relevant_left_Q = params.left_checkpoints_up.Q[(num_time_slices - it) // restabilization_frequency - 1]
            # relevant_left_D = params.left_checkpoints_up.D[(num_time_slices - it) // restabilization_frequency - 1]
            # relevant_left_T = params.left_checkpoints_up.T[(num_time_slices - it) // restabilization_frequency - 1]
            relevant_left_Q = params.left_checkpoints_up.Q[0]
            relevant_left_D = params.left_checkpoints_up.D[0]
            relevant_left_T = params.left_checkpoints_up.T[0]

            all_cycled_propagator = form_propagators(params.auxiliary_field[:restabilization_frequency, :],
                                                     params.expKineticTerm,
                                                     params.expInteractionTermFunc,
                                                     params.time_slice_size,
                                                     params.interaction_strength)
            cycled_propagator = np.eye(all_cycled_propagator.shape[1], all_cycled_propagator.shape[2])
            for i in range(all_cycled_propagator.shape[0]):
                cycled_propagator = all_cycled_propagator[i] @ cycled_propagator

            to_decompose = relevant_left_D.reshape(-1, 1) * relevant_left_Q @ cycled_propagator
            new_left_Q, new_left_D, new_left_T = graded_TDQ_decomposition(to_decompose)
            new_left_T = relevant_left_T @ new_left_T

            # if ((num_time_slices - it) // restabilization_frequency) != params.left_checkpoints_up.Q.shape[0]:
            #     params.update_checkpoints((num_time_slices - it) // restabilization_frequency,
            #                               new_left_Q,
            #                               new_left_D,
            #                               new_left_T,
            #                               side='left', spin='up')
            params.update_checkpoints(0,
                                      new_left_Q,
                                      new_left_D,
                                      new_left_T,
                                      side='left', spin='up')

            params.G_up = stabilization_from_checkpoints(relevant_right_Q,
                                                         relevant_right_D,
                                                         relevant_right_T,
                                                         new_left_T, new_left_D, new_left_Q)

            # relevant_right_Q = params.right_checkpoints_down.Q[-(num_time_slices - it) // restabilization_frequency]
            # relevant_right_D = params.right_checkpoints_down.D[-(num_time_slices - it) // restabilization_frequency]
            # relevant_right_T = params.right_checkpoints_down.T[-(num_time_slices - it) // restabilization_frequency]
            relevant_right_Q = params.right_checkpoints_down.Q[(num_time_slices - it) // restabilization_frequency - 1]
            relevant_right_D = params.right_checkpoints_down.D[(num_time_slices - it) // restabilization_frequency - 1]
            relevant_right_T = params.right_checkpoints_down.T[(num_time_slices - it) // restabilization_frequency - 1]
            # relevant_left_Q = params.left_checkpoints_down.Q[(num_time_slices - it) // restabilization_frequency - 1]
            # relevant_left_D = params.left_checkpoints_down.D[(num_time_slices - it) // restabilization_frequency - 1]
            # relevant_left_T = params.left_checkpoints_down.T[(num_time_slices - it) // restabilization_frequency - 1]
            relevant_left_Q = params.left_checkpoints_down.Q[0]
            relevant_left_D = params.left_checkpoints_down.D[0]
            relevant_left_T = params.left_checkpoints_down.T[0]

            all_cycled_propagator = form_propagators(-1.0 * params.auxiliary_field[:restabilization_frequency, :],
                                                     params.expKineticTerm,
                                                     params.expInteractionTermFunc,
                                                     params.time_slice_size,
                                                     params.interaction_strength)
            cycled_propagator = np.eye(all_cycled_propagator.shape[1], all_cycled_propagator.shape[2])
            for i in range(all_cycled_propagator.shape[0]):
                cycled_propagator = all_cycled_propagator[i] @ cycled_propagator

            to_decompose = relevant_left_D.reshape(-1, 1) * relevant_left_Q @ cycled_propagator
            new_left_Q, new_left_D, new_left_T = graded_TDQ_decomposition(to_decompose)
            new_left_T = relevant_left_T @ new_left_T

            # if ((num_time_slices - it) // restabilization_frequency) != params.left_checkpoints_down.Q.shape[0]:
            #     params.update_checkpoints((num_time_slices - it) // restabilization_frequency,
            #                               new_left_Q,
            #                               new_left_D,
            #                               new_left_T,
            #                               side='left', spin='down')
            params.update_checkpoints(0,
                                      new_left_Q,
                                      new_left_D,
                                      new_left_T,
                                      side='left', spin='down')

            params.G_down = stabilization_from_checkpoints(relevant_right_Q,
                                                           relevant_right_D,
                                                           relevant_right_T,
                                                           new_left_T, new_left_D, new_left_Q)

        if ((it - num_time_slices + 2) % measurement_freq) == 0:
            measurements.append(measurement_func(params))

    return measurements
