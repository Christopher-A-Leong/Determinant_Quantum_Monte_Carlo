import numpy as np
import scipy
from DQMC.update import *
from DQMC.stabilization import full_stabilization_computation, form_propagators_with_grouping, form_propagators
from DQMC.sweep import *
from DQMC.OOP import StateSpinFactorizeWithCheckpoints


def grouping_freq_calibration(grouping_freq_list, system_params_dict, num_time_slices, time_slice_size, U,
                              hamiltonian_func, hubbard_hs_func):
    max_diff_G_up_list = []
    max_diff_G_down_list = []

    try:
        expKineticTerm = np.asarray(scipy.linalg.expm(-time_slice_size * hamiltonian_func(**system_params_dict).toarray()))
    except AttributeError:
        expKineticTerm = np.asarray(scipy.linalg.expm(-time_slice_size * hamiltonian_func(**system_params_dict)))

    aux_field = np.random.choice([-1, 1], replace=True, size=int(num_time_slices * expKineticTerm.shape[0])).reshape(num_time_slices, expKineticTerm.shape[0])

    prop_up = form_propagators(aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U)
    prop_down = form_propagators(-1.0 * aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U)
    G_up_nogrouping = full_stabilization_computation(prop_up, checkpointing_freq=None)
    G_down_nogrouping = full_stabilization_computation(prop_down, checkpointing_freq=None)

    for grouping_freq in grouping_freq_list:
        prop_up = form_propagators_with_grouping(aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U, grouping_freq)
        prop_down = form_propagators_with_grouping(-1.0 * aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U, grouping_freq)
        G_up = full_stabilization_computation(prop_up, checkpointing_freq=None)
        G_down = full_stabilization_computation(prop_down, checkpointing_freq=None)
        max_diff_G_up_list.append(np.max(np.abs(G_up_nogrouping - G_up)))
        max_diff_G_down_list.append(np.max(np.abs(G_down_nogrouping - G_down)))

    return max_diff_G_up_list, max_diff_G_down_list


def full_sweep_diagnostic_mode(params, grouping_freq,
                               do_delayed_update=False, delayed_update_freq=None,
                               do_submatrix_update=False, flush_update_freq=None):
    stabilization_max_diff_G_up_list = []
    stabilization_max_diff_G_down_list = []
    num_time_slices = params.auxiliary_field.shape[0]

    for it in range(num_time_slices - 1, -1, -1):
        try:
            st = time.time()
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

            print(it, time.time() - st)

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

            prop_up = form_propagators_with_grouping(params.auxiliary_field, params.expKineticTerm, params.expInteractionTermFunc, params.time_slice_size, params.interaction_strength, grouping_freq)
            prop_down = form_propagators_with_grouping(-1.0 * params.auxiliary_field, params.expKineticTerm, params.expInteractionTermFunc, params.time_slice_size, params.interaction_strength, grouping_freq)
            G_up_full_stabilize = full_stabilization_computation(prop_up, checkpointing_freq=None)
            G_down_full_stabilize = full_stabilization_computation(prop_down, checkpointing_freq=None)

            stabilization_max_diff_G_up_list.append(np.max(np.abs(params.G_up - G_up_full_stabilize)))
            stabilization_max_diff_G_down_list.append(np.max(np.abs(params.G_down - G_down_full_stabilize)))

        except ValueError:
            return stabilization_max_diff_G_up_list, stabilization_max_diff_G_down_list

    return stabilization_max_diff_G_up_list, stabilization_max_diff_G_down_list


def restabilization_freq_calibration(system_params_dict, num_time_slices, time_slice_size, U,
                           hamiltonian_func, hubbard_hs_func, hubbard_hs_inv_func,
                           do_grouping=False, grouping_freq=None, do_delayed_update=False, delayed_update_freq=None,
                           do_submatrix_update=False, flush_update_freq=None):

    try:
        expKineticTerm = np.asarray(scipy.linalg.expm(-time_slice_size * hamiltonian_func(**system_params_dict).toarray()))
        expKineticInvTerm = scipy.linalg.expm(time_slice_size * hamiltonian_func(**system_params_dict).toarray())
    except AttributeError:
        expKineticTerm = np.asarray(scipy.linalg.expm(-time_slice_size * hamiltonian_func(**system_params_dict)))
        expKineticInvTerm = scipy.linalg.expm(time_slice_size * hamiltonian_func(**system_params_dict))

    aux_field = np.random.choice([-1, 1], replace=True, size=int(num_time_slices * expKineticTerm.shape[0])).reshape(num_time_slices, expKineticTerm.shape[0])

    state = StateSpinFactorizeWithCheckpoints(G_up=None,
                                              G_down=None,
                                              auxiliary_field=aux_field,
                                              interaction_strength=U,
                                              time_slice_size=time_slice_size,
                                              expKineticTerm=expKineticTerm,
                                              expInteractionTermFunc=hubbard_hs_func,
                                              expKineticInvTerm=expKineticInvTerm,
                                              expInteractionInvTermFunc=hubbard_hs_inv_func,
                                              left_checkpoints_up=None,
                                              right_checkpoints_up=None,
                                              left_checkpoints_down=None,
                                              right_checkpoints_down=None
                                              )

    if do_grouping:
        prop_up = form_propagators_with_grouping(aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U, grouping_freq)
        prop_down = form_propagators_with_grouping(-1.0 * aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U, grouping_freq)
        state.G_up = full_stabilization_computation(prop_up, checkpointing_freq=None)
        state.G_down = full_stabilization_computation(prop_down, checkpointing_freq=None)
    else:
        prop_up = form_propagators(aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U)
        prop_down = form_propagators(-1.0 * aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U)
        state.G_up = full_stabilization_computation(prop_up, checkpointing_freq=None)
        state.G_down = full_stabilization_computation(prop_down, checkpointing_freq=None)

    max_diff_up, max_diff_down = full_sweep_diagnostic_mode(state,
                                                            grouping_freq,
                                                            do_delayed_update=do_delayed_update,
                                                            delayed_update_freq=delayed_update_freq,
                                                            do_submatrix_update=do_submatrix_update,
                                                            flush_update_freq=flush_update_freq)

    return max_diff_up, max_diff_down

