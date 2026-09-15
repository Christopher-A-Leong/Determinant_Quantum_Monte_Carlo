import numpy as np
import scipy
from DQMC.OOP import StateSpinFactorizeWithCheckpoints
from DQMC.stabilization import form_propagators, form_propagators_with_grouping, full_stabilization_computation
import time
import pickle


def run_simulation_hubbard(system_params_dict, num_time_slices, time_slice_size, U, num_sweeps,
                           measure_func, hamiltonian_func, hubbard_hs_func, hubbard_hs_inv_func,
                           do_grouping=True, measurement_freq=1, do_delayed_update=False, delayed_update_freq=None,
                           do_submatrix_update=False, flush_update_freq=None,
                           restabilization_freq=1, save=True, save_name=None, save_directory=None,
                           backup_save_freq=None):
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
        prop_up = form_propagators_with_grouping(aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U, restabilization_freq)
        prop_down = form_propagators_with_grouping(-1.0 * aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U, restabilization_freq)
        state.initial_form_greens(prop_up, prop_down, checkpointing_freq=1)
    else:
        prop_up = form_propagators(aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U)
        prop_down = form_propagators(-1.0 * aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U)
        state.initial_form_greens(prop_up, prop_down, checkpointing_freq=restabilization_freq)

    measurements = []
    per_sweep_times = []

    for i in range(num_sweeps):

        sweep_st = time.time()
        measurements_result = state.full_sweep_compute(restabilization_freq, measure_func, measurement_freq,
                                                       do_delayed_update, delayed_update_freq,
                                                       do_submatrix_update, flush_update_freq)
        measurements.append(measurements_result)

        if do_grouping:
            propagators_up = form_propagators_with_grouping(state.auxiliary_field,
                                                            state.expKineticTerm,
                                                            state.expInteractionTermFunc,
                                                            state.time_slice_size,
                                                            state.interaction_strength,
                                                            restabilization_freq)
            state.G_up, state.left_checkpoints_up, state.right_checkpoints_up = full_stabilization_computation(propagators_up,
                                                                                                               checkpointing_freq=1)

            propagators_down = form_propagators_with_grouping(-1.0 * state.auxiliary_field,
                                                              state.expKineticTerm,
                                                              state.expInteractionTermFunc,
                                                              state.time_slice_size,
                                                              state.interaction_strength,
                                                              restabilization_freq)
            state.G_down, state.left_checkpoints_down, state.right_checkpoints_down = full_stabilization_computation(propagators_down,
                                                                                                                     checkpointing_freq=1)
        else:
            propagators_up = form_propagators(state.auxiliary_field,
                                              state.expKineticTerm,
                                              state.expInteractionTermFunc,
                                              state.time_slice_size,
                                              state.interaction_strength)
            state.G_up, state.left_checkpoints_up, state.right_checkpoints_up = full_stabilization_computation(propagators_up,
                                                                                                               checkpointing_freq=restabilization_freq)

            propagators_down = form_propagators(-1.0 * state.auxiliary_field,
                                                state.expKineticTerm,
                                                state.expInteractionTermFunc,
                                                state.time_slice_size,
                                                state.interaction_strength)
            state.G_down, state.left_checkpoints_down, state.right_checkpoints_down = full_stabilization_computation(propagators_down,
                                                                                                                     checkpointing_freq=restabilization_freq)
        per_sweep_times.append(time.time() - sweep_st)

        if (backup_save_freq is not None) and (((i + 1) % backup_save_freq) == 0):
            result = {
                'system_params_dict': system_params_dict,
                'num_time_slices': num_time_slices,
                'time_slice_size': time_slice_size,
                'interaction_strength': U,
                'energy_measurements': measurements,
                'times_per_sweep': per_sweep_times,
                'num_sweeps': i + 1,
                'restabilization_frequency': restabilization_freq,
                'do_delayed_update': do_delayed_update,
                'delayed_update_freq': delayed_update_freq,
                'do_submatrix_update': do_submatrix_update,
                'flush_update_freq': flush_update_freq,
                'last_auxiliary_field': state.auxiliary_field
            }

            savefile = open(save_directory + '/' + save_name, 'wb')
            pickle.dump(result, savefile)
            savefile.close()

    if save:
        result = {
            'system_params_dict': system_params_dict,
            'num_time_slices': num_time_slices,
            'time_slice_size': time_slice_size,
            'interaction_strength': U,
            'energy_measurements': measurements,
            'times_per_sweep': per_sweep_times,
            'num_sweeps': num_sweeps,
            'restabilization_frequency': restabilization_freq,
            'do_delayed_update': do_delayed_update,
            'delayed_update_freq': delayed_update_freq,
            'do_submatrix_update': do_submatrix_update,
            'flush_update_freq': flush_update_freq,
            'last_auxiliary_field': state.auxiliary_field
        }

        savefile = open(save_directory + '/' + save_name, 'wb')
        pickle.dump(result, savefile)
        savefile.close()


def resume_simulation_hubbard(prev_run_data_file, num_sweeps,
                              measure_func, hamiltonian_func, hubbard_hs_func, hubbard_hs_inv_func,
                              do_grouping=True, measurement_freq=1,
                              save=True, save_name=None, save_directory=None,
                              backup_save_freq=None):

    system_params_dict = prev_run_data_file['system_params_dict']
    time_slice_size = prev_run_data_file['time_slice_size']
    num_time_slices = prev_run_data_file['num_time_slices']
    U = prev_run_data_file['interaction_strength']
    completed_num_sweeps = prev_run_data_file['num_sweeps']
    num_sweeps_to_do = num_sweeps - completed_num_sweeps
    if num_sweeps_to_do <= 0:
        raise ValueError

    aux_field = prev_run_data_file['last_auxiliary_field']

    restabilization_freq = prev_run_data_file['restabilization_frequency']
    do_delayed_update = prev_run_data_file['do_delayed_update']
    delayed_update_freq = prev_run_data_file['delayed_update_freq']
    do_submatrix_update = prev_run_data_file['do_submatrix_update']
    flush_update_freq = prev_run_data_file['flush_update_freq']

    measurements = prev_run_data_file['energy_measurements']
    per_sweep_times = prev_run_data_file['times_per_sweep']

    try:
        expKineticTerm = np.asarray(scipy.linalg.expm(-time_slice_size * hamiltonian_func(**system_params_dict).toarray()))
        expKineticInvTerm = np.asarray(scipy.linalg.expm(time_slice_size * hamiltonian_func(**system_params_dict).toarray()))
    except AttributeError:
        expKineticTerm = np.asarray(scipy.linalg.expm(-time_slice_size * hamiltonian_func(**system_params_dict)))
        expKineticInvTerm = np.asarray(scipy.linalg.expm(time_slice_size * hamiltonian_func(**system_params_dict)))

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
        prop_up = form_propagators_with_grouping(aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U,
                                                 restabilization_freq)
        prop_down = form_propagators_with_grouping(-1.0 * aux_field, expKineticTerm, hubbard_hs_func, time_slice_size,
                                                   U, restabilization_freq)
        state.initial_form_greens(prop_up, prop_down, checkpointing_freq=1)
    else:
        prop_up = form_propagators(aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U)
        prop_down = form_propagators(-1.0 * aux_field, expKineticTerm, hubbard_hs_func, time_slice_size, U)
        state.initial_form_greens(prop_up, prop_down, checkpointing_freq=restabilization_freq)

    for i in range(num_sweeps_to_do):

        sweep_st = time.time()
        measurements_result = state.full_sweep_compute(restabilization_freq, measure_func, measurement_freq,
                                                       do_delayed_update, delayed_update_freq,
                                                       do_submatrix_update, flush_update_freq)
        measurements.append(measurements_result)

        if do_grouping:
            propagators_up = form_propagators_with_grouping(state.auxiliary_field,
                                                            state.expKineticTerm,
                                                            state.expInteractionTermFunc,
                                                            state.time_slice_size,
                                                            state.interaction_strength,
                                                            restabilization_freq)
            state.G_up, state.left_checkpoints_up, state.right_checkpoints_up = full_stabilization_computation(
                propagators_up,
                checkpointing_freq=1)

            propagators_down = form_propagators_with_grouping(-1.0 * state.auxiliary_field,
                                                              state.expKineticTerm,
                                                              state.expInteractionTermFunc,
                                                              state.time_slice_size,
                                                              state.interaction_strength,
                                                              restabilization_freq)
            state.G_down, state.left_checkpoints_down, state.right_checkpoints_down = full_stabilization_computation(
                propagators_down,
                checkpointing_freq=1)
        else:
            propagators_up = form_propagators(state.auxiliary_field,
                                              state.expKineticTerm,
                                              state.expInteractionTermFunc,
                                              state.time_slice_size,
                                              state.interaction_strength)
            state.G_up, state.left_checkpoints_up, state.right_checkpoints_up = full_stabilization_computation(
                propagators_up,
                checkpointing_freq=restabilization_freq)

            propagators_down = form_propagators(-1.0 * state.auxiliary_field,
                                                state.expKineticTerm,
                                                state.expInteractionTermFunc,
                                                state.time_slice_size,
                                                state.interaction_strength)
            state.G_down, state.left_checkpoints_down, state.right_checkpoints_down = full_stabilization_computation(
                propagators_down,
                checkpointing_freq=restabilization_freq)

        per_sweep_times.append(time.time() - sweep_st)

        if (backup_save_freq is not None) and (((i + 1) % backup_save_freq) == 0):
            result = {
                'system_params_dict': system_params_dict,
                'num_time_slices': num_time_slices,
                'time_slice_size': time_slice_size,
                'interaction_strength': U,
                'energy_measurements': measurements,
                'times_per_sweep': per_sweep_times,
                'num_sweeps': i + 1 + completed_num_sweeps,
                'restabilization_frequency': restabilization_freq,
                'do_delayed_update': do_delayed_update,
                'delayed_update_freq': delayed_update_freq,
                'do_submatrix_update': do_submatrix_update,
                'flush_update_freq': flush_update_freq,
                'last_auxiliary_field': state.auxiliary_field
            }

            savefile = open(save_directory + '/' + save_name, 'wb')
            pickle.dump(result, savefile)
            savefile.close()

    if save:
        result = {
            'system_params_dict': system_params_dict,
            'num_time_slices': num_time_slices,
            'time_slice_size': time_slice_size,
            'interaction_strength': U,
            'energy_measurements': measurements,
            'times_per_sweep': per_sweep_times,
            'num_sweeps': num_sweeps,
            'restabilization_frequency': restabilization_freq,
            'do_delayed_update': do_delayed_update,
            'delayed_update_freq': delayed_update_freq,
            'do_submatrix_update': do_submatrix_update,
            'flush_update_freq': flush_update_freq,
            'last_auxiliary_field': state.auxiliary_field
        }

        savefile = open(save_directory + '/' + save_name, 'wb')
        pickle.dump(result, savefile)
        savefile.close()
