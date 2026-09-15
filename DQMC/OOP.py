from DQMC.stabilization import full_stabilization_computation
from DQMC.sweep import full_sweep_spin_factorize, full_sweep_spin_factorize_with_checkpoints


class State:

    def __init__(self, G=None, auxiliary_field=None,
                 interaction_strength=None, time_slice_size=None, expKineticTerm=None,
                 expInteractionTermFunc=None, expKineticInvTerm=None, expInteractionInvTermFunc=None):
        self.G = G
        self.auxiliary_field = auxiliary_field
        self.interaction_strength = interaction_strength
        self.time_slice_size = time_slice_size
        self.expKineticTerm = expKineticTerm
        self.expInteractionTermFunc = expInteractionTermFunc
        self.expKineticInvTerm = expKineticInvTerm
        self.expInteractionInvTermFunc = expInteractionInvTermFunc

    # def initial_form_greens(self, propagators):
    #     self.G = full_stabilization_computation(propagators)
    #
    # def full_sweep_compute(self, restablization_frequency, measurement_func, measurement_freq):
    #     return full_sweep(self, restablization_frequency, measurement_func, measurement_freq)


class StateSpinFactorize:

    def __init__(self, G_up=None, G_down=None, auxiliary_field=None, interaction_strength=None,
                 time_slice_size=None, expKineticTerm=None, expInteractionTermFunc=None, expKineticInvTerm=None,
                 expInteractionInvTermFunc=None):
        self.G_up = G_up
        self.G_down = G_down
        self.auxiliary_field = auxiliary_field
        self.interaction_strength = interaction_strength
        self.time_slice_size = time_slice_size
        self.expKineticTerm = expKineticTerm
        self.expInteractionTermFunc = expInteractionTermFunc
        self.expKineticInvTerm = expKineticInvTerm
        self.expInteractionInvTermFunc = expInteractionInvTermFunc

    def initial_form_greens(self, propagators_up, propagators_down):
        self.G_up = full_stabilization_computation(propagators_up)
        self.G_down = full_stabilization_computation(propagators_down)

    def full_sweep_compute(self, restablization_frequency, measurement_func, measurement_freq, do_delayed_update=False, delayed_update_freq=None):
        return full_sweep_spin_factorize(self, restablization_frequency, measurement_func, measurement_freq)

    def return_single_spin(self, which_spin):
        if which_spin == 'up':
            return State(G=self.G_up,
                         auxiliary_field=self.auxiliary_field,
                         interaction_strength=self.interaction_strength,
                         time_slice_size=self.time_slice_size,
                         expKineticTerm=self.expKineticTerm,
                         expInteractionTermFunc=self.expInteractionTermFunc,
                         expKineticInvTerm=self.expKineticInvTerm,
                         expInteractionInvTermFunc=self.expInteractionInvTermFunc)
        elif which_spin == 'down':
            return State(G=self.G_down,
                         auxiliary_field=(-1.0 * self.auxiliary_field),
                         interaction_strength=self.interaction_strength,
                         time_slice_size=self.time_slice_size,
                         expKineticTerm=self.expKineticTerm,
                         expInteractionTermFunc=self.expInteractionTermFunc,
                         expKineticInvTerm=self.expKineticInvTerm,
                         expInteractionInvTermFunc=self.expInteractionInvTermFunc)
        else:
            raise ValueError


class StateSpinFactorizeWithCheckpoints(StateSpinFactorize):
    def __init__(self, G_up=None, G_down=None, auxiliary_field=None, interaction_strength=None,
                 time_slice_size=None, expKineticTerm=None, expInteractionTermFunc=None, expKineticInvTerm=None,
                 expInteractionInvTermFunc=None, left_checkpoints_up=None, right_checkpoints_up=None,
                 left_checkpoints_down=None, right_checkpoints_down=None):
        super().__init__(G_up, G_down, auxiliary_field, interaction_strength,
                         time_slice_size, expKineticTerm, expInteractionTermFunc, expKineticInvTerm,
                         expInteractionInvTermFunc)
        self.left_checkpoints_up = left_checkpoints_up
        self.right_checkpoints_up = right_checkpoints_up
        self.left_checkpoints_down = left_checkpoints_down
        self.right_checkpoints_down = right_checkpoints_down

    def update_checkpoints(self, update_ind, new_Q, new_D, new_T, side=None, spin=None):
        if side == 'left' and spin == 'up':
            self.left_checkpoints_up.update(update_ind, new_Q, new_D, new_T)
        elif side == 'left' and spin == 'down':
            self.left_checkpoints_down.update(update_ind, new_Q, new_D, new_T)
        elif side == 'right' and spin == 'up':
            self.right_checkpoints_up.update(update_ind, new_Q, new_D, new_T)
        elif side == 'right' and spin == 'down':
            self.right_checkpoints_down.update(update_ind, new_Q, new_D, new_T)
        else:
            raise ValueError

    def initial_form_greens(self, propagators_up, propagators_down, checkpointing_freq=1):
        self.G_up, self.left_checkpoints_up, self.right_checkpoints_up = full_stabilization_computation(propagators_up,
                                                                                                        checkpointing_freq=checkpointing_freq)
        self.G_down, self.left_checkpoints_down, self.right_checkpoints_down = full_stabilization_computation(propagators_down,
                                                                                                              checkpointing_freq=checkpointing_freq)

    def full_sweep_compute(self, restablization_frequency, measurement_func, measurement_freq,
                           do_delayed_update=False, delayed_update_freq=None,
                           do_submatrix_update=False, flush_update_freq=None):
        return full_sweep_spin_factorize_with_checkpoints(self, restablization_frequency,
                                                          measurement_func, measurement_freq,
                                                          do_delayed_update, delayed_update_freq,
                                                          do_submatrix_update, flush_update_freq)

