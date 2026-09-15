import numpy as np
import scipy


def form_propagators(auxiliary_field, exp_kinetic_term, exp_interaction_term_func,
                     time_slice_size, interaction_strength):
    propagators = np.zeros((auxiliary_field.shape[0], auxiliary_field.shape[1], auxiliary_field.shape[1]),
                           dtype=np.float64)
    for ts in range(auxiliary_field.shape[0]):
        propagators[ts] = exp_interaction_term_func(auxiliary_field[ts, :],
                                                    time_slice_size,
                                                    interaction_strength) @ exp_kinetic_term
    return propagators


def form_propagators_with_grouping(auxiliary_field, exp_kinetic_term, exp_interaction_term_func,
                                   time_slice_size, interaction_strength, grouping_width):
    all_propagators = form_propagators(auxiliary_field, exp_kinetic_term, exp_interaction_term_func,
                                       time_slice_size, interaction_strength)
    if (all_propagators.shape[0] % grouping_width) == 0:
        propagators_grouped = np.split(all_propagators, all_propagators.shape[0] // grouping_width, axis=0)
        propagators_grouped_reduced = np.zeros(shape=(len(propagators_grouped),
                                                      all_propagators.shape[1],
                                                      all_propagators.shape[2]), dtype=np.float64)
        for i in range(len(propagators_grouped)):
            propagators_grouped_reduced[i] = np.linalg.multi_dot(np.flip(propagators_grouped[i], axis=0))
    else:
        raise ValueError

    return propagators_grouped_reduced


# Class to help store and update checkpoints
class Checkpoint:
    # Initialize the checkpoint class with attributes with proper sizing
    def __init__(self, numCheckpoints, numSites):
        self.Q = np.zeros((numCheckpoints, numSites, numSites), dtype=np.float64)
        self.D = np.zeros((numCheckpoints, numSites), dtype=np.float64)
        self.T = np.zeros((numCheckpoints, numSites, numSites), dtype=np.float64)

    # Function to update the Q, D, and T at a given checkpoint index
    def update(self, updateInd, newQ, newD, newT):
        self.Q[updateInd] = newQ
        self.D[updateInd] = newD
        self.T[updateInd] = newT


def full_stabilization_computation(propagators, checkpointing_freq=None):

    num_checkpoints = None
    checkpointing = False
    if checkpointing_freq is not None:
        checkpointing = True
        if (propagators.shape[0] % checkpointing_freq) == 0:
            num_checkpoints = (propagators.shape[0] // checkpointing_freq) - 1
        else:
            num_checkpoints = propagators.shape[0] // checkpointing_freq
        right_checkpoints = Checkpoint(num_checkpoints, propagators.shape[1])

    # Graded QR with pivot of first propagator
    arrQ, arrR, permP = scipy.linalg.qr(propagators[0], pivoting=True)
    diagD = np.diag(arrR)
    arrT = ((1 / diagD).reshape(-1, 1) * arrR)[:, permP.argsort()]
    if checkpointing and ((propagators.shape[0] - 1) % checkpointing_freq) == 0:
        right_checkpoints.update(num_checkpoints - 1, arrQ, diagD, arrT)  # noqa

    # For loop to do Graded QR with presorted column norms of rest of the propagator chain
    # (updating right checkpoints as I go)
    for i in range(1, propagators.shape[0]):
        arrC = (propagators[i] @ arrQ) * diagD
        columnNormsOrder = np.flip(np.argsort(np.sum(arrC * np.conj(arrC), axis=0)))
        arrQ, arrR = scipy.linalg.qr(arrC[:, columnNormsOrder], pivoting=False) # noqa
        diagD = np.diag(arrR)
        arrT = ((1 / diagD).reshape(-1, 1) * arrR) @ arrT[columnNormsOrder, :]
        if checkpointing and ((propagators.shape[0] - (i + 1)) % checkpointing_freq) == 0 and i != (propagators.shape[0] - 1):
            right_checkpoints.update(num_checkpoints - (i + 1) // checkpointing_freq, arrQ, diagD, arrT)

    # Compute G in the end
    diagDb = np.ones(len(diagD))
    diagDb[np.abs(diagD) > 1] = 1 / np.abs(diagD[np.abs(diagD) > 1])
    diagDs = np.sign(diagD)
    diagDs[np.abs(diagD) <= 1] = diagD[np.abs(diagD) <= 1]
    arrG = (np.linalg.inv(arrT) * diagDb.reshape(1, -1)) @ np.linalg.inv((arrQ.T @ np.linalg.inv(arrT)) * diagDb.reshape(1, -1) + np.diag(diagDs)) @ arrQ.T

    if checkpointing:
        # left_checkpoints = Checkpoint(num_checkpoints, propagators.shape[1])
        left_checkpoints = Checkpoint(1, propagators.shape[1])
        left_checkpoints.update(0,
                                np.eye(propagators.shape[1]),
                                np.repeat(1.0, propagators.shape[1]),
                                np.eye(propagators.shape[1]))

        return arrG, left_checkpoints, right_checkpoints  # noqa

    else:
        return arrG


def stabilization_from_checkpoints(right_Q, right_D, right_T, left_T, left_D, left_Q):
    # middle_product = np.linalg.inv(left_Q @ right_Q)
    middle_product = right_Q.T @ left_Q.T
    # middle_product = middle_product + right_D.reshape(-1, 1) * (right_T @ left_T) * left_D
    other_product = (right_T @ left_T)
    np.multiply(other_product, left_D, out=other_product)
    np.multiply(right_D.reshape(-1, 1), other_product, out=other_product)
    middle_product += other_product
    middle_Q, middle_R, middle_perm = scipy.linalg.qr(middle_product, pivoting=True)

    middle_D = np.diag(middle_R)
    # middle_T = (1 / middle_D.reshape(-1, 1)) * middle_R[:, middle_perm.argsort()]
    middle_T = (1 / middle_D.reshape(-1, 1)) * middle_R

    # final_G = (1 / middle_D).reshape(-1, 1) * np.linalg.inv(right_Q @ middle_Q)
    final_G = (1 / middle_D).reshape(-1, 1) * middle_Q.T @ right_Q.T
    # final_G = np.linalg.inv(middle_T @ left_Q) @ final_G
    final_G = scipy.linalg.solve_triangular(middle_T.T, left_Q[middle_perm, :], lower=True).T @ final_G

    return final_G


def graded_TDQ_decomposition(arr):
    arr_Q, arr_R, perm = scipy.linalg.qr(arr.T, pivoting=True)
    diag_D = np.diag(arr_R)
    arr_T = (1 / diag_D.reshape(-1, 1)) * arr_R[:, perm.argsort()]

    return arr_Q.T, diag_D, arr_T.T

