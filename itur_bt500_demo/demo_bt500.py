import argparse
import csv
import sys
import pprint

import numpy as np
from scipy import linalg


def read_csv_into_3darray(csv_filepath):
    """
    Read data from CSV file.

    The data should be organized in a 2D matrix, separated by comma. Each row
    correspond to a PVS; each column corresponds to a subject. If a vote is
    missing, a 'nan' is put in place.

    If some subjects evaluated a PVS multiple times, another 2D matrix of the
    same size [num_PVS, num_subjects] can be added under the first one. A row
    with a single comma (,) should be placed before the repetition matrix.
    Where the repeated vote is not available, a 'nan' is put in place.

    :param csv_filepath: filepath to the CSV file.
    :return: the numpy array in 3D [num_PVS, num_subjects, num_repetitions].
    """

    data = []
    data3dlist = []
    with open(csv_filepath, 'rt') as datafile:
        datareader = csv.reader(datafile, delimiter=',')

        for row in datareader:
            if row != ["", ""]:
                data.append(np.array(row, dtype=np.float64))
            else:
                data3dlist.append(data)
                data = []
        data3dlist.append(data)

    data3d = np.zeros([len(data3dlist[0]), len(data3dlist[0][0]), len(data3dlist)])

    for r_idx, r_mat in enumerate(data3dlist):
        data3d[:, :, r_idx] = r_mat

    return data3d


def weighed_nanmean_2d(a, wts, axis):
    """
    Compute the weighted arithmetic mean along the specified axis, ignoring
    NaNs. It is similar to numpy's nanmean function, but with a weight.

    :param a: 1D array.
    :param wts: 1D array carrying the weights.
    :param axis: either 0 or 1, specifying the dimension along which the means
    are computed.
    :return: 1D array containing the mean values.
    """

    assert len(a.shape) == 2
    assert axis in [0, 1]
    d0, d1 = a.shape
    if axis == 0:
        return np.divide(
            np.nansum(np.multiply(a, np.tile(wts, (d1, 1)).T), axis=0),
            np.nansum(np.multiply(~np.isnan(a), np.tile(wts, (d1, 1)).T), axis=0)
        )
    elif axis == 1:
        return np.divide(
            np.nansum(np.multiply(a, np.tile(wts, (d0, 1))), axis=1),
            np.nansum(np.multiply(~np.isnan(a), np.tile(wts, (d0, 1))), axis=1),
        )
    else:
        assert False


def one_or_nan(x):
    """
    Construct a "mask" array with the same dimension as x, with element NaN
    where x has NaN at the same location; and element 1 otherwise.

    :param x: array_like
    :return: an array with the same dimension as x
    """
    y = np.ones(x.shape)
    y[np.isnan(x)] = float('nan')
    return y


def get_sos_j(sig_r_j, o_jir):
    """
    Compute SOS (standard deviation of score) for PVS j
    :param sig_r_j: 
    :param o_jir:
    :return: array containing the SOS for PVS j
    """
    den = np.nansum(
        stack_3rd_dimension_along_axis(
            one_or_nan(o_jir) / np.tile(sig_r_j ** 2, (o_jir.shape[1], 1)).T[:, :, None],
            axis=1),
        axis=1)
    s_j_std = 1.0 / np.sqrt(np.maximum(0., den))
    return s_j_std


def stack_3rd_dimension_along_axis(o_jir, axis):
    """
        Take the 3D input matrix, slice it along the 3rd axis and stack the resulting 2D matrices
        along the selected matrix while maintaining the correct order.
        :param o_jir: 3D array of the shape [J, I, R]
        :param axis: 0 or 1
        :return: 2D array containing the values
            - if axis=0, the new shape is [R*J, I]
            - if axis = 1, the new shape is [J, R*I]
    """

    assert len(o_jir.shape) == 3
    J, I, R = o_jir.shape

    if axis == 0:
        o = np.zeros([R * J, I])

        for r in range(R):
            o[r * J:(r+1)*J, :] = o_jir[:, :, r]

    elif axis == 1:
        o = np.zeros([J, R * I])

        for r in range(R):
            o[:, r * I:(r + 1) * I] = o_jir[:, :, r]

    else:
        NotImplementedError

    return o



def run_alternating_projection(o_jir):
    """
    Run Alternating Projection (AP) algorithm.

    :param o_jir: 3D numpy array containing raw votes. The first dimension
    corresponds to the PVSs (j); the second dimension corresponds to the
    subjects (i); the third dimension correspons to the repetitions (r).
    If a vote is missing, the element is NaN.

    :return: dictionary containing results keyed by 'mos_j', 'sos_j', 'bias_i'
    and 'inconsistency_i'.
    """
    J, I, R = o_jir.shape

    # video by video, estimate MOS by averaging over subjects
    psi_j = np.nanmean(stack_3rd_dimension_along_axis(o_jir, axis=1), axis=1)  # mean marginalized over i

    # subject by subject, estimate subject bias by comparing with MOS
    b_jir = o_jir - np.tile(psi_j, (I, 1)).T[:, :, None]
    b_i = np.nanmean(stack_3rd_dimension_along_axis(b_jir, axis=0), axis=0)  # mean marginalized over j

    MAX_ITR = 1000
    DELTA_THR = 1e-8
    EPSILON = 1e-8

    itr = 0
    while True:

        psi_j_prev = psi_j

        # subject by subject, estimate subject inconsistency by averaging the
        # residue over stimuli
        r_jir = o_jir - np.tile(psi_j, (I, 1)).T[:, :, None] - np.tile(b_i, (J, 1))[:, :, None]
        sig_r_i = np.nanstd(stack_3rd_dimension_along_axis(r_jir, axis=0), axis=0)
        sig_r_j = np.nanstd(stack_3rd_dimension_along_axis(r_jir, axis=1), axis=1)

        # video by video, estimate MOS by averaging over subjects, inversely
        # weighted by residue variance
        w_i = 1.0 / (sig_r_i ** 2 + EPSILON)
        # mean marginalized over i:
        psi_j = weighed_nanmean_2d(
            stack_3rd_dimension_along_axis(o_jir - np.tile(b_i, (J, 1))[:, :, None], axis=1),
            wts=np.tile(w_i, R),  # same weights for the repeated observations
            axis=1)

        # subject by subject, estimate subject bias by comparing with MOS,
        # inversely weighted by residue variance
        b_jir = o_jir - np.tile(psi_j, (I, 1)).T[:, :, None]
        # mean marginalized over j:
        b_i = np.nanmean(stack_3rd_dimension_along_axis(b_jir, axis=0), axis=0)

        itr += 1

        delta_s_j = linalg.norm(psi_j_prev - psi_j)

        msg = 'Iteration {itr:4d}: change {delta_psi_j}, psi_j {psi_j}, ' \
              'b_i {b_i}, sig_r_i {sig_r_i}'.format(
            itr=itr, delta_psi_j=delta_s_j, psi_j=np.mean(psi_j),
            b_i=np.mean(b_i), sig_r_i=np.mean(sig_r_i))

        sys.stdout.write(msg + '\r')
        sys.stdout.flush()

        if delta_s_j < DELTA_THR:
            break

        if itr >= MAX_ITR:
            break

    psi_j_std = get_sos_j(sig_r_j, o_jir)
    sys.stdout.write("\n")

    mean_b_i = np.mean(b_i)
    b_i -= mean_b_i
    psi_j += mean_b_i

    return {
        'mos_j': list(psi_j),
        'sos_j': list(psi_j_std),
        'bias_i': list(b_i),
        'inconsistency_i': list(sig_r_i),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-csv", dest="input_csv", nargs=1, type=str,
        help="Filepath to input CSV file. The data should be organized in a 2D "
             "matrix, separated by comma. The rows correspond to PVSs; the "
             "columns correspond to subjects. If a vote is missing, input 'nan'"
             " instead.", required=True)

    args = parser.parse_args()
    input_csv = args.input_csv[0]

    o_jir = read_csv_into_3darray(input_csv)

    ret = run_alternating_projection(o_jir)

    pprint.pprint(ret)


