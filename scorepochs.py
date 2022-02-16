"""
 scorEpochs

 Function to select the best (most homogenoous) M/EEG epochs from a
 resting-state recordings.

 INPUT
    cfg: dictionary with the following key-value pairs
         freqRange    - list with the frequency range used to compute the power spectrum (see scipy.stats.spearmanr()
                        function)
         fs           - integer representing sample frequency
         windowL      - integer representing the window length (in seconds)
         smoothFactor - smoothing factor for the power spectrum (0 by default)
         wOverlap     - integer representing the number of seconds of overlap between two consecutive epochs (0 by
                        default)

    data: 2d array with the time-series (channels X time samples)



 OUTPUT

    idx_best_ep: list of indexes sorted according to the best score (this list should be used for the selection of the
                  best epochs)

    epoch:       3d list of the data divided in equal length epochs of length windowL (epochs X channels X time samples)

    score_Xep:   array of score per epoch


 Author:       Simone Maurizio La Cava
 Contributors: Benedetta Presicci





     Copyright (C) 2020 Simone Maurizio La Cava

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np
from scipy import signal as sig
from scipy import stats as st
import sys


def scorEpochs(cfg, data):
    """
    :param cfg:  dictionary with the following key-value pairs
                 freqRange    - list with the frequency range used to compute the power spectrum (see
                                scipy.stats.spearmanr() function)
                 fs           - integer representing sample frequency
                 windowL      - integer representing the window length (in seconds)
                 smoothFactor - smoothing factor for the power spectrum (0 by default)
                 wOverlap     - integer representing the number of seconds of overlap between two consecutive epochs (0
                                by default)
    :param data: 2d array with the time-series (channels X time samples)

    :return:     idx_best_ep - list of indexes sorted according to the best score (this list should be used for the
                               selection of the best epochs)
                 epoch       - 3d list of the data divided in equal length epochs of length windowL
                               (channels X epochs X time samples)
                 score_Xep   - array of score per epoch
    """
    epLen = cfg['windowL'] * cfg['fs']         # Number of samples of each epoch (for each channel)
    dataLen = len(data[0])                     # Total number of samples of the whole signal
    nCh = len(data)                            # Number of channels

    isOverlap = 'wOverlap' in cfg.keys()  # isOverlap = True if the user wants a sliding window; the user will assign the value in seconds of the overlap desired to the key 'wOverlap'
    if isOverlap:
        idx_jump = (cfg['windowL'] - cfg['wOverlap']) * cfg['fs']  # idx_jump is the number of samples that separates the beginning of an epoch and the following one
    else:
        idx_jump = epLen
    idx_ep = range(0, dataLen - epLen + 1, idx_jump)  # Indexes from which start each epoch

    nEp = len(idx_ep)                          # Total number of epochs
    epoch = np.zeros((nEp, nCh, epLen))        # Initialization of the returned 3D matrix
    freqRange = cfg['freqRange']               # Cut frequencies
    smoothing_condition = 'smoothFactor' in cfg.keys() and cfg['smoothFactor'] > 1 # True if the smoothing has to be executed, 0 otherwise

    for e in range(nEp):
        for c in range(nCh):
            epoch[e][c][0:epLen] = data[c][idx_ep[e]:idx_ep[e]+epLen]
            # compute power spectrum
            f, aux_pxx = sig.welch(epoch[e][c].T, cfg['fs'], window='hamming', nperseg=round(epLen/8), detrend=False) # The nperseg allows the MATLAB pwelch correspondence
            if c == 0 and e == 0: # The various parameters are obtained in the first interation
                pxx, idx_min, idx_max, nFreq = _spectrum_parameters(f, freqRange, aux_pxx, nEp, nCh)
                if smoothing_condition:
                    window_range, initial_f, final_f = _smoothing_parameters(cfg['smoothFactor'], nFreq)
            if smoothing_condition:
                pxx[e][c] = _movmean(aux_pxx, cfg['smoothFactor'], initial_f, final_f, nFreq, idx_min, idx_max)
            else:
                pxx[e][c] = aux_pxx[idx_min:idx_max+1] # pxx takes the only interested spectrum-related sub-array
    pxxXch = np.zeros((nEp, idx_max-idx_min+1))
    score_chXep = np.zeros((nCh, nEp))
    for c in range(nCh):
        for e in range(nEp):
            pxxXch[e] = pxx[e][c]
        score_ch, p = st.spearmanr(pxxXch, axis=1)          # Correlation between the spectra of the epochs within each channel
        score_chXep[c][0:nEp] += np.mean(score_ch, axis=1)  # Mean similarity score of an epoch with all the epochs for each channel
    score_Xep = np.mean(score_chXep, axis=0)                # The score of each epoch is equal to the mean of the scores of all the channels in that epoch 
    idx_best_ep = np.argsort(score_Xep)                     # Obtains of the indexes from the worst epoch to the best
    idx_best_ep = idx_best_ep[::-1]                         # Reversing to obtain the descending order (from the best to the worst)
    return idx_best_ep, epoch, score_Xep


def _movmean(aux_pxx, smoothFactor, initial_f, final_f, nFreq, idx_min, idx_max):   #It is not weighted
    """
    Function used for computing the smoothed power spectrum through moving average filter, where each output sample is
    evaluated on the center of the window at each iteration (or the one furthest to the right of the two in the center,
    in case of a window with an even number of elements), without padding on edges (FOR INTERNAL USE ONLY).
     X = [X(0), X(1), X(2), X(3), X(4)]
     smoothFactor = 3
     Y(0) = (X(0))+X(1))/2
     Y(1) = (X(0)+X(1)+X(2))/3
     Y(2) = (X(1)+X(2)+X(3))/3
     Y(3) = (X(2)+X(3)+X(4))/3
     Y(4) = (X(3)+X(4))/2
    """
    smoothed = np.zeros((idx_max-idx_min+1,))
    for f in range(nFreq):
        if f < initial_f:
            smoothed[f] = np.mean(aux_pxx[idx_min:idx_min+f+initial_f+1])
        elif f >= final_f:
            smoothed[f] = np.mean(aux_pxx[idx_min+f-initial_f:])
        elif smoothFactor % 2 == 0:
            smoothed[f] = np.mean(aux_pxx[idx_min+f-initial_f:idx_min+f+initial_f])
        else:
            smoothed[f] = np.mean(aux_pxx[idx_min+f-initial_f:idx_min+f+initial_f+1])
    return smoothed


def _spectrum_parameters(f, freqRange, aux_pxx, nEp, nCh):
    """
    Function which defines the spectrum parameters for the scorEpochs function (FOR INTERNAL USE ONLY).
    """
    idx_min = int(np.argmin(abs(f-freqRange[0])))
    idx_max = int(np.argmin(abs(f-freqRange[-1])))
    nFreq = len(aux_pxx[idx_min:idx_max+1])
    pxx = np.zeros((nEp, nCh, nFreq))
    return pxx, idx_min, idx_max, nFreq


def _smoothing_parameters(smoothFactor, nFreq):
    """
    Function which defines the parameters of the window to be used by the scorEpochs function in smoothing the spectrum
    (FOR INTERNAL USE ONLY).
    """
    window_range = round(smoothFactor)
    initial_f = int(window_range/2)
    final_f = nFreq - initial_f
    return window_range, initial_f, final_f


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfg_str = ''
        data_str = ''
        check = 0
        for el in sys.argv[1:len(sys.argv)]:
            if el[-1] == '}':
                check = 1
                cfg_str = cfg_str + el
            elif check == 0:
                cfg_str = cfg_str + el
            elif check == 1:
                data_str = data_str + el
        cfg = eval(cfg_str)
        data = eval(data_str)
        idx_best_ep, epoch, score_Xep = scorEpochs(cfg, data)
        print("idx_best_ep = " + str(idx_best_ep))
        print("epoch = " + str(epoch))
        print("score_Xep = " + str(score_Xep))
    else:
        print(len(sys.argv))
        print("This function aims to select the best (most homogenoous) M/EEG epochs from a resting-state recordings.")
        print('\nThe only required arguments are the a dictionary, containing all the necessary pairs key-value, and ',
              'the time series.')
        print('The keys of the dictionary are: \n\t - freqRange: a list containing the initial and the final value of ',
              'the frequency band which has to be considered (in Hz) \n\t - fs: an integer representing the sampling ',
              'frequency (in Hz) \n\t - windowL: an integer representing the length of each epoch in which the time ',
              'series has to be divided (in seconds) \n\t - smoothFactor: the smoothing factor for the power spectrum',
              ' (optional) \n\t - wOverlap: an integer representing the number of seconds of overlap between two',
              ' consecurive epochs (optional)')
        print('\nThe function returns a list containing the indexes of the best epochs, a 3d list containing the time ',
              'series divided in epochs (channels X epochs X time series), and the list of the scores assigned to each',
              ' epoch.')
        print('\n\nTaking for example a random time series of 10 channels and 10000 samples, contained in a 2d list ',
              '(10 X 10000), having a sampling frequency equal to 100 Hz, in order to evaluate the best epochs of 5 ',
              'seconds, studying the frequency band between 10 and 40 Hz, considering a smoothing factor (length of ',
              'the window used by the moving average in the power spectrum), it is necessary to use the following ',
              "dictionary:\n\t\t{'freqRange':[10, 40], 'fs':100, 'windowL':5, 'smoothFactor':3}")
        print('So, in order to execute the function using these parameters, it is possible to use:')
        print("idx_best, epoch, scores = scorEpochs({'freqRange':[10, 40], 'fs':100, 'windowL':5, 'smoothFactor':3}",
              ", time_series)")
        np.random.seed()
        time_series = np.zeros((10, 10000))
        x = np.linspace(1, 10000, 10000)
        for i in range(10):
            for j in range(10000):
                time_series[i][j] = np.random.normal(scale=1)
        idx_best, epoch, scores = scorEpochs({'freqRange':[10, 40], 'fs':100, 'windowL':5, 'smoothFactor':3},
                                             time_series)
        print("\n\nAs result, idx_best contains the list of the best epochs:")
        print(idx_best)
        print("\nThe 3d list named epoch contains the original signal segmented in epochs (epochs x channels X time ",
              "series) and, in this case, it has the following dimensions:")
        print(str(len(epoch)) + " " + str(len(epoch[0])) + " " + str(len(epoch[0][0])))
        print("\nFinally, scores contains the score of each epoch:")
        print(scores)
        print("\n\nFor example, in order to extract the five best epochs, it is possible to execute: \n\t ")
        print("best_epochs = np.zeros((5, len(epoch), len(epoch[0][0])))")
        print("for c in range(len(epoch[0])):")
        print("\tfor e in range(5):")
        print("\t\tbest_epochs[e][c] = epoch[idx_best[e]][c]")
        best_epochs = np.zeros((5, len(epoch), len(epoch[0][0])))
        for c in range(len(epoch[0])):
            for e in range(5):
                best_epochs[e][c] = epoch[idx_best[e]][c]
        print('\n\nIncluding an overlap of 1 second, considering again epochs of 5 seconds, then the first epoch ',
              'starts from the first sample (i.e. the first sample of the signal) and ends to 5 seconds, the second ',
              'epoch starts from the 4th second and ends with the 9th second, and so on:')
        print("idx_best, epoch, scores = scorEpochs({'freqRange':[10, 40], 'fs':100, 'windowL':5, 'smoothFactor':3, ",
              "'wOverlap', 1}, time_series)")
        idx_best, epoch, scores = scorEpochs({'freqRange': [10, 40], 'fs': 100, 'windowL': 5, 'smoothFactor': 3,
                                              'wOverlap':1}, time_series)
        print("\n\nAs result, we can see that idx_best contains a list which is higher than the previous one:")
        print(idx_best)
        print('\n\nThis tool can be used through the command line (do not be afraid to put spaces, they will be ',
              'automatically managed) or by importing it')
        print('In the last case you have two possibility: \n - Import the function from the module:'
              "\n\t from scorEpochs import scorEpochs \n\t idx_best, epoch, scores = scorEpochs(cfg, data) \n ",
              "- Import the module and use the function through the dot notation: \n\t idx_best, epoch, scores = ",
              "scorEpochs.scorEpochs(cfg, data)")