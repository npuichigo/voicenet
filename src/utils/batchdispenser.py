# Copyright 2016 ASLP@NPU.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: npuichigo@gmail.com (zhangyuchao)
"""Mini-batch dispenser, including frame and utterance batch dispenser.

This module provides mini-batch dispensers for accoustic model training.
Frame batch dispenser is used to provide frame level mini-batches for DNN
training, while utterance batch dispenser is used to provide utterance
level mini-batches for sequential training (RNN-LSTM).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import struct
from io_funcs.kaldi_io import ArkReader


class FeatureReader:
    """Class that can prepare features read from a kaldi archive via ArkReader

    ArkReader class from kaldi_io module provides method for reading kaldi
    archive, while this class prepare final features for training
    (mini-batches) after process them (cmvn and splicing).

    Attributes:
        reader: An ArkReader for reading kaldi archive.
        cmvnfile:
    """

    def __init__(self, scpfile, context_width, cmvnfile=None):
        """Init ArkReader along with cmvn_stats and context_width.

        Args:
            scpfile: Path to the features .scp file.
            context_width: Context width for splicing the features.
            cmvnfile: Path to the cmvn file.
        """
        # create the feature reader
        self.reader = ArkReader(scpfile)

        # read cmvnfile and calculate mean and variance
        self.cmvnfile = cmvnfile
        if self.cmvnfile:
            self.read_cmvn_file(cmvnfile)

        # store the context width
        self.context_width = context_width

    def read_cmvn_file(self, cmvnfile):
        """Read cmvnfile with kaldi .ark format, then calculating mean and var.

        Args:
            cmvnfile: The name of cmvnfile with kaldi archive format.
        """
        # read the cmvn statistics
        cmvn_stats = ArkReader.read_ark(cmvnfile)

        # compute mean
        self.mean = cmvn_stats[0,:-1]/cmvn_stats[0,-1]

        # compute variance
        self.variance = cmvn_stats[1,:-1]/cmvn_stats[0,-1] - np.square(self.mean)

        # floor zero variance
        floor = 1.0e-20
        for index in range(self.variance.shape[0]):
            if self.variance[index] < floor:
                self.variance[index] = floor

    def get_next_utt(self):
        """Read the next features from the archive, normalize and splice them.

        Returns:
            The utterance id, correspoding normalized and spliced features and
            a bool indicating if the reader looped back to the begining.
        """
        # read utterance
        (utt_id, utt_mat, looped) = self.reader.read_next_utt()

        # apply cmvn
        if self.cmvnfile:
            utt_mat = apply_cmvn(utt_mat, self.mean, self.variance)

        # splice the utterance
        utt_mat = splice(utt_mat, self.context_width)

        return utt_id, utt_mat, looped

    def get_utt(self, utt_id):
        """Read the features of a certain utterance ID from the archive, normalize and splice them.

        Returns:
            The normalized and spliced features.
        """
        # read utterance
        utt_mat = self.reader.read_utt_data_from_id(utt_id)

        # apply cmvn
        if self.cmvnfile:
            utt_mat = apply_cmvn(utt_mat, self.mean, self.variance)

        # splice the utterance
        utt_mat = splice(utt_mat,self.context_width)

        return utt_mat

    def next_id(self):
        """Only gets the ID of the next utterance (also moves forward in the reader).

        Returns:
            The ID of the utterance.
        """
        return self.reader.read_next_scp()

    def prev_id(self):
        """Only gets the ID of the previous utterance (also moves backward in the reader).

        Returns:
            The ID of the utterance.
        """
        return self.reader.read_previous_scp()

    def split(self):
        """Split of the features that have been read so far."""
        self.reader.split()


class FrameBatchDispenser:
    """Class that dispenses mini-batches for DNN training.

    In view of DNN's training on frames, this class provide frame level
    mini-batches. Above all, this class implements kaldi's buffer mechanism,
    which can significantly decrease memory requirements needed for training.

    Attributes:
        feature_reader: A feature reader object.
        target_reader: A target reader object.
        minibatch_size: The size of one mini-batch.
        buffer_size: The size of the buffer pool for storing features and
            targets.
        data: A list of feature data (with buffer_size).
        labels: A list of target data (with buffer_size).
        data_begin: The anchor indicating the current pos of buffer pool.
        data_end: The anchor indicating the end pos of buffer pool.
        looped: A bool indicating whether an iteration should be done.
        mask: A randomized list for logically shufflling data and labels.
    """

    def __init__(self, feature_reader, target_reader, minibatch_size=64, buffer_size=32768):
        """FrameBatchDispenser constructor.

        Args:
            feature_reader: A feature reader object.
            target_reader: A target reader object.
            minibatch_size: The size of one mini-batch.
            buffer_size: The size of the buffer pool for storing features and
                targets.
        """
        # store the feature reader
        self.feature_reader = feature_reader
        self.feature_reader.reader.shuffle()

        # read the target reader
        self.target_reader = target_reader

        # store the batch size
        self.minibatch_size = minibatch_size

        # coarsely the number of lines read into memory at one time
        self.buffer_size = buffer_size
        # current mini-batch starting point and actual size of buffer
        self.data_begin = 0
        self.data_end = 0

        # indicate if utterances are looped
        self.looped = 0

    def prepare_data(self):
        """Prepare buffer pool data when certain conditions are meet.

        If anchor data_begin is zero, which means no data have been read, or
        if anchor data_end minus data_begin is less than minibatch_size, load
        new data.
        """
        # optionally put previous left-over to front
        leftover = self.data_end - self.data_begin;
        if self.data_begin > 0 and leftover > 0:
            self.data = [self.data[self.data_begin:, :]]
            self.labels = [self.labels[self.data_begin:, :]]
            self.data_begin = 0
            self.data_end = leftover
        else:
            self.data = []
            self.labels = []
            self.data_begin = 0
            self.data_end = 0

        while self.data_end < self.buffer_size:
            # read utterance
            utt_id, utt_mat, looped = self.feature_reader.get_next_utt()
            # if the reader looped back to the begining, one iteration was
            # done. So we need to set looped for noticing.
            if looped:
                self.looped = True
                break
            # check if utterance has an target
            if utt_id in self.target_reader.reader.utt_ids:
                # add the features and targets to the batch
                self.data.append(utt_mat)
                self.labels.append(self.target_reader.get_utt(utt_id))

                self.data_end += utt_mat.shape[0]
            else:
                print('WARNING no target for %s' % utt_id)

        # concatenate the feature and target data
        self.data = np.vstack(self.data)
        self.labels = np.vstack(self.labels)
        # mask for randomization
        self.mask = np.arange(self.data.shape[0])
        np.random.shuffle(self.mask)

    def get_batch(self):
        """Get a mini-batch of features and targets.

        This method is used to get a mini-batch, using mask to randomize data.

        Returns:
            A mini-batch of data and the corresponding labels.
        """
        if self.data_begin == 0 or self.data_end - self.data_begin < self.minibatch_size:
            self.prepare_data()

        # get randomized batch data and labels
        batch_data = self.data[self.mask[self.data_begin:self.data_begin+self.minibatch_size], :]
        batch_labels = self.labels[self.mask[self.data_begin:self.data_begin+self.minibatch_size], :]

        self.data_begin += self.minibatch_size

        return (batch_data, batch_labels)

    def done(self):
        """Check if one iteration is done.

        if utterance is looped back to the beginning and remaining data is
        insufficient for one minibatch, one iteration is done.

        Returns:
            A bool indicating whether one iteration is done.
        """
        if self.data_end - self.data_begin < self.minibatch_size and self.looped:
            # prepare next iteration(reset looped and shuffle utterances)
            self.looped = False
            self.feature_reader.reader.shuffle()
            return True
        else:
            return False

    def split(self):
        """Split of the part that has allready been read by the batchdispenser.

        This can be used to read a validation set and then split it of from
        the rest.
        """
        self.feature_reader.split()


class UttBatchDispenser:
    """Class that dispenses mini-batches for sequential training (RNN-LSTM).

    In view of RNN-LSTM's training on sequence, this class provide utterance
    level mini-batches. Unlike FrameBatchDispenser, this class provide list
    of features and targets data due to utterances have different lengths and
    cannot be put into a numpy matrix before padding.

    Attributes:
        feature_reader: A feature reader object.
        target_reader: A target reader object.
        minibatch_size: The size of one mini-batch.
        looped: A bool indicating whether an iteration should be done.
    """

    def __init__(self, feature_reader, target_reader, minibatch_size=64):
        """UttBatchDispenser constructor.

        Args:
            feature_reader: A feature reader object.
            target_reader: A target reader object.
            minibatch_size: The size of one mini-batch.
        """
        #store the feature reader
        self.feature_reader = feature_reader
        self.feature_reader.reader.shuffle()

        #read the target reader
        self.target_reader = target_reader

        #store the batch size
        self.minibatch_size = minibatch_size

        # indicate if utterances are looped
        self.looped = 0

    def get_batch(self):
        """Get a batch of features and targets(utterance).

        Returns:
            A list with one mini-batch of data, the corresponding labels.
        """
        n = 0
        batch_data = []
        batch_labels = []
        while n < self.minibatch_size:
            #read utterance
            utt_id, utt_mat, looped = self.feature_reader.get_next_utt()
            # if the reader looped back to the begining, one iteration was
            # done. So we need to set looped for noticing.
            if looped:
                self.looped = True
                break
            #check if utterances has targets
            if utt_id in self.target_reader.reader.utt_ids:
                #add the features and targets to the batch
                batch_data.append(utt_mat)
                batch_labels.append(self.target_reader.get_utt(utt_id))

                n += 1
            else:
                print('WARNING no target for %s' % utt_id)

        return (batch_data, batch_labels)

    def done(self):
        """Check if one iteration is done.

        if utterance is looped back to the beginning and remaining data is
        insufficient for one minibatch, one iteration is done.

        Returns:
            A bool indicating whether one iteration is done.
        """
        if self.looped:
            # prepare next iteration(reset looped and shuffle utterances)
            self.looped = False
            self.feature_reader.reader.shuffle()
            return True
        else:
            return False

    def split(self):
        """Split of the part that has allready been read by the batchdispenser.

        This can be used to read a validation set and then split it of from
        the rest.
        """
        self.feature_reader.split()

    @property
    def num_utt(self):
        """The number of utterances"""
        return len(self.feature_reader.reader.utt_ids)


def apply_cmvn(utt, mean, variance, reverse=False):
    """Apply mean and variance normalisation based on previously computed statistics.

    Args:
        utt: The utterance feature numpy matrix.
        stats: A numpy array containing the mean and variance statistics.
            The first row contains the sum of all the fautures and as a last
            element the total numbe of features. The second row contains the
            squared sum of the features and a zero at the end

    Returns:
        A numpy array containing the mean and variance normalized features
    """
    if not reverse:
        #return mean and variance normalised utterance
        return np.divide(np.subtract(utt, mean), np.sqrt(variance))
    else:
        #reversed normalization
        return np.add(np.multiply(utt, np.sqrt(variance)), mean)


def splice(utt, context_width):
    """Splice the utterance.

    Args:
        utt: Numpy matrix containing the utterance features to be spliced.
        context_width: How many frames to the left and right should be
        concatenated.

    Returns:
        A numpy array containing the spliced features
    """
    #create spliced utterance holder
    utt_spliced = np.zeros(shape = [utt.shape[0],utt.shape[1]*(1+2*context_width)], dtype=np.float32)

    #middle part is just the uttarnce
    utt_spliced[:,context_width*utt.shape[1]:(context_width+1)*utt.shape[1]] = utt

    for i in range(context_width):
        #add left context
        utt_spliced[
            i+1:utt_spliced.shape[0],
            (context_width-i-1)*utt.shape[1]:(context_width-i)*utt.shape[1]
            ] = utt[0:utt.shape[0]-i-1,:]
        utt_spliced[
            0:i+1,
            (context_width-i-1)*utt.shape[1]:(context_width-i)*utt.shape[1]
            ] = np.tile(utt[0,:], (i+1,1))

        #add right context
        utt_spliced[
            0:utt_spliced.shape[0]-i-1,
            (context_width+i+1)*utt.shape[1]:(context_width+i+2)*utt.shape[1]
            ] = utt[i+1:utt.shape[0],:]
        utt_spliced[
            utt_spliced.shape[0]-i-1:utt_spliced.shape[0],
            (context_width+i+1)*utt.shape[1]:(context_width+i+2)*utt.shape[1]
            ] = np.tile(utt[utt.shape[0]-1,:], (i+1,1))

    return utt_spliced
