import argparse
import datetime
import gzip
import logging
import os
import pickle
import sys
import time

import nest
import numpy as np
from actor_crytic import POLL_TIME, PongNetDopa, PongNetRSTDP


class AIPong:
    def __init__(self, p1, p2):#, out_dir=""):
        """A class to run and store pong simulations of two competing spiking
        neural networks.

        Args:
            p1 (PongNet): Network to play on the left side.
            p2 (PongNet): Network to play on the right side.
            out_folder (str, optional): Name of the output folder. Defaults to
            current time stamp (YYYY-mm-dd-HH-MM-SS).
        """
        #self.game = pong.GameOfPong()
#        self.player1 = p1
        self.player2 = p2

    def run_games(self, max_runs=10000):
        """Runs a simulation of pong games and stores the results.

        Args:
            max_runs (int, optional): Number of iterations to simulate.
            Defaults to 10000.
        """
        start_time = time.time()
        self.run = 0
        biological_time = 0

        self.input_index = 0#self.game.ball.get_cell()[1]

        while self.run < max_runs:
            # input neuron in run (one iteration etc. ball position)
            self.input_index += 1
            if(self.input_index == 20):
                self.input_index = 0

            self.target_index = 1

            # set inputs for both players
#            self.player1.set_input_spiketrain(self.input_index, biological_time, self.target_index)
            self.player2.set_input_spiketrain(self.input_index, biological_time, self.target_index)

#            network1 = self.player1
            network2 = self.player2

#            print("Running simulation", self.run)
            # simulate for 200ms (1 iteration)
            nest.Simulate(POLL_TIME)
            # current time in iteration (ms)
            biological_time = nest.GetKernelStatus("biological_time")

            """
            spike_counts1 = network1.get_spike_counts()
            target_n_spikes1 = spike_counts1[self.target_index]
            # avoid zero division if none of the neurons fired.
            total_n_spikes1 = max(sum(spike_counts1), 1)

            activation_target1 = target_n_spikes1 / total_n_spikes1

            spike_counts2 = network2.get_spike_counts()
            target_n_spikes2 = spike_counts2[self.target_index]
            # avoid zero division if none of the neurons fired.
            total_n_spikes2 = max(sum(spike_counts2), 1)

            activation_target2 = target_n_spikes2 / total_n_spikes2
            print("Dopa:", activation_target1, "R-STDP:", activation_target2)
            """

#            self.player1.apply_synaptic_plasticity(biological_time)
#            self.player1.reset()
            self.player2.apply_synaptic_plasticity(biological_time)
            self.player2.reset()

            

            # after each iteration ... apply dopamine
            # network.winning_neuron ?= O1 ? ... apply reward
            """
            for network, paddle in zip([self.player1, self.player2], [self.game.l_paddle, self.game.r_paddle]):
                network.apply_synaptic_plasticity(biological_time)
                network.reset()

                position_diff = network.winning_neuron - paddle.get_cell()[1]
                if position_diff > 0:
                    paddle.move_up()
                elif position_diff == 0:
                    paddle.dont_move()
                else:
                    paddle.move_down()
            """

            self.run += 1

        end_time = time.time()


if __name__ == "__main__":
    nest.set_verbosity("M_WARNING")

    apply_noise_setting = 1

#    p1 = PongNetDopa(apply_noise_setting)
    p2 = PongNetRSTDP(apply_noise_setting)

    AIPong(None, p2).run_games(max_runs=10000)
