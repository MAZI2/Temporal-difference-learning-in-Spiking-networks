import argparse
import datetime
import gzip
import logging
import os
import pickle
import sys
import time
import io
import re
from contextlib import contextmanager
import random

import nest
import numpy as np
import matplotlib.pyplot as plt

SEED = 12335

# reset kernel first (very important)
nest.ResetKernel()

# force single-threaded deterministic execution (recommended for debugging)
# if you want multi-threaded reproducibility you must ensure the same number
# of threads on each run and accept more complexity.
nest.SetKernelStatus({
    "rng_seed": SEED,
})



# also seed Python / NumPy RNGs (so any np.random or random calls are reproducible)
np.random.seed(SEED)
random.seed(SEED)
nest.set_verbosity("M_FATAL")

nest.Install("mymodule")

from rstdp_example import POLL_TIME, PongNetRSTDP 

RUNS = 10
class AIGridworldRSTDP:
    def __init__(self):
        self.debug = True 
        self.loadWeights = False

        self.state = 0
        self.player = PongNetRSTDP(True)

    def compute_avg_firing_rate(self, spike_events, num_neurons, bins, bin_size):
        """
        spike_events: dictionary from NEST, keys 'senders' and 'times'
        num_neurons: number of neurons in this population
        bins: global bin edges
        """
        senders = spike_events['senders']
        times = spike_events['times']

        if len(times) == 0:
            return np.zeros(len(bins)-1)

        all_counts = np.zeros((num_neurons, len(bins)-1))
        neuron_ids = np.unique(senders)

        for i, neuron in enumerate(neuron_ids):
            mask = senders == neuron
            counts, _ = np.histogram(times[mask], bins=bins)
            all_counts[i, :] = counts

        # Convert to firing rate in Hz: spikes / neuron / second
        rates = all_counts.mean(axis=0) / (bin_size / 1000.0)
        return rates

    def plot_network_activity(self,
                              spike_records,
                              weight_history_motor,
                              time_points_motor,
                              weight_history_input5,
                                weight_history_input7,
                                weight_history_input6,
                                winning_history,
                              poll_time=POLL_TIME):
        # 1️⃣ Convert spike data into global raster points
        all_spikes = []
        for iter_idx, spike_times_list in enumerate(spike_records):
            #time_offset = iter_idx * poll_time
            time_offset = 0
            for neuron_idx, spike_times in enumerate(spike_times_list):
                for t in spike_times:
                    all_spikes.append((neuron_idx, t + time_offset))

        if all_spikes:
            neuron_ids, spike_times = zip(*all_spikes)
        else:
            neuron_ids, spike_times = [], []

        # 2️⃣ Convert weights and dopamine to arrays
        weight_history_motor = np.array(weight_history_motor)  # shape: (iterations, num_input_neurons)
        time_points_motor = np.array(time_points_motor)

        # 3️⃣ Plotting
        time_axis = np.arange(RUNS) * poll_time

        fig, axes = plt.subplots(6, 1, figsize=(12, 20), sharex=True)


        # Average weights plot
        weight_history_input5 = np.array(weight_history_input5)  # shape: (iterations, num_motor_neurons)
        weight_history_input7 = np.array(weight_history_input7)
        weight_history_input6 = np.array(weight_history_input6)

        num_motor = weight_history_input5.shape[1]
        motor_indices = np.arange(num_motor)

        # Add two more subplots (we’ll use axes[2] and axes[3])
        axes[0].set_title("Weights from input neuron 11 → motor neurons")
        for j in range(num_motor):
            axes[0].plot(time_points_motor, weight_history_input5[:, j], label=f"Motor {j}")
        axes[0].set_ylabel("Weight (pA)")
        axes[0].legend(fontsize=7, ncol=4)

        axes[1].set_title("Weights from input neuron 13 → motor neurons")
        for j in range(num_motor):
            axes[1].plot(time_points_motor, weight_history_input7[:, j], label=f"Motor {j}")
        axes[1].set_ylabel("Weight (pA)")
        axes[1].legend(fontsize=7, ncol=4)

        axes[2].set_title("Weights from input neuron 14 → motor neurons")
        for j in range(num_motor):
            axes[2].plot(time_points_motor, weight_history_input6[:, j], label=f"Motor {j}")
        axes[2].set_ylabel("Weight (pA)")
        axes[2].legend(fontsize=7, ncol=4)

        # Raster plot
        axes[3].scatter(spike_times, neuron_ids, marker='.', color='black')
        axes[3].set_ylabel("Input neuron index")
        axes[3].set_title("Input neuron spikes (raster)")
        axes[3].set_yticks(np.arange(3))
        axes[3].set_ylim(-0.5, 2.5)
        axes[3].grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)

        axes[4].step(np.arange(len(winning_history)) * poll_time, winning_history, where='post', color='green')
        axes[4].set_ylabel("Winning neuron")
        axes[4].set_xlabel("Time (ms)")
        axes[4].set_title("Winning motor neuron per iteration")
        axes[4].set_yticks(np.arange(len(self.player.motor_neurons)))
        axes[4].set_ylim(-0.5, len(self.player.motor_neurons)-0.5)
        axes[4].grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)

        # raster plot
        motor_events = nest.GetStatus(self.player.motor_recorder, "events")[0]  # dictionary with 'senders' and 'times'
        motor_senders = motor_events['senders']
        motor_times = motor_events['times']

        motor_id_to_idx = {neuron.global_id: i for i, neuron in enumerate(self.player.motor_neurons)}
        motor_indices = np.array([motor_id_to_idx[s] for s in motor_senders])

        axes[5].scatter(motor_times, motor_indices, marker='.', color='green')
        axes[5].set_ylabel("Motor neuron")
        axes[5].set_title("Motor neuron spikes (raster)")
        axes[5].set_yticks(np.arange(len(self.player.motor_neurons)))
        axes[5].set_ylim(-0.5, len(self.player.motor_neurons)-0.5)
        axes[5].grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)

        plt.tight_layout()
        plt.show()


    def run_games(self, max_runs=10000):
        start_time = time.time()
        self.run = 0
        biological_time = 0

        logging.info(f"Starting simulation of {max_runs} iterations of " f"{POLL_TIME}ms each.")

        # 1 state transition
        spike_records = []

        time_points_motor = []
        weight_history_motor = []

        weight_history_input5 = []
        weight_history_input7 = []
        weight_history_input6 = []

        winning_history = []
        for local_idx, neuron in enumerate(self.player.input_neurons):
            print(f"Local index: {local_idx}, Global ID: {neuron.global_id}")

        for local_idx, neuron in enumerate(self.player.motor_neurons):
            print(f"Local index: {local_idx}, Global ID: {neuron.global_id}")


        while self.run < max_runs:
            """
            if REWARDED_STATES[self.run] == 1:
                self.player.reward = True
            else:
                self.player.reward = False
            """

            self.input_index = self.state 
            self.player.set_input_spiketrain(self.input_index, biological_time)

            logging.debug("Running simulation...")
            print("sumulating ", self.run)
            if self.debug:
                step_size = 10
            else:
                step_size = POLL_TIME

            for t in range(0, POLL_TIME, step_size):
                biological_time = nest.GetKernelStatus("biological_time")
                nest.Simulate(step_size)

                conns_motor = nest.GetConnections(source=self.player.input_neurons, target=self.player.motor_neurons)
                sources_motor = np.array(conns_motor.source)
                weights_motor = np.array(conns_motor.get("weight"))

                targets_motor = np.array(conns_motor.target)

                # compute mean per input neuron
                means_per_input_motor = []
                for src in self.player.input_neurons:
                    mask = sources_motor == src.global_id
                    if np.any(mask):
                        means_per_input_motor.append(np.mean(weights_motor[mask]))
                    else:
                        means_per_input_motor.append(np.nan)

                weight_history_motor.append(means_per_input_motor)
                time_points_motor.append(self.run * POLL_TIME + t + step_size)


                if self.debug:
                    num_motor_neurons = len(self.player.motor_neurons)

                    # prepare per-target weight vectors
                    weights_input5 = np.full(num_motor_neurons, np.nan)
                    weights_input7 = np.full(num_motor_neurons, np.nan)
                    weights_input6 = np.full(num_motor_neurons, np.nan)

                    input_neuron1 = 0
                    input_neuron2 = 1
                    input_neuron3 = 2

                    for idx, target in enumerate(self.player.motor_neurons):
                        mask5 = (sources_motor == self.player.input_neurons[input_neuron1].global_id) & (targets_motor == target.global_id)
                        mask7 = (sources_motor == self.player.input_neurons[input_neuron2].global_id) & (targets_motor == target.global_id)
                        mask6 = (sources_motor == self.player.input_neurons[input_neuron3].global_id) & (targets_motor == target.global_id)
                        if np.any(mask5):
                            weights_input5[idx] = weights_motor[mask5][0]
                        if np.any(mask7):
                            weights_input7[idx] = weights_motor[mask7][0]
                        if np.any(mask6):
                            weights_input6[idx] = weights_motor[mask6][0]

                    weight_history_input5.append(weights_input5)
                    weight_history_input7.append(weights_input7)
                    weight_history_input6.append(weights_input6)

            biological_time = nest.GetKernelStatus("biological_time")

            if self.debug:
                # Record input spikes
                generators = nest.NodeCollection(self.player.input_generators)
                spike_times_list = nest.GetStatus(generators, "spike_times")
                spike_records.append(spike_times_list)

            # Cleanup for next iteration
            # Reset only generators' spike_times and the motor spike counters

            self.player.winning_neuron = self.player.get_max_activation()
            action = self.player.winning_neuron
            winning_history.append(action)

            self.player.apply_synaptic_plasticity(biological_time)
            self.state = random.randint(0, 2)
            self.player.set_state(self.state)
            for g in self.player.input_generators:
                nest.SetStatus(g, {"spike_times": []})
            self.player.reset()  # clears motor spike recorders



            self.run += 1
            
        if self.debug:
            self.plot_network_activity(spike_records,
                                       weight_history_motor,
                                       time_points_motor,
                                       weight_history_input5,
                                        weight_history_input7,
                                        weight_history_input6,
                                        winning_history
                                       )
        end_time = time.time()
        

if __name__ == "__main__":
#    runs=len(NEXT_STATES)
    runs=RUNS

    AIGridworldRSTDP().run_games(max_runs=runs)
