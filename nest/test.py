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

#[(0, 0) ...
NEXT_STATES = [(1, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
REWARDED_STATES = [1, 0, 0, 0, 0]
SEED = 12301

# reset kernel first (very important)
nest.ResetKernel()

# force single-threaded deterministic execution (recommended for debugging)
# if you want multi-threaded reproducibility you must ensure the same number
# of threads on each run and accept more complexity.
nest.SetKernelStatus({
    "rng_seed": SEED,
    "local_num_threads": 1
})



# also seed Python / NumPy RNGs (so any np.random or random calls are reproducible)
np.random.seed(SEED)
random.seed(SEED)
nest.set_verbosity("M_FATAL")

nest.Install("mymodule")

import gridworld
from gridworld_ac import POLL_TIME, GridWorldAC


RUNS = 150
class AIGridworld:
    def __init__(self):
        self.grid_size = (3, 3)
        self.start = (0, 0)
        self.goal = (2, 2)

        self.done = False

        self.game = gridworld.GridWorld(size=self.grid_size, start=self.start, goal=self.goal)
        self.state = self.game.reset()
        self.player = GridWorldAC(False)

        logging.info(f"setup complete for gridworld")


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
                              weight_history_str, 
                              time_points_str, 
                              time_points_motor, 
                              dopa_spikes, 
                              str_spikes, 
                              vp_spikes, 
                              poll_time=POLL_TIME):
        """
        Plot raster of input spikes, average weights to striatum, and dopamine signal.

        Args:
            spike_records: list of lists of spike times (per iteration, per neuron)
            weight_history: list of arrays of average weights per input neuron
            dopamine_history: list of dopamine amplitudes per iteration
            poll_time: duration of each iteration in ms
        """
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

        weight_history_str = np.array(weight_history_str)  # shape: (iterations, num_input_neurons)
        time_points_str = np.array(time_points_str)

        # 3️⃣ Plotting
        time_axis = np.arange(RUNS) * poll_time

        fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)


        # Average weights plot
        axes[0].plot(time_points_motor, weight_history_motor[:, 0], label=f"N0")
        axes[0].plot(time_points_motor, weight_history_motor[:, 1], label=f"N1")
        axes[0].plot(time_points_motor, weight_history_motor[:, 2], label=f"N2")
        axes[0].plot(time_points_motor, weight_history_motor[:, 3], label=f"N3")
        axes[0].plot(time_points_motor, weight_history_motor[:, 4], label=f"N4")
        axes[0].plot(time_points_motor, weight_history_motor[:, 5], label=f"N5")
        axes[0].plot(time_points_motor, weight_history_motor[:, 6], label=f"N6")
        axes[0].plot(time_points_motor, weight_history_motor[:, 7], label=f"N7")
        axes[0].plot(time_points_motor, weight_history_motor[:, 8], label=f"N8")
        axes[0].set_ylabel("Avg weight to motor")
        axes[0].set_title("Average synaptic weights: input → motor")
        axes[0].legend(loc='upper right', ncol=5, fontsize=8)

        axes[1].plot(time_points_str, weight_history_str[:, 0], label=f"N0")
        axes[1].plot(time_points_str, weight_history_str[:, 1], label=f"N1")
        axes[1].plot(time_points_str, weight_history_str[:, 2], label=f"N2")
        axes[1].plot(time_points_str, weight_history_str[:, 3], label=f"N3")
        axes[1].plot(time_points_str, weight_history_str[:, 4], label=f"N4")
        axes[1].plot(time_points_str, weight_history_str[:, 5], label=f"N5")
        axes[1].plot(time_points_str, weight_history_str[:, 6], label=f"N6")
        axes[1].plot(time_points_str, weight_history_str[:, 7], label=f"N7")
        axes[1].plot(time_points_str, weight_history_str[:, 8], label=f"N8")
        axes[1].set_ylabel("Avg weight to striatum")
        axes[1].set_title("Average synaptic weights: input → striatum")
        axes[1].legend(loc='upper right', ncol=5, fontsize=8)


        bin_size = 15.0           # ms
        bins = np.arange(0, time_axis[-1] + bin_size, bin_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        str_rates = self.compute_avg_firing_rate(str_spikes, num_neurons=self.player.n_critic, bins=bins, bin_size=bin_size)
        vp_rates = self.compute_avg_firing_rate(vp_spikes, num_neurons=self.player.n_critic, bins=bins, bin_size=bin_size)
        dopa_rates = self.compute_avg_firing_rate(dopa_spikes, num_neurons=self.player.n_critic, bins=bins, bin_size=bin_size)

        axes[2].plot(bin_centers, str_rates, color='k')
        axes[2].set_ylabel("STR firing rate (Hz)")
        axes[2].set_title("Average STR activity")

        axes[3].plot(bin_centers, vp_rates, color='r')
        axes[3].set_ylabel("VP firing rate (Hz)")
        axes[3].set_title("Average VP activity")

        axes[4].plot(bin_centers, dopa_rates, color='b')
        axes[4].set_ylabel("Dopa firing rate (Hz)")
        axes[4].set_xlabel("Time (ms)")
        axes[4].set_title("Average Dopa activity")

        # Raster plot
        axes[5].scatter(spike_times, neuron_ids, marker='.', color='black')
        axes[5].set_ylabel("Input neuron index")
        axes[5].set_title("Input neuron spikes (raster)")
        axes[5].set_yticks(np.arange(9))
        axes[5].set_ylim(-0.5, 8.5)
        axes[5].grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)

        for ax in axes:
            ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)
            ax.set_xticks(np.arange(0, time_axis[-1] + 200, 200))  # vertical grid every 200 ms

        
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

        time_points_str = []
        weight_history_str = []

        dopamine_history = []

        while self.run < max_runs:
            """
            if REWARDED_STATES[self.run] == 1:
                self.player.reward = True
            else:
                self.player.reward = False
            """

            self.input_index = self.state[0] * self.grid_size[1] + self.state[1]
            self.player.set_input_spiketrain(self.input_index, biological_time)

            logging.debug("Running simulation...")
            print("sumulating ", self.run)
            step_size = 10
            for t in range(0, POLL_TIME, step_size):
                nest.Simulate(step_size)

                conns = nest.GetConnections(source=self.player.input_neurons, target=self.player.striatum)
                sources = np.array(conns.source)
                weights = np.array(conns.get("weight"))

                conns_motor = nest.GetConnections(source=self.player.input_neurons, target=self.player.motor_neurons)
                sources_motor = np.array(conns_motor.source)
                weights_motor = np.array(conns_motor.get("weight"))


                # compute mean per input neuron
                means_per_input = []
                means_per_input_motor = []
                for src in self.player.input_neurons:
                    mask = sources == src.global_id
                    if np.any(mask):
                        means_per_input.append(np.mean(weights[mask]))
                    else:
                        means_per_input.append(np.nan)

                    mask = sources_motor == src.global_id
                    if np.any(mask):
                        means_per_input_motor.append(np.mean(weights_motor[mask]))
                    else:
                        means_per_input_motor.append(np.nan)

                weight_history_str.append(means_per_input)
                time_points_str.append(self.run * POLL_TIME + t + step_size)

                weight_history_motor.append(means_per_input_motor)
                time_points_motor.append(self.run * POLL_TIME + t + step_size)


            #nest.Simulate(POLL_TIME)

            str_events = nest.GetStatus(self.player.str_recorder, "events")[0]

            # 'senders' contains the neuron IDs that emitted spikes
            neuron_ids = str_events['senders']

            #print("Striatum neuron IDs that fired spikes:", neuron_ids)

            biological_time = nest.GetKernelStatus("biological_time")


            # Record input spikes
            generators = nest.NodeCollection(self.player.input_generators)
            spike_times_list = nest.GetStatus(generators, "spike_times")
            spike_records.append(spike_times_list)

            str_events = nest.GetStatus(self.player.str_recorder, "events")[0]
            vp_events = nest.GetStatus(self.player.vp_recorder, "events")[0]
            dopa_events = nest.GetStatus(self.player.dopa_recorder, "events")[0]

            # Cleanup for next iteration
            # Reset only generators' spike_times and the motor spike counters
            for g in self.player.input_generators:
                nest.SetStatus(g, {"spike_times": []})
            self.player.reset()  # clears motor spike recorders

            action = self.player.winning_neuron

            #print()
            #print("state", self.state)
            #print("action", action)

            self.state, self.reward, self.done = self.game.step(action)
            print(self.state)
            print(self.reward)
            #self.state = NEXT_STATES[self.run]
            #self.done = False

            if self.reward == 1.0:
                self.player.reward = True
            else:
                self.player.reward = False

            self.player.apply_synaptic_plasticity(biological_time)

            self.player.set_state(self.state)


            self.run += 1
            if self.done:
                print("REACHED GOAL")
                self.done = False
                self.state = self.game.reset()
                self.player.set_state(self.state)

        #print(spike_records)
        #print(weight_history)

        self.plot_network_activity(spike_records, 
                                   weight_history_motor, 
                                   weight_history_str, 
                                   time_points_str, 
                                   time_points_motor, 
                                   dopa_events, 
                                   str_events, 
                                   vp_events)
        end_time = time.time()

        weights = self.player.get_all_weights()
        policy = np.zeros((3, 3, 4))

        k = 0
        for i in range(3):
            for j in range(3):
                policy[i][j] = weights[k]/np.sum(weights[k])
                k += 1

        #self.game.plot_policy((3, 3), policy, start=(0,0), goal=self.goal)


if __name__ == "__main__":
#    runs=len(NEXT_STATES)
    runs=RUNS

    AIGridworld().run_games(max_runs=runs)
