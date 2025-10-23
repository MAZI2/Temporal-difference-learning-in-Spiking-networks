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

import nest
import numpy as np
import matplotlib.pyplot as plt

import gridworld
from gridworld_ac import POLL_TIME, GridWorldAC
#[(0, 0) ...
NEXT_STATES = [(1, 0), (0, 0), (0, 0), (0, 0)]
REWARDED_STATES = [0, 0, 0, 0]

class AIGridworld:
    def __init__(self):
        self.grid_size = (5, 5)
        self.start = (0, 0)
        self.goal = (4, 4)

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

    def plot_network_activity(self, spike_records, weight_history, weight_history_str, time_points_str, dopa_spikes, str_spikes, vp_spikes, poll_time=POLL_TIME):
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
        weight_history = np.array(weight_history)  # shape: (iterations, num_input_neurons)
        weight_history = weight_history[:, [0, 5, 10]]

        weight_history_str = np.array(weight_history_str)  # shape: (iterations, num_input_neurons)
        time_points_str = np.array(time_points_str)

        # 3️⃣ Plotting
        iterations = len(weight_history)
        time_axis = np.arange(iterations) * poll_time

        fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

        # Raster plot
        axes[0].scatter(spike_times, neuron_ids, marker='.', color='black')
        axes[0].set_ylabel("Input neuron index")
        axes[0].set_title("Input neuron spikes (raster)")

        # Average weights plot
        for neuron_idx in range(weight_history.shape[1]):
            axes[1].plot(time_axis, weight_history[:, neuron_idx], label=f"N{neuron_idx}")
        axes[1].set_ylabel("Avg weight to motor")
        axes[1].set_title("Average synaptic weights: input → motor")
        axes[1].legend(loc='upper right', ncol=5, fontsize=8)

        axes[2].plot(time_points_str, weight_history_str[:, 0], label=f"N0")
        axes[2].plot(time_points_str, weight_history_str[:, 5], label=f"N5")

        axes[2].set_ylabel("Avg weight to striatum")
        axes[2].set_title("Average synaptic weights: input → striatum")
        axes[2].legend(loc='upper right', ncol=5, fontsize=8)


        bin_size = 5.0           # ms
        bins = np.arange(0, time_axis[-1] + bin_size, bin_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        str_rates = self.compute_avg_firing_rate(str_spikes, num_neurons=self.player.n_critic, bins=bins, bin_size=bin_size)
        vp_rates = self.compute_avg_firing_rate(vp_spikes, num_neurons=self.player.n_critic, bins=bins, bin_size=bin_size)
        dopa_rates = self.compute_avg_firing_rate(dopa_spikes, num_neurons=self.player.n_critic, bins=bins, bin_size=bin_size)

        axes[3].plot(bin_centers, str_rates, color='k')
        axes[3].set_ylabel("STR firing rate (Hz)")
        axes[3].set_title("Average STR activity")

        axes[4].plot(bin_centers, vp_rates, color='r')
        axes[4].set_ylabel("VP firing rate (Hz)")
        axes[4].set_title("Average VP activity")

        axes[5].plot(bin_centers, dopa_rates, color='b')
        axes[5].set_ylabel("Dopa firing rate (Hz)")
        axes[5].set_xlabel("Time (ms)")
        axes[5].set_title("Average Dopa activity")
        
        plt.tight_layout()
        plt.show()


    def run_games(self, max_runs=10000):
        start_time = time.time()
        self.run = 0
        biological_time = 0

        logging.info(f"Starting simulation of {max_runs} iterations of " f"{POLL_TIME}ms each.")

        # 1 state transition
        spike_records = []
        weight_history = []
        time_points_str = []
        weight_history_str = []
        dopamine_history = []

        while self.run < max_runs:
            if REWARDED_STATES[self.run] == 1:
                self.player.reward = True
            else:
                self.player.reward = False

            if self.run == 0:
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

                # compute mean per input neuron
                means_per_input = []
                for src in self.player.input_neurons:
                    mask = sources == src.global_id
                    if np.any(mask):
                        means_per_input.append(np.mean(weights[mask]))
                    else:
                        means_per_input.append(np.nan)

                weight_history_str.append(means_per_input)
                time_points_str.append(self.run * POLL_TIME + t + step_size)


            #nest.Simulate(POLL_TIME)

            str_events = nest.GetStatus(self.player.str_recorder, "events")[0]

            # 'senders' contains the neuron IDs that emitted spikes
            neuron_ids = str_events['senders']

            print("Striatum neuron IDs that fired spikes:", neuron_ids)

            biological_time = nest.GetKernelStatus("biological_time")

            self.player.apply_synaptic_plasticity(biological_time)

            # Record input spikes
            generators = nest.NodeCollection(self.player.input_generators)
            spike_times_list = nest.GetStatus(generators, "spike_times")
            spike_records.append(spike_times_list)

            # Weights input -> motor 
            x_offset = self.player.input_neurons[0].get("global_id")
            y_offset = self.player.motor_neurons[0].get("global_id")

            weight_matrix = np.zeros((self.player.num_input_neurons, self.player.num_output_neurons))
            conns = nest.GetConnections(self.player.input_neurons, self.player.motor_neurons)
            #print(conns)
            for conn in conns:
                source, target, weight = conn.get(["source", "target", "weight"]).values()
                weight_matrix[source - x_offset, target - y_offset] = weight
            
            avg_weights = weight_matrix.mean(axis=1)
            weight_history.append(avg_weights.copy())

            # Weights input -> striatum
            """
            x_offset = self.player.input_neurons[0].get("global_id")
            y_offset = self.player.striatum[0].get("global_id")

            weight_matrix_str = np.zeros((self.player.num_input_neurons, self.player.n_critic))
            conns = nest.GetConnections(self.player.input_neurons, self.player.striatum)
            #print(conns)
            for conn in conns:
                source, target, weight = conn.get(["source", "target", "weight"]).values()
                weight_matrix_str[source - x_offset, target - y_offset] = weight
            
            avg_weights = weight_matrix_str.mean(axis=1)
            weight_history_str.append(avg_weights.copy())
            """
            
            """
            # The iteration window is (biological_time - POLL_TIME, biological_time]
            t0 = biological_time - POLL_TIME
            t1 = biological_time
            # Count spikes that occurred in that window
            if dopa_times.size:
                mask = (dopa_times > t0) & (dopa_times <= t1)
                dopamine_history.append(int(mask.sum()))
            else:
                dopamine_history.append(0)
            """

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

            #self.state, self.reward, self.done = self.game.step(action)
            self.state = NEXT_STATES[self.run]
            self.done = False


            self.player.set_state(self.state)

            self.run += 1
            if self.done:
                print("Reached goal")
                self.done = False
                self.state = self.game.reset()
                self.player.set_state(self.state)

        #print(spike_records)
        #print(weight_history)

        self.plot_network_activity(spike_records, weight_history, weight_history_str, time_points_str, dopa_events, str_events, vp_events)
        end_time = time.time()

        weights = self.player.get_all_weights()
        policy = np.zeros((5, 5, 4))

        k = 0
        for i in range(5):
            for j in range(5):
                policy[i][j] = weights[k]/np.sum(weights[k])
                k += 1

        #self.game.plot_policy((5, 5), policy, start=(0,0), goal=self.goal)


if __name__ == "__main__":
    runs=len(NEXT_STATES)

    AIGridworld().run_games(max_runs=runs)
