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
import matplotlib.pyplot as plt

import gridworld
from gridworld_ac import POLL_TIME, GridWorldAC

NEXT_STATES = [(1, 0), (0, 0), (0, 0)]

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


    def plot_network_activity(self, spike_records, weight_history, weight_history_str, dopamine_history, striatum_vms, poll_time=POLL_TIME):
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


        # 3️⃣ Plotting
        iterations = len(weight_history)
        time_axis = np.arange(iterations) * poll_time

        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

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

        for neuron_idx in range(weight_history_str.shape[1]):
            axes[2].plot(time_axis, weight_history_str[:, neuron_idx], label=f"N{neuron_idx}")
        axes[2].set_ylabel("Avg weight to striatum")
        axes[2].set_title("Average synaptic weights: input → striatum")
        axes[2].legend(loc='upper right', ncol=5, fontsize=8)

        # Dopamine current plot
        dopa_times = np.array(dopamine_history[1], dtype=float)

        if dopa_times.size > 0:
            # Bin spikes by 1 ms (or smaller if you want higher temporal precision)
            bin_size = 1.0
            t_start, t_end = 0, time_axis[-1] + poll_time
            bins = np.arange(t_start, t_end + bin_size, bin_size)
            counts, _ = np.histogram(dopa_times, bins=bins)
            t = bins[:-1]

            axes[3].plot(t, counts, color='orange', drawstyle='steps-post')
        else:
            axes[3].text(0.5, 0.5, "No dopamine spikes", ha='center', va='center', transform=axes[2].transAxes)

        axes[3].set_ylabel("# Dopa neurons spiking")
        axes[3].set_xlabel("Time (ms)")
        axes[3].set_title("Dopamine neuron population activity")
        axes[3].set_xlim(0, time_axis[-1] + poll_time)


        # 4️⃣ Striatum membrane potentials
        senders = striatum_vms['senders']
        times = striatum_vms['times']
        V_m = striatum_vms['V_m']

        neuron_ids = np.unique(senders)

        for neuron_id in neuron_ids:
            mask = senders == neuron_id
            axes[4].plot(times[mask], V_m[mask], label=f"Neuron {neuron_id}")
        axes[4].set_ylabel("V_m (mV)")
        axes[4].set_xlabel("Time (ms)")
        axes[4].set_title("Striatum neuron membrane potentials")


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
        weight_history_str = []
        dopamine_history = []

        while self.run < max_runs:
            if self.run == 0:
                self.player.reward = True

            self.input_index = self.state[0] * self.grid_size[1] + self.state[1]
            self.player.set_input_spiketrain(self.input_index, biological_time)

            logging.debug("Running simulation...")
            print("sumulating ", self.run)
            nest.Simulate(POLL_TIME)
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
            
            # Dopamine neuron spikes for this iteration
            dopa_events = nest.GetStatus(self.player.dopa_recorder, "events")[0]
            dopa_times = np.array(dopa_events.get("times", []), dtype=float)

            
            dopamine_history.append(dopa_times)
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

            str_neuron_status = nest.GetStatus(self.player.str_multimeter, "events")[0]

            print("STRIATUM", str_events)
            print("VP", vp_events)

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

        self.plot_network_activity(spike_records, weight_history, weight_history_str, dopamine_history, str_neuron_status)
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
