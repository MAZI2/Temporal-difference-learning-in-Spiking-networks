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


    def plot_network_activity(self, spike_records, weight_history, dopamine_history, poll_time=POLL_TIME):
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
            time_offset = iter_idx * poll_time
            for neuron_idx, spike_times in enumerate(spike_times_list):
                for t in spike_times:
                    all_spikes.append((neuron_idx, t + time_offset))

        if all_spikes:
            neuron_ids, spike_times = zip(*all_spikes)
        else:
            neuron_ids, spike_times = [], []

        # 2️⃣ Convert weights and dopamine to arrays
        weight_history = np.array(weight_history)  # shape: (iterations, num_input_neurons)
        dopamine_history = np.array(dopamine_history)  # shape: (iterations,)

        # 3️⃣ Plotting
        iterations = len(weight_history)
        time_axis = np.arange(iterations) * poll_time

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Raster plot
        axes[0].scatter(spike_times, neuron_ids, marker='.', color='black')
        axes[0].set_ylabel("Input neuron index")
        axes[0].set_title("Input neuron spikes (raster)")

        # Average weights plot
        for neuron_idx in range(weight_history.shape[1]):
            axes[1].plot(time_axis, weight_history[:, neuron_idx], label=f"N{neuron_idx}")
        axes[1].set_ylabel("Avg weight to striatum")
        axes[1].set_title("Average synaptic weights: input → striatum")
        axes[1].legend(loc='upper right', ncol=5, fontsize=8)

        # Dopamine current plot
        axes[2].plot(time_axis, dopamine_history, color='orange')
        axes[2].set_ylabel("Dopamine current amplitude")
        axes[2].set_xlabel("Time (ms)")
        axes[2].set_title("Dopamine level (reward signal)")

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
        dopamine_history = []

        while self.run < max_runs:
            if self.run == 1:
                self.player.reward = True

            self.input_index = self.state[0] * self.grid_size[1] + self.state[1]
            self.player.set_input_spiketrain(self.input_index, biological_time)

            logging.debug("Running simulation...")
            nest.Simulate(POLL_TIME)
            biological_time = nest.GetKernelStatus("biological_time")

            self.player.apply_synaptic_plasticity(biological_time)

            # ------------------------------
            # 1) Record input spikes (immediately after simulate)
            # ------------------------------
            generators = nest.NodeCollection(self.player.input_generators)
            spike_times_list = nest.GetStatus(generators, "spike_times")
            spike_records.append(spike_times_list)

            # ------------------------------
            # 2) Read weights input -> striatum (safe read each iteration)
            # ------------------------------
            x_offset = self.player.input_neurons[0].get("global_id")
            y_offset = self.player.striatum[0].get("global_id")


            weight_matrix = np.zeros((self.player.num_input_neurons, self.player.n_critic))
            conns = nest.GetConnections(self.player.input_neurons, self.player.striatum)
            #print(conns)
            for conn in conns:
                source, target, weight = conn.get(["source", "target", "weight"]).values()
                weight_matrix[source - x_offset, target - y_offset] = weight
            
            avg_weights = weight_matrix.mean(axis=1)
            weight_history.append(avg_weights.copy())
            
            # ------------------------------
            # 3) Record dopamine neuron spikes for this iteration
            # ------------------------------
            # Get spike times recorded on dopa_recorder (all time)
            dopa_events = nest.GetStatus(self.player.dopa_recorder, "events")[0]
            print("dopa events", dopa_events)
            dopa_times = np.array(dopa_events.get("times", []), dtype=float)

            # The iteration window is (biological_time - POLL_TIME, biological_time]
            t0 = biological_time - POLL_TIME
            t1 = biological_time
            # Count spikes that occurred in that window
            if dopa_times.size:
                mask = (dopa_times > t0) & (dopa_times <= t1)
                dopamine_history.append(int(mask.sum()))
            else:
                dopamine_history.append(0)

            # ------------------------------
            # 4) cleanup for next iteration
            # ------------------------------
            # Reset only generators' spike_times and the motor spike counters
            for g in self.player.input_generators:
                nest.SetStatus(g, {"spike_times": []})
            self.player.reset()  # clears motor spike recorders


            action = self.player.winning_neuron

            #print()
            #print("state", self.state)
            #print("action", action)

            #self.state, self.reward, self.done = self.game.step(action)
            self.state = (1, 0)
            self.done = False


            self.player.set_state(self.state)

            self.run += 1
            if self.done:
                print("Reached goal")
                self.done = False
                self.state = self.game.reset()
                self.player.set_state(self.state)

        print(weight_history)

        self.plot_network_activity(spike_records, weight_history, dopamine_history)
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
    runs=3

    AIGridworld().run_games(max_runs=runs)
