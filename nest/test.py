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
#SEED = 12301
SEED = 12333

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

import gridworld
from gridworld_ac import POLL_TIME, GridWorldAC


RUNS = 3000
class AIGridworld:
    def __init__(self):
        self.grid_size = (4, 4)
        self.start = (1, 2)
        self.goal = (3, 3)
        self.debug = False 
        self.loadWeights = False 

        self.done = False

        self.game = gridworld.GridWorld(size=self.grid_size, start=self.start, goal=self.goal)
        self.state = self.game.reset()
        self.player = GridWorldAC(False)
        if self.loadWeights:
            if os.path.exists("connections.pkl"):
                with open("connections.pkl", "rb") as f:
                    connections_data = pickle.load(f)
                self.player.load_saved_weights(connections_data)

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
                              weight_history_input5,
                                weight_history_input7,
                                weight_history_input6,
                                winning_history,
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

        fig, axes = plt.subplots(8, 1, figsize=(12, 20), sharex=True)


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

        axes[3].plot(time_points_str, weight_history_str[:, 0], label=f"N0")
        axes[3].plot(time_points_str, weight_history_str[:, 1], label=f"N1")
        axes[3].plot(time_points_str, weight_history_str[:, 2], label=f"N2")
        axes[3].plot(time_points_str, weight_history_str[:, 3], label=f"N3")
        axes[3].plot(time_points_str, weight_history_str[:, 4], label=f"N4")
        axes[3].plot(time_points_str, weight_history_str[:, 5], label=f"N5")
        axes[3].plot(time_points_str, weight_history_str[:, 6], label=f"N6")
        axes[3].plot(time_points_str, weight_history_str[:, 7], label=f"N7")
        axes[3].plot(time_points_str, weight_history_str[:, 8], label=f"N8")
        axes[3].plot(time_points_str, weight_history_str[:, 9], label=f"N9")
        axes[3].plot(time_points_str, weight_history_str[:, 10], label=f"N10")
        axes[3].plot(time_points_str, weight_history_str[:, 11], label=f"N11")
        axes[3].plot(time_points_str, weight_history_str[:, 12], label=f"N12")
        axes[3].plot(time_points_str, weight_history_str[:, 13], label=f"N13")
        axes[3].plot(time_points_str, weight_history_str[:, 14], label=f"N14")

        axes[3].set_ylabel("Avg weight to striatum")
        axes[3].set_title("Average synaptic weights: input → striatum")
        axes[3].legend(loc='upper right', ncol=5, fontsize=8)


        bin_size = 15.0           # ms
        bins = np.arange(0, time_axis[-1] + bin_size, bin_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        str_rates = self.compute_avg_firing_rate(str_spikes, num_neurons=self.player.n_critic, bins=bins, bin_size=bin_size)
        vp_rates = self.compute_avg_firing_rate(vp_spikes, num_neurons=self.player.n_critic, bins=bins, bin_size=bin_size)
        dopa_rates = self.compute_avg_firing_rate(dopa_spikes, num_neurons=self.player.n_critic, bins=bins, bin_size=bin_size)

        """
        axes[4].plot(bin_centers, str_rates, color='k')
        axes[4].set_ylabel("STR firing rate (Hz)")
        axes[4].set_title("Average STR activity")

        axes[5].plot(bin_centers, vp_rates, color='r')
        axes[5].set_ylabel("VP firing rate (Hz)")
        axes[5].set_title("Average VP activity")
        """

        axes[4].plot(bin_centers, dopa_rates, color='b')
        axes[4].set_ylabel("Dopa firing rate (Hz)")
        axes[4].set_xlabel("Time (ms)")
        axes[4].set_title("Average Dopa activity")

        # Raster plot
        axes[5].scatter(spike_times, neuron_ids, marker='.', color='black')
        axes[5].set_ylabel("Input neuron index")
        axes[5].set_title("Input neuron spikes (raster)")
        axes[5].set_yticks(np.arange(16))
        axes[5].set_ylim(-0.5, 15.5)
#        axes[5].set_yticks(np.arange(9))
#        axes[5].set_ylim(-0.5, 8.5)     
        axes[5].grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)


        axes[6].step(np.arange(len(winning_history)) * poll_time, winning_history, where='post', color='green')
        axes[6].set_ylabel("Winning neuron")
        axes[6].set_xlabel("Time (ms)")
        axes[6].set_title("Winning motor neuron per iteration")
        axes[6].set_yticks(np.arange(len(self.player.motor_neurons)))
        axes[6].set_ylim(-0.5, len(self.player.motor_neurons)-0.5)
        axes[6].grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)

        # raster plot
        motor_events = nest.GetStatus(self.player.motor_recorder, "events")[0]  # dictionary with 'senders' and 'times'
        motor_senders = motor_events['senders']
        motor_times = motor_events['times']

        motor_id_to_idx = {neuron.global_id: i for i, neuron in enumerate(self.player.motor_neurons)}
        motor_indices = np.array([motor_id_to_idx[s] for s in motor_senders])

        axes[7].scatter(motor_times, motor_indices, marker='.', color='green')
        axes[7].set_ylabel("Motor neuron")
        axes[7].set_title("Motor neuron spikes (raster)")
        axes[7].set_yticks(np.arange(len(self.player.motor_neurons)))
        axes[7].set_ylim(-0.5, len(self.player.motor_neurons)-0.5)
        axes[7].grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)

        def plot_raster(ax, recorder, neurons, color, title):
            events = nest.GetStatus(recorder, "events")[0]
            senders = events['senders']
            times = events['times']
            id_to_idx = {n.global_id: i for i, n in enumerate(neurons)}
            indices = np.array([id_to_idx[s] for s in senders if s in id_to_idx])
            ax.scatter(times, indices, marker='.', color=color)
            ax.set_ylabel("Neuron index")
            ax.set_title(title)
            ax.set_yticks(np.arange(len(neurons)))
            ax.set_ylim(-0.5, len(neurons) - 0.5)
            ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)


        # Copy previous plots into new figure
#        for i, ax_old in enumerate(fig.axes[:8]):
#            fig.axes[i] = ax_old

        # Append three raster plots
#        plot_raster(axes[8], self.player.intermediate_motor1_recorder, self.player.intermediate_motor1, "orange", "Intermediate Motor 1 spikes (raster)")
#        plot_raster(axes[9], self.player.intermediate_motor2_recorder, self.player.intermediate_motor2, "teal", "Intermediate Motor 2 spikes (raster)")
#        plot_raster(axes[10], self.player.intermediate_motor_recorder, self.player.intermediate_motor, "purple", "Intermediate Motor spikes (raster)")
        
        #
        # for ax in axes:
        #     ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)
        #     ax.set_xticks(np.arange(0, time_axis[-1] + 200, 10))  # vertical grid every 200 ms

        
        plt.tight_layout()
        plt.show()


    def run_games(self, max_runs=10000):
        best_str_sum = -np.inf
        best_connections_data = None
        

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

        time_points_str = []
        weight_history_str = []

        dopamine_history = []
        winning_history = []
        for local_idx, neuron in enumerate(self.player.motor_neurons):
            print(f"Local index: {local_idx}, Global ID: {neuron.global_id}")

        for local_idx, neuron in enumerate(self.player.intermediate_motor):
            print(f"Local index: {local_idx}, Global ID: {neuron.global_id}")



        while self.run < max_runs:
            """
            if REWARDED_STATES[self.run] == 1:
                self.player.reward = True
            else:
                self.player.reward = False
            """

            self.input_index = self.state[0] * self.grid_size[1] + self.state[1]
            self.player.set_input_spiketrain(self.input_index, biological_time)
            self.player.suppress_dopa(biological_time)

            logging.debug("Running simulation...")
            print("sumulating ", self.run)
            if self.debug:
                step_size = 10
            else:
                step_size = POLL_TIME
            """
            step_size = 1
            """
            for t in range(0, POLL_TIME, step_size):
                biological_time = nest.GetKernelStatus("biological_time")
                nest.Simulate(step_size)

                """
                if biological_time <= (self.run+1)*POLL_TIME:
                    self.player.get_action(biological_time, (self.run+1)*POLL_TIME)
                """

                                    #print("supp current", self.player.supp_current.amplitude)
                conns = nest.GetConnections(source=self.player.input_neurons, target=self.player.striatum)
                sources = np.array(conns.source)
                weights = np.array(conns.get("weight"))

                conns_motor = nest.GetConnections(source=self.player.intermediate_motor, target=self.player.motor_neurons)
                sources_motor = np.array(conns_motor.source)
                weights_motor = np.array(conns_motor.get("weight"))

                targets_motor = np.array(conns_motor.target)


                # compute mean per input neuron
                means_per_input = []
                means_per_input_motor = []
                for src in self.player.intermediate_motor:
                    mask = sources_motor == src.global_id
                    if np.any(mask):
                        means_per_input_motor.append(np.mean(weights_motor[mask]))
                    else:
                        means_per_input_motor.append(np.nan)

                for src in self.player.input_neurons:
                    mask = sources == src.global_id
                    if np.any(mask):
                        means_per_input.append(np.mean(weights[mask]))
                    else:
                        means_per_input.append(np.nan)

                weight_history_str.append(means_per_input)
                time_points_str.append(self.run * POLL_TIME + t + step_size)

                weight_history_motor.append(means_per_input_motor)
                time_points_motor.append(self.run * POLL_TIME + t + step_size)

                current_str_weights = np.array(weight_history_str[-1])
                current_str_sum = np.sum(current_str_weights)

                # Check if this is the best sum so far
                if current_str_sum > best_str_sum:
                    best_str_sum = current_str_sum
                    # Save current connections
                    best_connections_data = {
                        "input_to_motor": nest.GetConnections(
                            source=self.player.intermediate_motor,
                            target=self.player.motor_neurons
                        ).get(["source", "target", "weight"]),
                        "input_to_striatum": nest.GetConnections(
                            source=self.player.input_neurons,
                            target=self.player.striatum
                        ).get(["source", "target", "weight"]),
                    }
                    print(f"✅ New best striatum sum: {best_str_sum:.2f} at iteration {self.run}")


                if self.debug:
                    num_motor_neurons = len(self.player.motor_neurons)

                    # prepare per-target weight vectors
                    weights_input5 = np.full(num_motor_neurons, np.nan)
                    weights_input7 = np.full(num_motor_neurons, np.nan)
                    weights_input6 = np.full(num_motor_neurons, np.nan)

                    input_neuron1 = 11
                    input_neuron2 = 13
                    input_neuron3= 14

                    for idx, target in enumerate(self.player.motor_neurons):
                        mask5 = (sources_motor == self.player.intermediate_motor[input_neuron1].global_id) & (targets_motor == target.global_id)
                        mask7 = (sources_motor == self.player.intermediate_motor[input_neuron2].global_id) & (targets_motor == target.global_id)
                        mask6 = (sources_motor == self.player.intermediate_motor[input_neuron3].global_id) & (targets_motor == target.global_id)
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
                str_events = nest.GetStatus(self.player.str_recorder, "events")[0]

                # 'senders' contains the neuron IDs that emitted spikes
                neuron_ids = str_events['senders']

                #print("Striatum neuron IDs that fired spikes:", neuron_ids)


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

            start = self.run*POLL_TIME
            self.player.winning_neuron = self.player.get_max_activation(start, start+POLL_TIME)

            action = self.player.winning_neuron
            #action = self.player.action 
            winning_history.append(action)
            #print("winning neuron", action)

            
            print("prev", self.state)
            self.state, self.reward, self.done = self.game.step(action)
            #print("new", self.state)
            self.action = None

            if self.reward == 1.0:
                self.player.reward = True
            else:
                self.player.reward = False

            self.player.apply_synaptic_plasticity(biological_time, start, start+POLL_TIME)
            self.player.set_state(self.state)

            """
            decay_factor = 0.99998  # for example, keep 98% of previous weight each run

            conns = nest.GetConnections(source=self.player.intermediate_motor,
                                        target=self.player.motor_neurons)

            # Get all weights as a NumPy array for efficiency
            weights = np.array(conns.get("weight"))
            new_weights = weights * decay_factor

            # Apply back to NEST
            for conn, w_new in zip(conns, new_weights):
                conn.set({"weight": float(w_new)})
            """
            Wmin = 200.0
            Wmax = 1000.0

            k = 0.01  # maximum decay contribution

            conns = nest.GetConnections(source=self.player.intermediate_motor,
                                        target=self.player.motor_neurons)

            weights = np.array(conns.get("weight"), dtype=float)

            # Normalize weight position between 0…1
            alpha = (weights - Wmin) / (Wmax - Wmin)
            alpha = np.clip(alpha, 0.0, 1.0)  # ensure stability

            # Decay factor becomes smaller closer to Wmax
            decay_factor = 1.0 - k * alpha

            new_weights = weights * decay_factor

            # Apply back to NEST
            for conn, w_new in zip(conns, new_weights):
                conn.set({"weight": float(w_new)})

            # print(f"Applied decay factor {decay_factor} to intermediate_motor → motor weights")


            #self.state = NEXT_STATES[self.run]
            #self.done = False

            self.player.action = None


            self.run += 1
            if self.done:
                print("REACHED GOAL")
                self.done = False
                self.state = self.game.reset(random_start=True)
                self.player.set_state(self.state)

        #print(spike_records)
        #print(weight_history)

        if self.debug:
            self.plot_network_activity(spike_records, 
                                       weight_history_motor, 
                                       weight_history_str, 
                                       time_points_str, 
                                       time_points_motor, 
                                       dopa_events, 
                                       str_events, 
                                       vp_events,
                                       weight_history_input5,
                                        weight_history_input7,
                                        weight_history_input6,
                                        winning_history
                                       )
        end_time = time.time()

        if best_connections_data is not None:
            with open("best_connections.pkl", "wb") as f:
                pickle.dump(best_connections_data, f)
            print("✅ Saved best NEST connections to best_connections.pkl")
        

        if True:#not self.debug:
            connections_data = {}

            # Collect connections for key projections
            connections_data["input_to_motor"] = nest.GetConnections(source=self.player.intermediate_motor,
                                                                     target=self.player.motor_neurons).get(["source", "target", "weight"])
            connections_data["input_to_striatum"] = nest.GetConnections(source=self.player.input_neurons,
                                                                        target=self.player.striatum).get(["source", "target", "weight"])

            # Save with pickle
            with open("connections.pkl", "wb") as f:
                pickle.dump(connections_data, f)

            print("✅ Saved NEST connections to connections.pkl")


        # Convert lists to arrays

        # Convert lists to arrays
        weight_history_str_arr = np.array(weight_history_str)  # shape: (iterations, num_input_neurons)
        sum_weights_over_time_str = np.sum(weight_history_str_arr, axis=1)   # sum across input neurons

        weight_history_motor_arr = np.array(weight_history_motor)  # shape: (iterations, num_motor_neurons)
        # Number of input neurons connecting to motor neurons
        num_input_neurons_motor = weight_history_motor_arr.shape[1]

        # Create two vertical subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # --- Top plot: sum of input→striatum weights ---
        axes[0].plot(sum_weights_over_time_str, color='blue', label="Sum of input→striatum weights")
        axes[0].set_ylabel("Weight (pA)")
        axes[0].set_title("Input→Striatum weights over training")
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # --- Bottom plot: average weights per input neuron → motor neurons ---
        for neuron_idx in range(num_input_neurons_motor):
            axes[1].plot(weight_history_motor_arr[:, neuron_idx], label=f"Input neuron {neuron_idx}")

        axes[1].set_xlabel("Training iteration")
        axes[1].set_ylabel("Weight (pA)")
        axes[1].set_title("Input→Motor weights per input neuron")
        axes[1].legend(ncol=4, fontsize=8)
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()
                
                                


if __name__ == "__main__":
#    runs=len(NEXT_STATES)
    runs=RUNS

    AIGridworld().run_games(max_runs=runs)
