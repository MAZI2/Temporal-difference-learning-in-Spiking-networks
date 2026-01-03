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

sys.path.append("../../pong-classes-singleplayer")
import pong

import nest
import numpy as np
import matplotlib.pyplot as plt

SEED = 12337

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

from rstdp_pong import POLL_TIME, PongNetRSTDP 

RUNS = 4000
class AIPongRSTDP:
    def __init__(self):
        self.debug = True 
        self.loadWeights = False

        self.state = 0
        self.player = PongNetRSTDP(True)
        self.game = pong.GameOfPong()

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
                              dopa_spikes,
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

        fig, axes = plt.subplots(figsize=(12,5))


        # Average weights plot
        weight_history_input5 = np.array(weight_history_input5)  # shape: (iterations, num_motor_neurons)
        weight_history_input7 = np.array(weight_history_input7)
        weight_history_input6 = np.array(weight_history_input6)

        num_motor = weight_history_input5.shape[1]
        motor_indices = np.arange(num_motor)

        # Add two more subplots (we’ll use axes[2] and axes[3])
        axes.set_title("Uteži sinaps med vhodnim nevronom 5 in izhodnimi nevroni")
        for j in range(num_motor):
            axes.plot(time_points_motor, weight_history_input5[:, j], label=f"Izhodni nevron {j}")
        axes.set_ylabel(r"$w_{\text{motor}}$", fontsize=12)
        axes.legend(fontsize=10, ncol=4, loc='upper left')
        axes.set_xlabel("Čas (ms)", fontsize=12)
        """

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
        """

        """
        axes[1].step(np.arange(len(winning_history)) * poll_time, winning_history, where='post', color='green')
        axes[1].set_ylabel("Winning neuron")
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_title("Winning motor neuron per iteration")
        axes[1].set_yticks(np.arange(len(self.player.motor_neurons)))
        axes[1].set_ylim(-0.5, len(self.player.motor_neurons)-0.5)
        axes[1].grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)

        # raster plot
        motor_events = nest.GetStatus(self.player.motor_recorder, "events")[0]  # dictionary with 'senders' and 'times'
        motor_senders = motor_events['senders']
        motor_times = motor_events['times']

        motor_id_to_idx = {neuron.global_id: i for i, neuron in enumerate(self.player.motor_neurons)}
        motor_indices = np.array([motor_id_to_idx[s] for s in motor_senders])

        axes[2].scatter(motor_times, motor_indices, marker='.', color='green')
        axes[2].set_ylabel("Motor neuron")
        axes[2].set_title("Motor neuron spikes (raster)")
        axes[2].set_yticks(np.arange(len(self.player.motor_neurons)))
        axes[2].set_ylim(-0.5, len(self.player.motor_neurons)-0.5)
        axes[2].grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)

        """

        # axes[1].scatter(spike_times, neuron_ids, marker='.', color='black')
        # axes[1].set_ylabel("Indeks stanja (vhodni nevron)")
        # axes[1].set_title("Stanje")
        # axes[1].set_yticks(np.arange(16))
        # axes[1].set_ylim(-0.5, 20.5)
        # axes[1].grid(True, which='both', axis='both', linestyle='--', linewidth=0.6, alpha=0.7)
        #
        #
        # bin_size = 15.0           # ms
        # bins = np.arange(0, time_axis[-1] + bin_size, bin_size)
        # bin_centers = (bins[:-1] + bins[1:]) / 2.0
        #
        # dopa_rates = self.compute_avg_firing_rate(dopa_spikes, num_neurons=8, bins=bins, bin_size=bin_size)
        #
        # axes[2].plot(bin_centers, dopa_rates, color='black')
        # axes[2].set_ylabel("Aktivnost (Hz)")
        # axes[2].set_xlabel("Čas (ms)")
        # axes[2].set_title("Povprečna aktivnost dopaminergičnih nevronov")



        plt.tight_layout()
        fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.2)
        plt.show()


    def run_games(self, max_runs=10000):
        self.game_data = []
        self.reward_history = []
        self.weight_snapshots = []

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

        dopamine_history = []


        winning_history = []
        for local_idx, neuron in enumerate(self.player.input_neurons):
            print(f"Local index: {local_idx}, Global ID: {neuron.global_id}")

        for local_idx, neuron in enumerate(self.player.motor_neurons):
            print(f"Local index: {local_idx}, Global ID: {neuron.global_id}")

        time_since_last_miss = 0
        survival_times = []

        while self.run < max_runs:
            """
            if REWARDED_STATES[self.run] == 1:
                self.player.reward = True
            else:
                self.player.reward = False
            """

            #self.input_index = self.state 
            ball_y_cell = self.game.ball.get_cell()[1]
            #self.player.set_input_spiketrain(self.input_index, biological_time)
            self.player.set_input_spiketrain(ball_y_cell, biological_time)

            logging.debug("Running simulation...")
            print("sumulating ", self.run)
            if self.debug:
                #step_size = 10
                step_size = POLL_TIME
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
                self.weight_snapshots.append(weight_history_motor[-1])

                if self.debug:
                    num_motor_neurons = len(self.player.motor_neurons)

                    # prepare per-target weight vectors
                    weights_input5 = np.full(num_motor_neurons, np.nan)
                    weights_input7 = np.full(num_motor_neurons, np.nan)
                    weights_input6 = np.full(num_motor_neurons, np.nan)

                    input_neuron1 = 5
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
                dopa_events = nest.GetStatus(self.player.dopa_recorder, "events")[0]


            # Cleanup for next iteration
            # Reset only generators' spike_times and the motor spike counters
            """
            Wmin = 1200.0
            Wmax = 2000.0

            k = 0.0001  # maximum decay contribution

            conns = nest.GetConnections(source=self.player.input_neurons,
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
            """


            # GAME PLAY

            self.player.winning_neuron = self.player.get_max_activation()
            action = self.player.winning_neuron
            winning_history.append(action)

            self.player.apply_synaptic_plasticity(biological_time)



            paddle_pos = self.game.l_paddle.get_cell()
            position_diff = self.player.winning_neuron - paddle_pos[1]
            print(position_diff)

            if position_diff > 0:
                self.game.l_paddle.move_up()
            elif position_diff == 0:
                self.game.l_paddle.dont_move()
            else:
                self.game.l_paddle.move_down()

            result = self.game.step()
            time_since_last_miss += POLL_TIME

            survival_times.append(time_since_last_miss)

            # Use pong reward
            if result == pong.RIGHT_SCORE:
                  # store survival time
                time_since_last_miss = 0  # reset
                # player missed
                self.player.reward = False
                self.game.reset_ball(towards_left=False)

            elif result == pong.GAME_CONTINUES:
                self.player.reward = None  # no reward

            else:  # LEFT_SCORE (unlikely now)
                self.player.reward = True
                self.game.reset_ball(True)

            self.reward_history.append(self.player.reward)

            self.game_data.append([
                self.game.ball.get_pos(),
                self.game.l_paddle.get_pos(),
                (None, None),           # placeholder for removed right paddle
                result                  # score event: 0, -1, +1
            ])


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
                                       dopa_events, 
                                       weight_history_input5,
                                        weight_history_input7,
                                        weight_history_input6,
                                        winning_history
                                       )

        end_time = time.time()

        save_obj = {}

        # A) Game data (converted to numpy)
        game_array = np.array(self.game_data, dtype=object)

        save_obj["ball_pos"]     = game_array[:, 0]
        save_obj["left_paddle"]  = game_array[:, 1]
        save_obj["right_paddle"] = game_array[:, 2]
        save_obj["score"]        = game_array[:, 3]

        # B) Network data
        save_obj["network"] = {
            "network_type": repr(self.player),
            "with_noise": self.player.apply_noise,
            "mean_rewards": np.array(self.player.mean_reward_history),
            "weights": np.array(self.weight_snapshots, dtype=object),
            "survival_times": np.array(survival_times)
        }

        # Write .pkl
        out_path = "experiment_output.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(save_obj, f)

        print(f"Saved training data to: {out_path}")

        final_weights = self.player.get_all_weights()  # shape (num_input_neurons, num_motor_neurons)

        # 2️⃣ Save weights to file
        import json
        weights_outfile = "final_weights.npy"
        np.save(weights_outfile, final_weights)
        print(f"Saved final weight matrix to {weights_outfile}")

        # 3️⃣ Print table of input → motor neuron connections
        print("\nFinal synaptic weights (Input → Motor Neurons):")
        header = ["Input\\Motor"] + [f"M{j}" for j in range(self.player.num_neurons)]
        print("\t".join(header))
        for i, row in enumerate(final_weights):
            row_str = "\t".join([f"{w:.1f}" for w in row])
            print(f"I{i}\t{row_str}")

        """
        num_inputs, num_motors = final_weights.shape
        x = np.arange(num_inputs)  # x positions for input neurons

        width = 0.04  # width of each bar
        fig, ax = plt.subplots(figsize=(12, 6))

        for m in range(num_motors):
            ax.bar(x + m*width, final_weights[:, m], width=width, label=f'M{m}')

        ax.set_xlabel("Input neurons")
        ax.set_ylabel("Weight")
        ax.set_title("Final synaptic weights (Input → Motor Neurons)")
        ax.set_xticks(x + width*(num_motors-1)/2)
        ax.set_xticklabels([f"I{i}" for i in range(num_inputs)])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
        plt.tight_layout()
        plt.show()
        """

        reward_history_array = np.array(self.player.mean_reward_history)  # shape: (iterations, num_neurons)

        # Compute mean across neurons for each iteration
        mean_reward_over_time = reward_history_array.mean(axis=1)

        survival_times = np.array(survival_times)
        print(survival_times)

        # Optional: rolling average with window=3 (or full cumulative average)
        cumulative_avg_survival = np.cumsum(survival_times) / np.arange(1, len(survival_times)+1)

        print(cumulative_avg_survival)
        # Plot
        reward_history_array = np.array(self.player.mean_reward_history)
        mean_reward_over_time = reward_history_array.mean(axis=1)

        survival_times = np.array(survival_times)
        cumulative_avg_survival = np.cumsum(survival_times) / np.arange(1, len(survival_times) + 1)

        # Time axis in ms
        iterations = np.arange(len(mean_reward_over_time))
        time_ms = iterations * POLL_TIME

        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()

        # Global mean reward
        ax1.plot(time_ms, mean_reward_over_time, color='blue', lw=2,
                label='Skupna povprečna nagrada')
        ax1.set_xlabel("Čas (ms)", fontsize=12)
        ax1.set_ylabel("Povprečna nagrada", color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Second y-axis
        ax2 = ax1.twinx()
        ax2.plot(time_ms, cumulative_avg_survival, color='red', lw=2,
                label='Povprečen čas preživetja')
        ax2.set_ylabel("Povprečen čas preživetja (ms)", color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

        plt.title("Skupna povprečna nagrada in povprečen čas preživetja")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
#    runs=len(NEXT_STATES)
    runs=RUNS

    AIPongRSTDP().run_games(max_runs=runs)
