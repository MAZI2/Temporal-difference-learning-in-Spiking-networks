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
SEED = 123330

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

VERBOSITY = "M_FATAL"


def _set_nest_verbosity(level=VERBOSITY):
    if hasattr(nest, "set_verbosity"):
        nest.set_verbosity(level)
    else:
        # Fallback for APIs exposing VerbosityLevel enum via nest.set(...)
        nest.set(verbosity=nest.VerbosityLevel.FATAL)


_set_nest_verbosity(VERBOSITY)

nest.Install("mymodule")

import gridworld
from gridworld_ac import POLL_TIME, GridWorldAC


RUNS = 10
class AIGridworld:
    def __init__(self, config):
        self.config = config

        self.grid_size = (4, 4)
        self.start = (1, 2)
        self.goal = (3, 3)
        self.debug = True
        self.loadWeights = False
        self.current_reset = (0, 0)
        self.step_count = 1

        self.done = False

        self.game = gridworld.GridWorld(size=self.grid_size, start=self.start, goal=self.goal)
        self.state = self.game.reset()
        self.player = GridWorldAC(config, False)

        if self.loadWeights:
            if os.path.exists("conns/0_4000_0.3_0_0_0-connections.pkl"):
                with open("conns/0_4000_0.3_0_0_0-connections.pkl", "rb") as f:
                    connections_data = pickle.load(f)
                self.player.load_saved_weights(connections_data)

        logging.info(f"setup complete for gridworld")

    def manhattan(self):
        return np.abs(self.current_reset[0]-self.goal[0])+np.abs(self.current_reset[1]-self.goal[1])



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
        time_axis = np.arange(len(spike_records)) * poll_time

        fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)

        # Input neuron activity per iteration
        iter_time = np.arange(len(spike_records)) * poll_time
        input_rates = np.zeros((2, len(spike_records)))
        for i, spike_times_list in enumerate(spike_records):
            for n in range(min(2, len(spike_times_list))):
                input_rates[n, i] = len(spike_times_list[n]) / (poll_time / 1000.0)

        axes[0].plot(iter_time, input_rates[0], color='green', label='Vhodni nevron 0')
        axes[0].plot(iter_time, input_rates[1], color='purple', label='Vhodni nevron 1')
        axes[0].set_title('Povprečna aktivnost vhodnih nevronov')
        axes[0].set_ylabel('Aktivnost (Hz)')
        axes[0].legend(loc='upper right')

        # Average weights input -> output (first two input neurons)
        axes[1].plot(time_points_motor, weight_history_motor[:, 0], color='C0', label='Vhodni nevron 0')
        axes[1].plot(time_points_motor, weight_history_motor[:, 1], color='C1', label='Vhodni nevron 1')
        axes[1].set_title('Povprečna utež sinaps med vhodnimi in izhodnimi nevroni')
        axes[1].set_ylabel(r'$\overline{w}_{\mathrm{in}\to\mathrm{in\_motor}}$')
        axes[1].legend(loc='upper right')

        # Average weights input -> striatum (first two input neurons)
        axes[2].plot(time_points_str, weight_history_str[:, 0], color='C0', label='Vhodni nevron 0')
        axes[2].plot(time_points_str, weight_history_str[:, 1], color='C1', label='Vhodni nevron 1')
        axes[2].set_title('Povprečna utež sinaps med vhodnimi nevroni in striatumom')
        axes[2].set_ylabel(r'$\overline{w}_{\mathrm{in}\to\mathrm{str}}$')
        axes[2].legend(loc='upper right')

        bin_size = 15.0  # ms
        bins = np.arange(0, time_axis[-1] + bin_size, bin_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        str_rates = self.compute_avg_firing_rate(str_spikes, num_neurons=self.player.n_critic, bins=bins, bin_size=bin_size)
        vp_rates = self.compute_avg_firing_rate(vp_spikes, num_neurons=self.player.n_critic, bins=bins, bin_size=bin_size)
        dopa_rates = self.compute_avg_firing_rate(dopa_spikes, num_neurons=self.player.n_critic, bins=bins, bin_size=bin_size)

        axes[3].plot(bin_centers, str_rates, color='black')
        axes[3].set_title('Povprečna aktivnost nevronov striatuma')
        axes[3].set_ylabel('Aktivnost STR (Hz)')

        axes[4].plot(bin_centers, vp_rates, color='red')
        axes[4].set_title('Povprečna aktivnost nevronov ventralnega palliduma')
        axes[4].set_ylabel('Aktivnost VP (Hz)')

        axes[5].plot(bin_centers, dopa_rates, color='blue')
        axes[5].set_title('Povprečna aktivnost dopaminergičnih nevronov')
        axes[5].set_ylabel('Aktivnost Dopa (Hz)')
        axes[5].set_xlabel('Čas (ms)')

        plt.tight_layout()

        os.makedirs("plots", exist_ok=True)
        out_pdf = f"plots/gridworld_network_activity_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf"
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
        print(f"Saved plot to {out_pdf}")

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
        # for local_idx, neuron in enumerate(self.player.motor_neurons):
        #     print(f"Local index: {local_idx}, Global ID: {neuron.global_id}")

        for local_idx, neuron in enumerate(self.player.striatum):
            print(f"Local index: {local_idx}, Global ID: {neuron.global_id}")

        step_count_history = []


        while self.run < max_runs:
            if self.run < len(REWARDED_STATES):
                self.player.reward = REWARDED_STATES[self.run] == 1

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
                spike_times_list = nest.GetStatus(self.player.input_generators, "spike_times")
                spike_records.append(spike_times_list)

                str_events = nest.GetStatus(self.player.str_recorder, "events")[0]
                vp_events = nest.GetStatus(self.player.vp_recorder, "events")[0]
                dopa_events = nest.GetStatus(self.player.dopa_recorder, "events")[0]



            # Cleanup for next iteration
            # Reset only generators' spike_times and the motor spike counters

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

            for g in self.player.input_generators:
                nest.SetStatus(g, {"spike_times": []})
            self.player.reset()  # clears motor spike recorders


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
            # Wmin = 1200.0
            # Wmax = 4000.0
            #
            # k = 0.0001  # maximum decay contribution
            #
            # conns = nest.GetConnections(source=self.player.intermediate_motor,
            #                             target=self.player.motor_neurons)
            #
            # weights = np.array(conns.get("weight"), dtype=float)
            #
            # # Normalize weight position between 0…1
            # alpha = (weights - Wmin) / (Wmax - Wmin)
            # alpha = np.clip(alpha, 0.0, 1.0)  # ensure stability
            #
            # # Decay factor becomes smaller closer to Wmax
            # decay_factor = 1.0 - k * alpha
            #
            # new_weights = weights * decay_factor
            #
            # # Apply back to NEST
            # for conn, w_new in zip(conns, new_weights):
            #     conn.set({"weight": float(w_new)})

            # print(f"Applied decay factor {decay_factor} to intermediate_motor → motor weights")


            if self.run < len(NEXT_STATES):
                self.state = NEXT_STATES[self.run]
                self.done = False

            self.player.action = None
            self.step_count += 1


            self.run += 1
            if self.done:
                print("REACHED GOAL")
                self.done = False
                self.state = self.game.reset(random_start=True)

                self.player.set_state(self.state)
                step_count_history.append((self.step_count/self.manhattan() ,biological_time/POLL_TIME))

                self.current_reset = self.state
                self.step_count = 0

        #print(spike_records)
        #print(weight_history)

        print(step_count_history)

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
            with open("conns/" + "_".join(str(v) for v in config.values()) + "-connections.pkl", "wb") as f:
                pickle.dump(best_connections_data, f)
            print("✅ Saved best NEST connections to best_connections.pkl")
        

        """
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
        """


        # Convert lists to arrays

        # Convert lists to arrays
        # Convert weight histories
        # Convert weight histories
        weight_history_str_arr = np.array(weight_history_str)
        sum_weights_over_time_str = np.sum(weight_history_str_arr, axis=1)

        weight_history_motor_arr = np.array(weight_history_motor)
        num_input_neurons_motor = weight_history_motor_arr.shape[1]

        # Convert step_count_history → arrays
        relative_steps_arr = np.array([float(x[0]) for x in step_count_history])
        bio_time_arr      = np.array([float(x[1]) for x in step_count_history])

        # Compute cumulative average relative step time (like survival time)
        cumulative_avg_steps = np.cumsum(relative_steps_arr) / np.arange(1, len(relative_steps_arr)+1)

        # Create plots
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

        # ==========================================================
        # --- TOP PLOT: weights + cumulative step time (two y-axes) ---
        # ==========================================================


        ax1.plot(sum_weights_over_time_str, color='blue', lw=2, label="Average weight of all connections between input neurons and striatum")
        ax1.set_ylabel(r"$\mu_{w_{\text{in}\to\text{str}}}$", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_title("Average weight of all connections between input neurons and striatum vs average relative number of steps")

        # Right axis for cumulative average relative step time
        ax2 = ax1.twinx()
        ax2.plot(bio_time_arr, cumulative_avg_steps, color='red', lw=2,
                label="Average relative number of steps")
        ax2.set_ylabel("Average relative number of steps", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.set_xlabel("Iteration")

        # ==========================================================
        # --- BOTTOM PLOT: Motor weights ---
        # ==========================================================

        # for neuron_idx in range(num_input_neurons_motor):
        #     axes[1].plot(weight_history_motor_arr[:, neuron_idx], label=f"Input neuron {neuron_idx}")
        #
        # axes[1].set_xlabel("Training iteration")
        # axes[1].set_ylabel("Weight (pA)")
        # axes[1].set_title("Input→Motor weights per input neuron")
        # axes[1].legend(ncol=4, fontsize=8)
        # axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()

        os.makedirs("learns", exist_ok=True)
        plt.savefig("learns/" + "_".join(str(v) for v in config.values()) + "-learn.png")

        os.makedirs("plots", exist_ok=True)
        out_pdf = f"plots/gridworld_learning_summary_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf"
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
        print(f"Saved plot to {out_pdf}")

        #plt.show()


                
                                

def nest_set_seed(seed):
    nest.ResetKernel()

    nest.SetKernelStatus({
        "rng_seed": seed,
    })

    np.random.seed(seed)
    random.seed(seed)
    _set_nest_verbosity(VERBOSITY)

    nest.Install("mymodule")

if __name__ == "__main__":
#
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", type=str, default=VERBOSITY, help="NEST verbosity level, e.g. M_FATAL")
    args = parser.parse_args()

    VERBOSITY = args.verbosity
    _set_nest_verbosity(VERBOSITY)

    runs = len(NEXT_STATES) if NEXT_STATES else RUNS

    """
    noise_rates = [1, 2, 3, 4]
    w_c_a_maxs = [1000, 2000]
    a_plus_minuss = [0.01, 0.015, 0.02, 0.03]
    w_ex_in_alls = [(0,0), (15,-10)]
    seed_iters = 1
    """
    noise_rates = [0]
    w_c_a_maxs = [4000]
    a_plus_minuss = [0.3]
    w_ex_in_alls = [(0,0)]
    seed_iters = 1


    config = {}
    for noise_rate in noise_rates:
        config["noise_rate"] = noise_rate
        for w_c_a_max in w_c_a_maxs:
            config["w_c_a_max"] = w_c_a_max
            for a_plus_minus in a_plus_minuss:
                config["a_plus_minus"] = a_plus_minus 
                for w_ex_all, w_in_all in w_ex_in_alls:
                    config["w_ex_all"] = w_ex_all
                    config["w_in_all"] = w_in_all

                    seed = SEED
                    for i in range(seed_iters):
                        config["iter"] = i
                        nest_set_seed(seed)
                        seed += i

                        AIGridworldI = AIGridworld(config) 
                        AIGridworldI.run_games(max_runs=runs)
