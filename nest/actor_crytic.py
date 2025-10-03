r"""Classes to encapsulate the neuronal networks.
----------------------------------------------------------------
PongNetDopa uses the actor-critic model described in [2]_ to determine the
amount of reward to send to the dopaminergic synapses between input and motor
neurons. In this framework, the motor neurons represent the actor, while a
secondary network of three populations (termed striatum, VP, and dopaminergic
neurons) form the critic which modulates dopamine concentration based on
temporal difference error.
"""

import logging
from abc import ABC, abstractmethod
from copy import copy

import nest
import numpy as np

# Simulation time per iteration in milliseconds.
POLL_TIME = 200
# Number of spikes in an input spiketrain per iteration.
N_INPUT_SPIKES = 20
# Inter-spike interval of the input spiketrain.
ISI = 10.0
# Standard deviation of Gaussian current noise in picoampere.
BG_STD = 220.0
# Reward to be applied depending on distance to target neuron.

# TODO
REWARDS_DICT = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}


class PongNet(ABC):
    def __init__(self, apply_noise=True, num_neurons=20):
        """Abstract base class for network wrappers that learn to play pong.
        Parts of the network that are required for both types of inheriting
        class are created here. Namely, spike_generators and their connected
        parrot_neurons, which serve as input, as well as iaf_psc_exp neurons
        and their corresponding spike_recorders which serve as output. The
        connection between input and output is not established here because it
        is dependent on the plasticity rule used.

        Args:
            num_neurons (int, optional): Number of neurons in both the input and
            output layer. Changes here need to be matched in the game
            simulation in pong.py. Defaults to 20.
            apply_noise (bool, optional): If True, Poisson noise is applied
            to the motor neurons of the network. Defaults to True.
        """
        self.apply_noise = apply_noise
        self.num_neurons = num_neurons

        self.weight_history = []
        self.mean_reward = np.array([0.0 for _ in range(self.num_neurons)])
        self.mean_reward_history = []
        self.winning_neuron = 0

        self.input_generators = nest.Create("spike_generator", self.num_neurons)
        self.input_neurons = nest.Create("parrot_neuron", self.num_neurons)
        nest.Connect(self.input_generators, self.input_neurons, {"rule": "one_to_one"})

        self.motor_neurons = nest.Create("iaf_psc_exp", self.num_neurons)
        self.spike_recorders = nest.Create("spike_recorder", self.num_neurons)
        nest.Connect(self.motor_neurons, self.spike_recorders, {"rule": "one_to_one"})

    def get_all_weights(self):
        """Returns all synaptic weights between input and motor neurons.

        Returns:
            numpy.array: 2D array of shape (n_neurons, n_neurons). Input
            neurons are on the first axis, motor neurons on the second axis.
        """
        x_offset = self.input_neurons[0].get("global_id")
        y_offset = self.motor_neurons[0].get("global_id")
        weight_matrix = np.zeros((self.num_neurons, self.num_neurons))
        conns = nest.GetConnections(self.input_neurons, self.motor_neurons)
        for conn in conns:
            source, target, weight = conn.get(["source", "target", "weight"]).values()
            weight_matrix[source - x_offset, target - y_offset] = weight

        return weight_matrix

    def set_all_weights(self, weights):
        """Sets synaptic weights between input and motor neurons of the network.

        Args:
            weights (numpy.array): 2D array of shape (n_neurons, n_neurons).
            Input neurons are on the first axis, motor neurons on the second
            axis. See get_all_weights().
        """
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                connection = nest.GetConnections(self.input_neurons[i], self.motor_neurons[j])
                connection.set({"weight": weights[i, j]})

    def get_spike_counts(self):
        """Returns the spike counts of all motor neurons from the
        spike_recorders.

        Returns:
            numpy.array: Array of spike counts of all motor neurons.
        """
        events = self.spike_recorders.get("n_events")
        return np.array(events)

    def reset(self):
        """Resets the network for a new iteration by clearing all spike
        recorders.
        """
        self.spike_recorders.set({"n_events": 0})

    def set_input_spiketrain(self, input_cell, biological_time):
        """Sets a spike train to the input neuron specified by an index.

        Args:
            input_cell (int): Index of the input neuron to be stimulated.
            biological_time (float): Current biological time within the NEST
            simulator (in ms).
        """
        self.target_index = input_cell
        self.input_train = [biological_time + self.input_t_offset + i * ISI for i in range(N_INPUT_SPIKES)]
        # Round spike timings to 0.1ms to avoid conflicts with simulation time
        self.input_train = [np.round(x, 1) for x in self.input_train]

        # clear all input generators
        for input_neuron in range(self.num_neurons):
            nest.SetStatus(self.input_generators[input_neuron], {"spike_times": []})

        nest.SetStatus(self.input_generators[input_cell], {"spike_times": self.input_train})

    def get_max_activation(self):
        """Finds the motor neuron with the highest activation (number of spikes).

        Returns:
            int: Index of the motor neuron with the highest activation.
        """
        spikes = self.get_spike_counts()
        logging.debug(f"Got spike counts: {spikes}")

        # If multiple neurons have the same activation, one is chosen at random
        return int(np.random.choice(np.flatnonzero(spikes == spikes.max())))

    def calculate_reward(self):
        """Calculates the reward to be applied to the network based on
        performance in the previous simulation (distance between target and
        actual output). For R-STDP this reward informs the learning rule,
        for dopaminergic plasticity this is just a metric of fitness used for
        plotting the simulation.

        Returns:
            float: Reward between 0 and 1.
        """
        self.winning_neuron = self.get_max_activation()
        distance = np.abs(self.winning_neuron - self.target_index)

        if distance in REWARDS_DICT:
            bare_reward = REWARDS_DICT[distance]
        else:
            bare_reward = 0

        reward = bare_reward - self.mean_reward[self.target_index]

        self.mean_reward[self.target_index] = float(self.mean_reward[self.target_index] + reward / 2.0)

        logging.debug(f"Applying reward: {reward}")
        logging.debug(f"Average reward across all neurons: {np.mean(self.mean_reward)}")

        self.weight_history.append(self.get_all_weights())
        self.mean_reward_history.append(copy(self.mean_reward))

        return reward

    def get_performance_data(self):
        """Retrieves the performance data of the network across all simulations.

        Returns:
            tuple: A Tuple of 2 numpy.arrays containing reward history and
            weight history.
        """
        return self.mean_reward_history, self.weight_history

    @abstractmethod
    def apply_synaptic_plasticity(self, biological_time):
        """Applies weight changes to the synapses according to a given learning
        rule.

        Args:
            biological_time (float): Current NEST simulation time in ms.
        """
        pass


class PongNetDopa(PongNet):
    # Base reward current that is applied regardless of performance
    baseline_reward = 100.0
    # Maximum reward current to be applied to the dopaminergic neurons
    max_reward = 1000
    # Constant scaling factor for determining the current to be applied to the
    # dopaminergic neurons
    dopa_signal_factor = 4800
    # Offset for input spikes at every iteration in milliseconds. This offset
    # reserves the first part of every simulation step for the application of
    # the dopaminergic reward signal, avoiding interference between it and the
    # spikes caused by the input of the following iteration
    input_t_offset = 32

    # Neuron and synapse parameters:
    # Initial mean weight for synapses between input- and motor neurons
    mean_weight = 1275.0
    # Standard deviation for starting weights
    weight_std = 8
    # Number of neurons per population in the critic-network
    n_critic = 8
    # Synaptic weights from striatum and VP to the dopaminergic neurons
    w_da = -1150
    # Synaptic weight between striatum and VP
    w_str_vp = -250
    # Synaptic delay for the direct connection between striatum and
    # dopaminergic neurons
    d_dir = 200
    # Rate (Hz) for the background poisson generators
    poisson_rate = 15

    def __init__(self, apply_noise=True, num_neurons=20):
        super().__init__(apply_noise, num_neurons)

        self.vt = nest.Create("volume_transmitter")
        nest.SetDefaults(
            "stdp_dopamine_synapse",
            {
                "vt": self.vt,
                "tau_c": 70,
                "tau_n": 30,
                "tau_plus": 45,
                "Wmin": 1220,
                "Wmax": 1550,
                "b": 0.028,
                "A_plus": 0.85,
            },
        )

        if apply_noise:
            nest.Connect(
                self.input_neurons,
                self.motor_neurons,
                {"rule": "all_to_all"},
                {
                    "synapse_model": "stdp_dopamine_synapse",
                    "weight": nest.random.normal(self.mean_weight, self.weight_std),
                },
            )
            self.poisson_noise = nest.Create("poisson_generator", self.num_neurons, params={"rate": self.poisson_rate})
            nest.Connect(self.poisson_noise, self.motor_neurons, {"rule": "one_to_one"}, {"weight": self.mean_weight})
        else:
            # Because the poisson_generators cause additional spikes in the
            # motor neurons, it is necessary to compensate for their absence by
            # slightly increasing the mean of the weights between input and
            # motor neurons
            nest.SetDefaults("stdp_dopamine_synapse", {"Wmax": 1750})
            nest.Connect(
                self.input_neurons,
                self.motor_neurons,
                {"rule": "all_to_all"},
                {
                    "synapse_model": "stdp_dopamine_synapse",
                    "weight": nest.random.normal(self.mean_weight * 1.3, self.weight_std),
                },
            )

        # Setup the 'critic' as a network of three populations, consisting of
        # the striatum, ventral pallidum (vp) and dopaminergic neurons (dopa)
        self.striatum = nest.Create("iaf_psc_exp", self.n_critic)
        nest.Connect(
            self.input_neurons,
            self.striatum,
            {"rule": "all_to_all"},
            {"synapse_model": "stdp_dopamine_synapse", "weight": nest.random.normal(self.mean_weight, self.weight_std)},
        )
        self.vp = nest.Create("iaf_psc_exp", self.n_critic)
        nest.Connect(self.striatum, self.vp, syn_spec={"weight": self.w_str_vp})
        self.dopa = nest.Create("iaf_psc_exp", self.n_critic)
        nest.Connect(self.vp, self.dopa, syn_spec={"weight": self.w_da})
        nest.Connect(self.striatum, self.dopa, syn_spec={"weight": self.w_da, "delay": self.d_dir})
        nest.Connect(self.dopa, self.vt)

        # Current generator to stimulate dopaminergic neurons based on
        # network performance
        self.dopa_current = nest.Create("dc_generator")
        nest.Connect(self.dopa_current, self.dopa)

    def apply_synaptic_plasticity(self, biological_time):
        """Injects a current into the dopaminergic neurons based on how much of
        the motor neurons' activity stems from the target output neuron.
        """
        spike_counts = self.get_spike_counts()
        target_n_spikes = spike_counts[self.target_index]
        # avoid zero division if none of the neurons fired.
        total_n_spikes = max(sum(spike_counts), 1)

        reward_current = self.dopa_signal_factor * target_n_spikes / total_n_spikes + self.baseline_reward

        # Clip the dopaminergic signal to avoid runaway synaptic weights
        reward_current = min(reward_current, self.max_reward)

        self.dopa_current.stop = biological_time + self.input_t_offset
        self.dopa_current.start = biological_time
        self.dopa_current.amplitude = reward_current

        self.calculate_reward()

    def __repr__(self) -> str:
        return ("noisy " if self.apply_noise else "clean ") + "TD"
