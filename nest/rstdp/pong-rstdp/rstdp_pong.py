# -*- coding: utf-8 -*-
#
# networks.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

r"""Classes to encapsulate the neuronal networks.
----------------------------------------------------------------
Two types of network capable of playing pong are implemented. PongNetRSTDP
can solve the problem by updating the weights of static synapses after
every simulation step according to the R-STDP rules defined in [1]_.

PongNetDopa uses the actor-critic model described in [2]_ to determine the
amount of reward to send to the dopaminergic synapses between input and motor
neurons. In this framework, the motor neurons represent the actor, while a
secondary network of three populations (termed striatum, VP, and dopaminergic
neurons) form the critic which modulates dopamine concentration based on
temporal difference error.

Both of them inherit some functionality from the abstract base class PongNet.

See Also
---------
`Original implementation <https://github.com/electronicvisions/model-sw-pong>`_

References
----------
.. [1] Wunderlich T., et al (2019). Demonstrating advantages of
       neuromorphic computation: a pilot study. Frontiers in neuroscience, 13,
       260. https://doi.org/10.3389/fnins.2019.00260

.. [2] Potjans W., Diesmann M.  and Morrison A. (2011). An imperfect
       dopaminergic error signal can drive temporal-difference learning. PLoS
       Computational Biology, 7(5), e1001133.
       https://doi.org/10.1371/journal.pcbi.1001133

:Authors: J Gille, T Wunderlich, Electronic Vision(s)
"""

import logging
from abc import ABC, abstractmethod
from copy import copy

import nest
import numpy as np

# Simulation time per iteration in milliseconds.
POLL_TIME = 200
# Number of spikes in an input spiketrain per iteration.
N_INPUT_SPIKES = 200
# Inter-spike interval of the input spiketrain.
ISI = 1.0
# Standard deviation of Gaussian current noise in picoampere.
BG_STD = 220.0
# Reward to be applied depending on distance to target neuron.
REWARDS_DICT = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}

neuron_params = {
    "C_m": 250.0,      # membrane capacitance in pF
    "tau_m": 10.0,     # membrane time constant in ms
    "V_reset": 0.0,  # reset potential mV
    "V_th": 20.0,     # spike threshold mV
    "t_ref": 0.5,      # absolute refractory period ms
    "V_m": 0.0,      # initial membrane potential mV
    "E_L": 0.0,      # resting potential mV
}
neuron_params_motor = {
        "C_m": 20.0,
        "tau_m": 10.0,
        "V_reset": 0.0,
        "V_th": 20.0,
        "t_ref": 0.1,
        "tau_syn_ex": 0.1,
        "tau_minus": 1.0,
        "V_m": 0.0,
        "E_L": 0.0,
    }

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
        self.state = 0

        self.weight_history = []
        self.mean_reward = np.array([0.0 for _ in range(self.num_neurons)])
        self.mean_reward_history = []
        self.winning_neuron = 0


        self.dopa = nest.Create("iaf_psc_alpha", 8, params=neuron_params)
        self.dopa_current = nest.Create("dc_generator")
        nest.Connect(self.dopa_current, self.dopa)
        self.dopa_recorder = nest.Create("spike_recorder")
        nest.Connect(self.dopa, self.dopa_recorder)

        self.vt = nest.Create("volume_transmitter")
        nest.Connect(self.dopa, self.vt)


        self.input_generators = nest.Create("spike_generator", self.num_neurons)
        self.input_pre = nest.Create("parrot_neuron", self.num_neurons)
        nest.Connect(self.input_generators, self.input_pre, {"rule": "one_to_one"})

        self.input_neurons = nest.Create("iaf_psc_alpha", self.num_neurons)
        nest.Connect(self.input_pre, self.input_neurons, {"rule": "one_to_one"}, {"weight": 120})
        self.input_recorder = nest.Create("spike_recorder", self.num_neurons)
        self.global_input_recorder = nest.Create("spike_recorder", self.num_neurons)
        nest.Connect(self.input_neurons, self.input_recorder, {"rule": "one_to_one"})

        self.motor_neurons = nest.Create("iaf_psc_exp", self.num_neurons)
        self.spike_recorders = nest.Create("spike_recorder", self.num_neurons)
        nest.Connect(self.motor_neurons, self.spike_recorders, {"rule": "one_to_one"})

        self.motor_recorder = nest.Create("spike_recorder")
        nest.Connect(self.motor_neurons, self.motor_recorder)


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

    def set_reward_current(self, biological_time, reward):
        reward_current = 600.0 * reward

        self.dopa_current.stop = biological_time + 200
        self.dopa_current.start = biological_time
        self.dopa_current.amplitude = reward_current 

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

    def set_state(self, state):
        self.state = state

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
            print("REWARDED: ", bare_reward)
        else:
            bare_reward = 0

        """
        if REWARDS_DICT_TEST[self.state] == self.winning_neuron:
            bare_reward = 1.0
            print("REWARDED!")
        else:
            bare_reward = 0
        """
        

        reward = bare_reward - self.mean_reward[self.target_index]
        if reward<0:
            reward = 0

        self.mean_reward[self.target_index] = float(self.mean_reward[self.target_index] + reward / 2.0)

        logging.debug(f"Applying reward: {reward}")
        logging.debug(f"Average reward across all neurons: {np.mean(self.mean_reward)}")
        print(f"Average reward across all neurons: {np.mean(self.mean_reward)}")


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

class PongNetRSTDP(PongNet):
    # Offset for input spikes in every iteration in milliseconds
    input_t_offset = 1
    # Learning rate to use in weight updates
    learning_rate = 0.7
    # Amplitude of STDP curve in arbitrary units
    stdp_amplitude = 36.0
    # Time constant of STDP curve in milliseconds
    stdp_tau = 64.0
    # Satuation value for accumulated STDP
    stdp_saturation = 128
    # Initial mean weight for synapses between input- and motor neurons
    mean_weight = 1300.0

    

    def __init__(self, apply_noise=True, num_neurons=20):
        super().__init__(apply_noise, num_neurons)
        nest.SetDefaults(
            "delayed_synapse",
            {
                "volume_transmitter": self.vt,
                "Wmax": 2000,
                "Wmin": 500,
                "tau_c": 5,
                "tau_c_delay": 200,
                "tau_n": 10,
                "tau_plus": 20,
                "b": 0.0,
                "A_plus": 0.7,
                "A_minus": 0.3
            },
        )

        if apply_noise:
            self.background_generator = nest.Create("noise_generator", self.num_neurons, params={"std": BG_STD})
            nest.Connect(self.background_generator, self.motor_neurons, {"rule": "one_to_one"})
            nest.Connect(
                self.input_neurons,
                self.motor_neurons,

                {"rule": "all_to_all"},
                {
                    "synapse_model": "delayed_synapse",
                    "weight": nest.random.normal(self.mean_weight, 1)},
            )

            self.poisson_da = nest.Create("poisson_generator", params={"rate": 1000}) 
            nest.Connect(
                self.poisson_da,
                self.dopa,
                conn_spec={"rule": "all_to_all"},
                syn_spec={"weight": 46}
            )

            
        else:
            print("EEEEE")
            # Because the noise_generators cause additional spikes in the motor
            # neurons, it is necessary to compensate for their absence by
            # slightly increasing the mean of the weights between input and
            # motor neurons
            nest.Connect(
                self.input_neurons,
                self.motor_neurons,
                {"rule": "all_to_all"},
                {"weight": nest.random.normal(self.mean_weight * 1.22, 5)},
            )

    def apply_synaptic_plasticity(self, biological_time):
        """Rewards network based on how close target and winning neuron are."""
        reward = self.calculate_reward()
        self.set_reward_current(biological_time, reward)

        #self.apply_rstdp(reward)

    def __repr__(self) -> str:
        return ("noisy " if self.apply_noise else "clean ") + "R-STDP"
