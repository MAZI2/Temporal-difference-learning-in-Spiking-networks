import logging
from abc import ABC, abstractmethod
from copy import copy

import nest
import numpy as np

# Simulation time per iteration in milliseconds.
POLL_TIME = 400
# Number of spikes in an input spiketrain per iteration.
N_INPUT_SPIKES = 200
# Inter-spike interval of the input spiketrain.
ISI = 1.0
# Standard deviation of Gaussian current noise in picoampere.
BG_STD = 220.0
# Reward to be applied depending on distance to target neuron.
#REWARDS_DICT = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}

neuron_params = {
    "C_m": 250.0,      # membrane capacitance in pF
    "tau_m": 10.0,     # membrane time constant in ms
    "V_reset": 0.0,  # reset potential mV
    "V_th": 20.0,     # spike threshold mV
    "t_ref": 0.5,      # absolute refractory period ms
    "V_m": 0.0,      # initial membrane potential mV
    "E_L": 0.0,      # resting potential mV
}

class PongNet(ABC):
    def __init__(self, apply_noise=True, num_neurons=25):
        self.apply_noise = apply_noise
        self.num_input_neurons = 25
        self.num_output_neurons = 4

        self.weight_history = []
        """
        self.mean_reward = np.array([0.0 for _ in range(self.num_neurons)])
        self.mean_reward_history = []
        """
        self.winning_neuron = 0

        # goal
        #self.target_index = (4, 4)

        self.input_generators = nest.Create("spike_generator", self.num_input_neurons)
        # N_s = 1
        self.input_neurons = nest.Create("parrot_neuron", self.num_input_neurons)
        nest.Connect(self.input_generators, self.input_neurons, {"rule": "one_to_one"})

        # Actor
        self.motor_neurons = nest.Create("iaf_psc_alpha", self.num_output_neurons, params=neuron_params)
        self.spike_recorders = nest.Create("spike_recorder", self.num_output_neurons)
        nest.Connect(self.motor_neurons, self.spike_recorders, {"rule": "one_to_one"})

    def get_all_weights(self):
        """Returns all synaptic weights between input and motor neurons.

        Returns:
            numpy.array: 2D array of shape (n_neurons, n_neurons). Input
            neurons are on the first axis, motor neurons on the second axis.
        """
        x_offset = self.input_neurons[0].get("global_id")
        y_offset = self.motor_neurons[0].get("global_id")
        weight_matrix = np.zeros((self.num_input_neurons, self.num_output_neurons))
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
        for i in range(self.num_input_neurons):
            for j in range(self.num_outupt_neurons):
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
        self.input_train = [biological_time + self.input_t_offset + i * ISI for i in range(N_INPUT_SPIKES)]
        # Round spike timings to 0.1ms to avoid conflicts with simulation time
        self.input_train = [np.round(x, 1) for x in self.input_train]

        # clear all input generators
        for input_neuron in range(self.num_input_neurons):
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


class GridWorldAC(PongNet):
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
    # TODO: might be wrong
    input_t_offset = 200

    # Neuron and synapse parameters:
    # Initial mean weight for synapses between input- and motor neurons
    weight_std = 8
    n_critic = 8

    w_c_a = 30
    w_c_a_max = 90

    w_c_str = 150
    w_c_str_max = 200

    w_str_vp = -50
    w_str_da = -100
    w_vp_da = -100

    d_dir = 200

    # External
    w_ex_vp = 45.61
    rate_ex_vp = 4000

    w_ex_da = 45.61
    rate_ex_da = 10000

    w_ex_all = 100
    w_in_all = -100
    rate_ex_all = 10
    rate_in_all = 5

    def __init__(self, apply_noise=True, num_neurons=20):
        super().__init__(apply_noise, num_neurons)



        self.vt = nest.Create("volume_transmitter")
        nest.SetDefaults(
            "delayed_synapse",
            {
                "volume_transmitter": self.vt,
                "tau_c": 50,
                "tau_c_delay": 200,
                "tau_n": 50,
                "tau_plus": 45,
                "Wmin": 1220,
                "Wmax": 1550,
                "b": 0.028,
                "A_plus": 0.81,
            },
        )

        # Because the poisson_generators cause additional spikes in the
        # motor neurons, it is necessary to compensate for their absence by
        # slightly increasing the mean of the weights between input and
        # motor neurons
        nest.SetDefaults("delayed_synapse", {"Wmax": 1750})
        # Input → motor
        nest.CopyModel(
            "delayed_synapse",
            "stdp_motor_synapse",
            {
                "volume_transmitter": self.vt,
                "Wmin": self.w_c_a,
                "Wmax": self.w_c_a_max,
            }
        )

        # Input → striatum
        nest.CopyModel(
            "delayed_synapse",
            "stdp_striatum_synapse",
            {
                "volume_transmitter": self.vt,
                "Wmin": self.w_c_str,
                "Wmax": self.w_c_str_max,
            }
        )

        nest.Connect(
            self.input_neurons,
            self.motor_neurons,
            {"rule": "all_to_all"},
            {
                "synapse_model": "stdp_motor_synapse",
                "weight": nest.random.normal(self.w_c_a, self.weight_std),
            },
        )

        self.striatum = nest.Create("iaf_psc_alpha", self.n_critic, params=neuron_params)
        nest.Connect(
            self.input_neurons,
            self.striatum,
            {"rule": "all_to_all"},
            {
                "synapse_model": "stdp_striatum_synapse", 
                "weight": nest.random.normal(self.w_c_str, self.weight_std),
            },
        )

        
        self.vp = nest.Create("iaf_psc_alpha", self.n_critic, params=neuron_params)
        nest.Connect(self.striatum, self.vp, syn_spec={"weight": self.w_str_vp})

        self.dopa = nest.Create("iaf_psc_alpha", self.n_critic, params=neuron_params)
        nest.Connect(self.vp, self.dopa, syn_spec={"weight": self.w_vp_da})
        nest.Connect(self.striatum, self.dopa, syn_spec={"weight": self.w_str_da, "delay": self.d_dir})
        nest.Connect(self.dopa, self.vt)

        # Current generator to stimulate dopaminergic neurons based on
        # network performance
        self.dopa_current = nest.Create("dc_generator")
        nest.Connect(self.dopa_current, self.dopa)

        self.state = (0, 0)
        self.reward = False

        # dopamine recorder
        self.dopa_recorder = nest.Create("spike_recorder")
        nest.Connect(self.dopa, self.dopa_recorder)

        # striatum recorder,multimeter
        self.str_recorder = nest.Create("spike_recorder")
        nest.Connect(self.striatum, self.str_recorder)

        self.str_multimeter = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.1})
        nest.Connect(self.str_multimeter, self.striatum) 

        # VP recorder
        self.vp_recorder = nest.Create("spike_recorder")
        nest.Connect(self.vp, self.vp_recorder)

        self.vp_multimeter = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.1})
        nest.Connect(self.vp_multimeter, self.vp) 


        # Cortex recorder
        self.cortex_recorder = nest.Create("spike_recorder")
        nest.Connect(self.input_neurons, self.cortex_recorder)

        # Motor recorder
        self.motor_recorder = nest.Create("spike_recorder")
        nest.Connect(self.motor_neurons, self.motor_recorder)

        # Poisson input to vp
        self.poisson_vp = nest.Create("poisson_generator", params={"rate": self.rate_ex_vp}) 
        nest.Connect(
            self.poisson_vp,
            self.vp,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": self.w_ex_vp}
        )
        self.poisson_da = nest.Create("poisson_generator", params={"rate": self.rate_ex_da}) 
        nest.Connect(
            self.poisson_da,
            self.dopa,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": self.w_ex_da}
        )

        # Poisson input to all neurons (excitatory and inhibitory)
        self.poisson_all_ex = nest.Create("poisson_generator", params={"rate": self.rate_ex_all})
        self.poisson_all_inh = nest.Create("poisson_generator", params={"rate": self.rate_in_all})
        nest.Connect(
            self.poisson_all_ex,
            self.vp,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": self.w_ex_all}
        )
        nest.Connect(
            self.poisson_all_inh,
            self.vp,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": self.w_in_all}
        )
        """
        nest.Connect(
            self.poisson_all_ex,
            self.striatum,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": self.w_ex_all}
        )
        nest.Connect(
            self.poisson_all_inh,
            self.striatum,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": self.w_in_all}
        )

        nest.Connect(
            self.poisson_all_ex,
            self.input_neurons,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": self.w_ex_all}
        )
        nest.Connect(
            self.poisson_all_inh,
            self.input_neurons,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": self.w_in_all}
        )
        """


        """
        nest.Connect(
            self.poisson_all_ex,
            self.motor_neurons,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": self.w_ex_all}
        )
        
        nest.Connect(
            self.poisson_all_ex,
            self.dopa,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": self.w_ex_all}
        )

        # inhibitory
        nest.Connect(
            self.poisson_all_inh,
            self.motor_neurons,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": self.w_in_all}
        )
        
        nest.Connect(
            self.poisson_all_inh,
            self.dopa,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": self.w_in_all}
        )
        """



    def set_state(self, state):
        """
        state from 0 to num_input_neurons
        """
        self.state = state

    def apply_synaptic_plasticity(self, biological_time):
        """
        injects external reward
        Injects a current into the dopaminergic neurons based on how much of
        the motor neurons' activity stems from the target output neuron.
        """
        
        """
        spike_counts = self.get_spike_counts()
        target_n_spikes = spike_counts[self.target_index]
        # avoid zero division if none of the neurons fired.
        total_n_spikes = max(sum(spike_counts), 1)

        reward_current = self.dopa_signal_factor * target_n_spikes / total_n_spikes + self.baseline_reward

        # Clip the dopaminergic signal to avoid runaway synaptic weights
        reward_current = min(reward_current, self.max_reward)
        """

        self.winning_neuron = self.get_max_activation()
        action = self.winning_neuron

        temp_state = [self.state[0], self.state[1]]

        # [up, down, left, right]
        """
        if action == 0:
            temp_state[0] -= 1
        elif action == 1:
            temp_state[0] += 1
        elif action == 2:
            temp_state[1] -= 1
        elif action == 3:
            temp_state[1] += 1
        
        if temp_state == [4, 4]:
            reward_current = 600
            print("REWARDED")
        else:
            reward_current = self.baseline_reward
        """
        if self.reward:
            print("Rewarded")
            reward_current = 2600
            self.reward = False
        else:
            reward_current = self.baseline_reward

        self.dopa_current.stop = biological_time + self.d_dir
        self.dopa_current.start = biological_time
        self.dopa_current.amplitude = reward_current

        print("dopa current", self.dopa_current.amplitude)

    def __repr__(self) -> str:
        return ("noisy " if self.apply_noise else "clean ") + "TD"
