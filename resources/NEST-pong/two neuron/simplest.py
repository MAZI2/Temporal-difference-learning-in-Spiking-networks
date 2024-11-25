import numpy as np
import nest

# Parameters
NUM_NEURONS = 20  # Number of input and output neurons
SIMULATION_TIME = 200  # Simulation time per iteration (ms)
NUM_ITERATIONS = 100  # Total number of iterations
LEARNING_RATE = 0.01  # R-STDP learning rate
REWARD_STRENGTH = 1.0  # Reward for correct output
PENALTY_STRENGTH = -0.5  # Penalty for incorrect output

# Create NEST environment
nest.ResetKernel()

nest.set_verbosity("M_WARNING")


# Create neurons
input_neurons = nest.Create("parrot_neuron", NUM_NEURONS)
output_neurons = nest.Create("iaf_psc_exp", NUM_NEURONS)
spike_generators = nest.Create("spike_generator", NUM_NEURONS)
spike_recorders = nest.Create("spike_recorder", NUM_NEURONS)
spike_recorders_in = nest.Create("spike_recorder", NUM_NEURONS)


# Connect input neurons to spike generators
nest.Connect(spike_generators, input_neurons, "one_to_one")

# Connect input neurons to output neurons
"""
nest.Connect(
    input_neurons, 
    output_neurons, 
    syn_spec={"synapse_model": "stdp_synapse", "weight": nest.random.uniform(0.1, 0.5)}
)
"""
mean_weight = 500
nest.Connect(
    input_neurons,
    output_neurons,
    {"rule": "all_to_all"},
    {"weight": nest.random.normal(mean_weight, 1)},
)


# Connect output neurons to spike recorders
nest.Connect(output_neurons, spike_recorders, "one_to_one")
nest.Connect(input_neurons, spike_recorders_in, "one_to_one")


# Target configuration
TARGET_OUTPUT = 10  # Only the first output neuron is correct

# Set up simulation
for iteration in range(NUM_ITERATIONS):
    # Generate spikes for all input neurons
    spike_times = np.arange(1.0, SIMULATION_TIME, step=10.0).tolist()
    for neuron_id in range(NUM_NEURONS):
        nest.SetStatus(
            spike_generators[neuron_id], {"spike_times": spike_times if neuron_id < 2 else []}
        )
    
    # Run simulation for 200 ms
    nest.Simulate(SIMULATION_TIME)

    # Get spike counts for output neurons
    spike_counts = np.array([recorder.get("n_events") for recorder in nest.GetStatus(spike_recorders)])
    spike_counts2 = np.array([recorder.get("n_events") for recorder in nest.GetStatus(spike_recorders_in)])

    print(spike_counts, spike_counts2) 

    # Identify the winning neuron (max spikes)
    winning_neuron = np.argmax(spike_counts)

    # Calculate reward
    if winning_neuron == TARGET_OUTPUT:
        reward = REWARD_STRENGTH
    else:
        reward = PENALTY_STRENGTH

    # Apply R-STDP reward rule
    connections = nest.GetConnections(input_neurons, output_neurons)
    if iteration == 0:
        print(connections)
    for conn in connections:
        pre = conn.source
        post = conn.target
        weight = conn.weight

        if post == TARGET_OUTPUT:
            weight += LEARNING_RATE * reward
        else:  # Incorrect output
            weight += LEARNING_RATE * reward * 0.5  # Smaller penalty for incorrect

        # Update weight (ensure it stays within limits)
        nest.SetStatus(conn, {"weight": max(0.1, min(weight, 1.0))})

    # Log results for debugging
    print(f"Iteration {iteration+1}/{NUM_ITERATIONS}: Winning neuron {winning_neuron}, Reward: {reward}")


