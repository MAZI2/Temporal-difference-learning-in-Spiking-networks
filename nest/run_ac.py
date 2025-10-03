def run_games(self, max_runs=10000):
    """Runs a simulation of pong games and stores the results.

    Args:
        max_runs (int, optional): Number of iterations to simulate.
        Defaults to 10000.
    """
    start_time = time.time()
    run = 0
    biological_time = 0

    while run < max_runs:
        # input neuron in run (one iteration etc. ball position)
        self.input_index = self.game.ball.get_cell()[1]

        # set inputs for both players
        self.player1.set_input_spiketrain(self.input_index, biological_time)
        self.player2.set_input_spiketrain(self.input_index, biological_time)

        logging.debug("Running simulation...")
        # simulate for 200ms (1 iteration)
        nest.Simulate(POLL_TIME)
        # current time in iteration (ms)
        biological_time = nest.GetKernelStatus("biological_time")
        print(biological_time)
        

        # after each iteration ... apply dopamine
        # network.winning_neuron ?= O1 ? ... apply reward
        """
        for network, paddle in zip([self.player1, self.player2], [self.game.l_paddle, self.game.r_paddle]):
            network.apply_synaptic_plasticity(biological_time)
            network.reset()

            position_diff = network.winning_neuron - paddle.get_cell()[1]
            if position_diff > 0:
                paddle.move_up()
            elif position_diff == 0:
                paddle.dont_move()
            else:
                paddle.move_down()
        """

        self.run += 1

    end_time = time.time()
