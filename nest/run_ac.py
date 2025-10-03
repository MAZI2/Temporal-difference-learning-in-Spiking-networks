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
import gridworld 
from gridworld_ac import POLL_TIME, GridWorldAC

class AIGridworld:
    def __init__(self):
        """A class to run and store pong simulations of two competing spiking
        neural networks.

        Args:
            p1 (PongNet): Network to play on the left side.
            p2 (PongNet): Network to play on the right side.
            out_folder (str, optional): Name of the output folder. Defaults to
            current time stamp (YYYY-mm-dd-HH-MM-SS).
        """
        self.grid_size = (5, 5)
        self.start = (0, 0)
        self.goal = (4, 4)

        self.done = False

        self.game = gridworld.GridWorld(size=self.grid_size, start=self.start, goal=self.goal)
        self.state = self.game.reset()
        self.player = GridWorldAC(False)

        """
        if out_dir == "":
            out_dir = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now())
        if os.path.exists(out_dir):
            print(f"output folder {out_dir} already exists!")
            sys.exit()
        os.mkdir(out_dir)
        self.out_dir = out_dir
        """

        logging.info(f"setup complete for gridworld")

    def run_games(self, max_runs=10000):
        """Runs a simulation of pong games and stores the results.

        Args:
            max_runs (int, optional): Number of iterations to simulate.
            Defaults to 10000.
        """
        self.game_data = []
        l_score, r_score = 0, 0

        start_time = time.time()
        self.run = 0
        biological_time = 0

        logging.info(f"Starting simulation of {max_runs} iterations of " f"{POLL_TIME}ms each.")

        # 1 state transition
        while self.run < max_runs:
            logging.debug("")
            logging.debug(f"Iteration {self.run}:")
            
            self.input_index = self.state[0] * self.grid_size[1] + self.state[1]

            self.player.set_input_spiketrain(self.input_index, biological_time)

            
            logging.debug("Running simulation...")
            nest.Simulate(POLL_TIME)
            biological_time = nest.GetKernelStatus("biological_time")

            self.player.apply_synaptic_plasticity(biological_time)
            self.player.reset()

            action = self.player.winning_neuron

            #print()
            #print("state", self.state)
            #print("action", action)

            self.state, self.reward, self.done = self.game.step(action)
            self.player.set_state(self.state)

            self.run += 1
            self.game_data.append(
                [
                    self.state,
                ]
            )
            if self.done:
                print("Reached goal")
                self.done = False
                self.state = self.game.reset()
                self.player.set_state(self.state)

        end_time = time.time()
        logging.info(
            f"Simulation of {max_runs} runs complete after: " f"{datetime.timedelta(seconds=end_time - start_time)}"
        )

        self.game_data = np.array(self.game_data)
        """

        out_data = dict()
        out_data["ball_pos"] = self.game_data[:, 0]
        out_data["left_paddle"] = self.game_data[:, 1]
        out_data["right_paddle"] = self.game_data[:, 2]
        out_data["score"] = self.game_data[:, 3]

        logging.info("saving game data...")
        with open(os.path.join(self.out_dir, "gamestate.pkl"), "wb") as file:
            pickle.dump(out_data, file)

        logging.info("saving network data...")

        for net, filename in zip([self.player1, self.player2], ["data_left.pkl.gz", "data_right.pkl.gz"]):
            with gzip.open(os.path.join(self.out_dir, filename), "w") as file:
                output = {"network_type": repr(net), "with_noise": net.apply_noise}
                performance_data = net.get_performance_data()
                output["rewards"] = performance_data[0]
                output["weights"] = performance_data[1]
                pickle.dump(output, file)

        """
        logging.info("Done.")

        weights = self.player.get_all_weights()
        policy = np.zeros((5, 5, 4))

        k = 0
        for i in range(5):
            for j in range(5):
                policy[i][j] = weights[k]/np.sum(weights[k])
                k += 1

        self.game.plot_policy((5, 5), policy, start=(0,0), goal=self.goal)
                

if __name__ == "__main__":
    nest.set_verbosity("M_WARNING")

    level = logging.INFO
    format = "%(asctime)s - %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=format, datefmt=datefmt)

    runs=2000

    AIGridworld().run_games(max_runs=runs)
