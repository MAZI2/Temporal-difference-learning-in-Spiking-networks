import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size=(5, 5), start=(0, 0), goal=(4, 4)):
        self.size = size
        self.start = start
        self.goal = goal
        self.state = start
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        dx, dy = self.actions[action]
        x, y = self.state
        nx, ny = x + dx, y + dy

        # keep inside grid
        nx = np.clip(nx, 0, self.size[0] - 1)
        ny = np.clip(ny, 0, self.size[1] - 1)

        self.state = (nx, ny)

        # reward
        if self.state == self.goal:
            return self.state, 1.0, True  # state, reward, done
        else:
            return self.state, 0.0, False  # small penalty for step

    def render(self):
        grid = np.full(self.size, " . ")
        x, y = self.state
        gx, gy = self.goal
        grid[gx, gy] = " G "
        grid[x, y] = " A "
        print("\n".join("".join(row) for row in grid))

    def plot_policy(self, grid_size, action_preferences, start=(0,0), goal=(2,2)):
        """
        Visualize action preferences in a GridWorld with correct arrow directions.

        Parameters
        ----------
        grid_size : tuple (rows, cols)
            Size of the grid world.
        action_preferences : np.ndarray, shape=(rows, cols, 4)
            Preference values for each state and action.
            Action order assumed: [up, down, left, right].
        start : tuple
            (row, col) coordinates of start cell.
        goal : tuple
            (row, col) coordinates of goal cell.
        """
        rows, cols = grid_size
        fig, ax = plt.subplots(figsize=(cols, rows))

        # Draw grid lines
        for x in range(cols+1):
            ax.axvline(x, color="k", lw=1)
        for y in range(rows+1):
            ax.axhline(y, color="k", lw=1)

        # Draw start and goal cells behind arrows
        ax.add_patch(plt.Rectangle((start[1], rows-start[0]-1), 1, 1, color='green', alpha=0.3))
        ax.add_patch(plt.Rectangle((goal[1], rows-goal[0]-1), 1, 1, color='red', alpha=0.3))

        # Draw arrows for each cell
        for i in range(rows):
            for j in range(cols):
                prefs = action_preferences[i, j]

                # Map actions to vectors
                # [up, down, left, right]
                dx = prefs[3] - prefs[2]   # right - left
                dy = prefs[0] - prefs[1]# up - down

                # No need to flip dy; matplotlib y-axis is inverted later
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx /= length
                    dy /= length

                scale = np.mean(np.abs(prefs))
                # Draw arrow
                ax.arrow(j + 0.5, rows - i - 0.5, dx*0.5*scale, dy*0.5*scale,
                         head_width=0.05, head_length=0.05, width=0.0005, fc="black", ec="black")

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.axis('off')
        plt.show()

