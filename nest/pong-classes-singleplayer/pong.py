import numpy as np

LEFT_SCORE = -1
RIGHT_SCORE = +1
GAME_CONTINUES = 0

MOVE_DOWN = -1
MOVE_UP = +1
DONT_MOVE = 0


class GameObject:
    def __init__(self, game, x_pos=0.5, y_pos=0.5, velocity=0.2, direction=(0, 0)):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.velocity = velocity
        self.direction = direction
        self.game = game
        self.update_cell()

    def get_cell(self):
        return self.cell

    def get_pos(self):
        return self.x_pos, self.y_pos

    def update_cell(self):
        x_cell = int(np.floor((self.x_pos / self.game.x_length) * self.game.x_grid))
        y_cell = int(np.floor((self.y_pos / self.game.y_length) * self.game.y_grid))
        self.cell = [x_cell, y_cell]


class Ball(GameObject):
    def __init__(self, game, x_pos=0.8, y_pos=0.5,
                 velocity=0.025, direction=(-1 / 2.0, 1 / 2.0), radius=0.025):
        super().__init__(game, x_pos, y_pos, velocity, direction)
        self.ball_radius = radius
        self.update_cell()


class Paddle(GameObject):
    length = 0.2

    def __init__(self, game, left, y_pos=0.5, velocity=0.05, direction=0):
        # Only left paddle exists.
        x_pos = 0.0 if left else game.x_length
        super().__init__(game, x_pos, y_pos, velocity, direction)
        self.update_cell()

    def move_up(self):
        self.direction = MOVE_UP

    def move_down(self):
        self.direction = MOVE_DOWN

    def dont_move(self):
        self.direction = DONT_MOVE


class GameOfPong(object):
    x_grid = 32
    y_grid = 20
    x_length = 1.6
    y_length = 1.0

    def __init__(self):
        # Only left paddle remains
        self.l_paddle = Paddle(self, True)

        # Removed r_paddle entirely
        self.r_paddle = None  # kept for interface compatibility

        self.reset_ball()
        self.result = 0

    def reset_ball(self, towards_left=False):
        initial_vx = 0.5 + 0.5 * np.random.random()
        initial_vy = 1.0 - initial_vx
        if towards_left:
            initial_vx *= -1
        initial_vy *= np.random.choice([-1.0, 1.0])

        self.ball = Ball(self, direction=[initial_vx, initial_vy])
        self.ball.y_pos = np.random.random() * self.y_length

    def update_ball_direction(self):
        # Bounce on top
        if self.ball.y_pos + self.ball.ball_radius >= self.y_length:
            self.ball.direction[1] = -abs(self.ball.direction[1])

        # Bounce on bottom
        elif self.ball.y_pos - self.ball.ball_radius <= 0:
            self.ball.direction[1] = abs(self.ball.direction[1])

        # LEFT SIDE → scoring or paddle bounce
        if self.ball.x_pos - self.ball.ball_radius <= 0:
            # Check if ball hits paddle
            if abs(self.l_paddle.y_pos - self.ball.y_pos) <= Paddle.length / 2:
                # Bounce right
                self.ball.direction[0] = abs(self.ball.direction[0])
            else:
                return RIGHT_SCORE  # player missed

        # RIGHT SIDE → now a solid wall (always bounce)
        elif self.ball.x_pos + self.ball.ball_radius >= self.x_length:
            self.ball.direction[0] = -abs(self.ball.direction[0])

        return GAME_CONTINUES

    def propagate_ball_and_paddles(self):
        # Only one paddle
        paddle = self.l_paddle
        paddle.y_pos += paddle.direction * paddle.velocity
        paddle.y_pos = min(max(0, paddle.y_pos), self.y_length)
        paddle.update_cell()

        # Move ball
        self.ball.y_pos += self.ball.velocity * self.ball.direction[1]
        self.ball.x_pos += self.ball.velocity * self.ball.direction[0]
        self.ball.update_cell()

    def get_ball_cell(self):
        return self.ball.get_cell()

    def step(self):
        ball_status = self.update_ball_direction()
        self.propagate_ball_and_paddles()
        self.result = ball_status
        return ball_status

