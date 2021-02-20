import pandas as pd
import numpy as np
import random
from collections import deque


class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50):
        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        # Action space from 0 to 3. 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # Order history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last
        # Lookback_window_size_sie steps
        self.orders_history = deque(maxlen=self.lookback_window_size)

        # Market history contains the OHCL values for the last lookback_window_size
        self.market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, 10)

        # Initial variables
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0

    # Reset the state of the environment to an initial state
    def reset(self, env_step_size=0):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0

        if env_step_size > 0:  # Used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_step_size)
            self.end_step = self.start_step + env_step_size
        else:  # Used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, 'High'],
                                        self.df.loc[current_step, 'Low'],
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume']
                                        ])
        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        current_price = random.uniform(
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'Close']
        )

        if action == 0:  # Hold
            pass
        elif action == 1 and self.balance > 0:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
        elif action == 2 and self.crypto_held > 0:
            # Sell 100% with current balance
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append(
            [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

        # Calculate reward
        reward = self.net_worth - self.prev_net_worth

        if self.net_worth <= self.initial_balance / 2:
            done = True
        else:
            done = False

        obs = self._next_observation()
        return obs, reward, done

    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, 'High'],
                                    self.df.loc[self.current_step, 'Low'],
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume']
                                    ])
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs

    # Render environment
    def render(self):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')

    # Random agent
    def Random_games(env, train_episodes = 50, train_batch_size = 500):
        average_net_worth = 0
        for episode in range(train_episodes):
            state = env.reset(env_step_size=train_batch_size)
            while True:
                env.render()
                action = np.random.randint(3, size=1)[0]
                state, reward, done = env.step(action)
                if env.current_step == env.end_step:
                    average_net_worth += env.net_worth
                    print("net_worth:", env.net_worth)
                    break
        print("average_net_worth: ", average_net_worth/train_episodes)