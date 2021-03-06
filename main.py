import pandas as pd

# Press the green button in the gutter to run the script.
from reinforcement_learning import CustomEnv

if __name__ == '__main__':
    df = pd.read_csv('./pricedata.csv')
    df = df.sort_values('Date')

    lookback_window_size = 50
    train_df = df[:-720 - lookback_window_size]
    test_df = df[-720 - lookback_window_size:]  # 30days

    train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
    test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size)

    CustomEnv.Random_games(train_env, train_episodes=10, train_batch_size=500)
