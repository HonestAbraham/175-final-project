import pandas as pd
import matplotlib.pyplot as plt

# Read the loss data
df = pd.read_csv('loss_data.csv')

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df['Episode'], df['Loss'], 'b-', label='Training Loss')
plt.title('DQN Training Loss Over Time')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Add a moving average line to show the trend
window_size = 5
df['Moving Average'] = df['Loss'].rolling(window=window_size).mean()
plt.plot(df['Episode'], df['Moving Average'], 'r--', label=f'{window_size}-Episode Moving Average')

plt.show()