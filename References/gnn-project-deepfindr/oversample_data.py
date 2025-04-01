# %% Load raw dataset
import pandas as pd

data = pd.read_csv("data/raw/HIV_train.csv")
data.index = data["index"]
data["HIV_active"].value_counts()
start_index = data.iloc[0]["index"]

# %% Apply oversampling

# Check how many additional samples we need
neg_class = data["HIV_active"].value_counts()[0]
pos_class = data["HIV_active"].value_counts()[1]
multiplier = int(neg_class/pos_class) - 1

# Replicate the dataset for the positive class
replicated_pos = [data[data["HIV_active"] == 1]]*multiplier

# Append replicated data
data = data.append(replicated_pos,
                    ignore_index=True)
print(data.shape)

# Shuffle dataset
data = data.sample(frac=1).reset_index(drop=True)

# Re-assign index (This is our ID later)
index = range(start_index, start_index + data.shape[0])
data.index = index
data["index"] = data.index
data.head()

# %% Save
data.to_csv("data/raw/HIV_train_oversampled.csv")

# %%
