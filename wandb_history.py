from pandas.core.interchange.dataframe_protocol import DataFrame

import wandb
import matplotlib.pyplot as plt
import pandas
id = 'snvev002'
run_name= f"/crusaderx/BRC/runs/{id}"
# data_path = rf'./wandb_results/h1-crawl-v0_20251013-203856_0.csv'
api = wandb.Api()
run = api.run(run_name)
h: DataFrame = run.history()
to_mean_keys = ['dormant_ratio', 'dead_percentage', 'feature_norm', 'dead_percentage', 'effective_rank',
                'parameter_norm', 'conflict_rate']

# --- Corrected logic to compute and log means ---

# 1. Prepare keys and column names
# Ensure keys are unique
to_mean_keys = sorted(list(set(to_mean_keys)))
all_cols = h.columns

# 2. Calculate row-wise means for matching columns
# Dictionary to hold the new time-series data (mean per key)
mean_data = {}

# Iterate through each key you want to average
for key in to_mean_keys:
    # Find all columns in the history DataFrame that contain this key
    matching_cols = [col for col in all_cols if key in col]

    if matching_cols:
        # Calculate the mean *across* these columns for each row (step)
        # axis=1 specifies row-wise mean
        row_wise_mean_series = h[matching_cols].mean(axis=1)

        # Store this new Series, naming it 'mean_KEY'
        mean_data[f"mean_{key}"] = row_wise_mean_series
    else:
        # Optional: Log a warning if no matching columns were found
        print(f"Warning: No history columns found containing the key: '{key}'")

# 3. Create a new DataFrame with the computed means
# The index (e.g., steps) is automatically aligned with 'h'
mean_frame = pandas.DataFrame(mean_data, index=h.index)

# 4. Log these new mean time-series back to wandb
# We need the original step column, usually '_step'
step_col = '_step'

with wandb.init(id=id, resume="allow") as run:

    if step_col in h.columns:
        # Get the actual step values from the original history
        steps = h[step_col].astype(int)

        # Iterate over each step and the corresponding row of mean data
        for step, (_, row) in zip(steps, mean_frame.iterrows()):
            # Create a dictionary of { 'mean_key': value } for this step
            # Drop any NaN values which can't be logged
            log_update = row.dropna().to_dict()

            # Log this data at the correct step
            if log_update:
                run.log(log_update, commit=False, step=step)
    else:
        print(f"Error: Cannot log means. Default step column '{step_col}' not found in run history.")

# 5. (Optional) Log the final average of these means to the run summary
# print("Logging final mean values to run.summary...")
# for key, final_mean_val in mean_frame.mean().items():  # .mean() computes mean over all steps
#     if pandas.notna(final_mean_val):
#         # This will add e.g., "mean_dormant_ratio" to the summary tab
#         run.summary[key] = final_mean_val




# import wandb
# import pandas as pd
#
# path  = 'wandb_results/h1-crawl-v0_20251013-203856_0.csv'
# # Read our CSV into a new DataFrame
# pandas_dataframe = pd.read_csv(path)
#
#
# # Convert the DataFrame into a W&B Table
# wandb_table = wandb.Table(dataframe=pandas_dataframe)
#
#
# # Add the table to an Artifact to increase the row
# wandb_table_artifact = wandb.Artifact(
#     "wandb_artifact",
#     type="dataset")
# wandb_table_artifact.add(wandb_table, "table")
#
#
# # Log the raw csv file within an artifact to preserve our data
# wandb_table_artifact.add_file(path)
#
#
# # Start a W&B run to log data
# run = wandb.init(project="...")
#
#
# # Log the table to visualize with a run...
# run.log({"data": wandb_table})
#
#
# # and Log as an Artifact
# run.log_artifact(wandb_table_artifact)



