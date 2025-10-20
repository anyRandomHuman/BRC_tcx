# # import wandb
# import matplotlib.pyplot as plt
# import pandas
# run_name= "/crusaderx/BRC/runs/wxmvzj11"
# data_path = rf'./wandb_results/h1-crawl-v0_20251013-203856_0.csv'
# # api = wandb.Api()
# # run = api.run(run_name)

# # run.history().to_csv(rf'./wandb_results/{run.name}.csv')
# row = 4
# col = 5
# fig, ax = plt.subplots(row,col, sharex=True, sharey=False)
# data = pandas.read_csv(data_path)
# for i in range(len(data.columns)):
#     ax[i/col][i%col].plot()
    
# plt.plot(data)
# plt.show()



import wandb
import pandas as pd

path  = 'wandb_results/h1-crawl-v0_20251013-203856_0.csv'
# Read our CSV into a new DataFrame
pandas_dataframe = pd.read_csv(path)


# Convert the DataFrame into a W&B Table
wandb_table = wandb.Table(dataframe=pandas_dataframe)


# Add the table to an Artifact to increase the row 
wandb_table_artifact = wandb.Artifact(
    "wandb_artifact", 
    type="dataset")        
wandb_table_artifact.add(wandb_table, "table")


# Log the raw csv file within an artifact to preserve our data
wandb_table_artifact.add_file(path)


# Start a W&B run to log data
run = wandb.init(project="...")


# Log the table to visualize with a run...
run.log({"data": wandb_table})


# and Log as an Artifact
run.log_artifact(wandb_table_artifact)



