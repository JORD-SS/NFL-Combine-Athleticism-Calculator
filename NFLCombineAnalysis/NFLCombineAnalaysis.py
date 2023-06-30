import pandas as pd
import numpy as np
from scipy import stats

# User Input Function


def get_player(combine):

    while True:
        player = input("Enter a participant from the 2013-2022 NFL Combine: ")
        if player not in combine.index:
            print("Player is not in dataset, try someone else.")
        else:
            return player


# Reading CSV's
combine_2013 = pd.read_csv('Datasets/2013_combine.csv')
combine_2014 = pd.read_csv('Datasets/2014_combine.csv')
combine_2015 = pd.read_csv('Datasets/2015_combine.csv')
combine_2016 = pd.read_csv('Datasets/2016_combine.csv')
combine_2017 = pd.read_csv('Datasets/2017_combine.csv')
combine_2018 = pd.read_csv('Datasets/2018_combine.csv')
combine_2019 = pd.read_csv('Datasets/2019_combine.csv')
combine_2020 = pd.read_csv('Datasets/2020_combine.csv')
combine_2021 = pd.read_csv('Datasets/2021_combine.csv')
combine_2022 = pd.read_csv('Datasets/2022_combine.csv')

combine_datasets = [combine_2013, combine_2014, combine_2015, combine_2016,
                    combine_2017, combine_2018, combine_2019, combine_2020, combine_2021, combine_2022]
combine_datasets = [combine_2013, combine_2014, combine_2015, combine_2016,
                    combine_2017, combine_2018, combine_2019, combine_2020, combine_2021, combine_2022]

combine = pd.concat(
    combine_datasets, ignore_index=True).sort_values(["Pos", "Player"])
combine = combine.set_index("Player")

heights = combine["Ht"].str.split(pat="-", expand=True)
heights[[0, 1]] = heights[[0, 1]].apply(pd.to_numeric)
combine["Ht"] = (heights[0] * 12) + heights[1]

player_name = get_player(combine)
player_data = combine.loc[player_name]

combine["Pos"] = combine["Pos"].str.replace(pat="DB", repl="CB", regex=False)
means_by_position = combine.pivot_table(
    values=["Ht", "Wt", "40yd", "Vertical", "Bench", "Broad Jump", "3Cone", "Shuttle"], index="Pos", aggfunc=[np.mean])

pos_group_data = combine[combine["Pos"] == player_data["Pos"]]
pos_group_mean = pos_group_data.loc[:, "Ht":"Shuttle"].mean().round(2)

heights = pos_group_data["Ht"].dropna().to_numpy()
weights = pos_group_data["Wt"].dropna().to_numpy()
fourtys = pos_group_data["40yd"].dropna().to_numpy()
verticals = pos_group_data["Vertical"].dropna().to_numpy()
bench_reps = pos_group_data["Bench"].dropna().to_numpy()
broad_jumps = pos_group_data["Broad Jump"].dropna().to_numpy()
three_cones = pos_group_data["3Cone"].dropna().to_numpy()
shuttle_times = pos_group_data["Shuttle"].dropna().to_numpy()

pd.options.mode.chained_assignment = None

pos_group_data["Ht Perc"] = pos_group_data["Ht"].map(
    lambda x: stats.percentileofscore(heights, x))
pos_group_data["Wt Perc"] = pos_group_data["Wt"].map(
    lambda x: stats.percentileofscore(weights, x))
pos_group_data["Fourty Perc"] = pos_group_data["40yd"].map(
    lambda x: 100 - stats.percentileofscore(fourtys, x))
pos_group_data["Vertical Perc"] = pos_group_data["Vertical"].map(
    lambda x: stats.percentileofscore(verticals, x))
pos_group_data["Bench Perc"] = pos_group_data["Bench"].map(
    lambda x: stats.percentileofscore(bench_reps, x))
pos_group_data["Broad Jump Perc"] = pos_group_data["Broad Jump"].map(
    lambda x: stats.percentileofscore(broad_jumps, x))
pos_group_data["3Cone Perc"] = pos_group_data["3Cone"].map(
    lambda x: 100 - stats.percentileofscore(three_cones, x))
pos_group_data["Shuttle Perc"] = pos_group_data["Shuttle"].map(
    lambda x: 100 - stats.percentileofscore(shuttle_times, x))

pos_group_data["Physical Score"] = pos_group_data.loc[:,
                                                      ["Ht Perc", "Wt Perc", "Bench"]].apply(np.mean, axis=1)
pos_group_data["Speed Score"] = pos_group_data.loc[:,
                                                   ["Fourty Perc"]].apply(np.mean, axis=1)
pos_group_data["Explosive Score"] = pos_group_data.loc[:,
                                                       ["Broad Jump Perc", "Vertical Perc"]].apply(np.mean, axis=1)
pos_group_data["Agility Score"] = pos_group_data.loc[:,
                                                     ["3Cone Perc", "Shuttle Perc"]].apply(np.mean, axis=1)


physical_score = pos_group_data["Physical Score"].dropna().to_numpy()
speed_scores = pos_group_data["Speed Score"].dropna().to_numpy()
explosive_score = pos_group_data["Explosive Score"].dropna().to_numpy()
agility_score = pos_group_data["Agility Score"].dropna().to_numpy()

pos_group_data["Physical Score"] = pos_group_data["Physical Score"].map(
    lambda x: stats.percentileofscore(physical_score, x))
pos_group_data["Speed Score"] = pos_group_data["Speed Score"].map(
    lambda x: stats.percentileofscore(speed_scores, x))
pos_group_data["Explosive Score"] = pos_group_data["Explosive Score"].map(
    lambda x: stats.percentileofscore(explosive_score, x))
pos_group_data["Agility Score"] = pos_group_data["Agility Score"].map(
    lambda x: stats.percentileofscore(agility_score, x))

pos_group_data["Physical Score"] = pos_group_data["Physical Score"].round(2)
pos_group_data["Speed Score"] = pos_group_data["Speed Score"].round(2)
pos_group_data["Explosive Score"] = pos_group_data["Explosive Score"].round(2)
pos_group_data["Agility Score"] = pos_group_data["Agility Score"].round(2)

player_scores = pos_group_data.loc[player_name,
                                   "Physical Score":"Agility Score"]

pos_group_data["Athletic Score"] = pos_group_data[["Physical Score",
                                                   "Speed Score", "Explosive Score", "Agility Score"]].mean(axis=1)

athletic_scores = pos_group_data["Athletic Score"].dropna().to_numpy()

pos_group_data["Athletic Score"] = pos_group_data["Athletic Score"].map(
    lambda x: stats.percentileofscore(athletic_scores, x))
pos_group_data["Athletic Score"] = pos_group_data["Athletic Score"].apply(
    lambda x: 0 if pos_group_data.loc[player_name, "40yd":"Shuttle"].isna().sum() > 4 else x)

pos_group_data["Athletic Score"] = pos_group_data["Athletic Score"].round(2)

athletic_score = pos_group_data.loc[player_name, "Athletic Score"]
print(
    f"{player_name} is a top {athletic_score}% athlete at the {player_data['Pos']} position")
