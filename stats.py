import matplotlib.pyplot as plt
from Statistics.sql import get_growth_data
import pandas as pd

def get_growth_rate(growth_data, roll_window, zone):
    """
    Calculate average growth rate of lived corals for a given area
    Arguments:
    - growth_data: dataframe with growth data necessary to this function, with this following columns: 'Type', 'Zone', 'avgrowth'
    informations only on lived corals and stock informations in livedf dataframe, takes informations of differents dates and make an average of growthering 
    - roll_window: a time window for calculating the moving average
    - zone: zone for which we want to calculate the growth rate
    Needs get_growth_data function for having data of growth's corals
    Return: list of lines drawn corresponding to the different zones or None if no data was found for the specified zone
    """
    zone_growth = growth_data[growth_data['Zone'] == zone]
    live_coral_data = zone_growth[(zone_growth['Type'] == 'Acropora') | (zone_growth['Type'] == 'Pocillopora')]
    live_coral_data = live_coral_data[['avgrowth']]
    if len(live_coral_data) > 0:
        growth_rate = live_coral_data.sort_index().rolling(int(roll_window)).mean() - 1
        growth_rate['Zone'] = zone_growth['Zone']
        growth_rate_mean = growth_rate.groupby('Zone')['avgrowth'].mean() # calcul des moyennes par zone
        return growth_rate_mean.iloc[0]
    else:
        print('No live coral data found for zone', zone)
        return None

def get_mortality_rate(survived, dead, roll_window):
    """
    Calculate mortality rate for each zone using the survived and dead data dataframes
    Arguments:
    - survived: dataframe containing data of survived corals
    - dead: dataframe containing data of dead corals
    - roll_window (int): number of day to use as rolling
    Return: dataframe containing the mortality rate for each zone
    Exemple: Want to calculate the mortality rate for each zone using a rolling window of 7 days, we can use the function like this:
    >>> get_mortality_rate(survived, dead, 7)
    """
    mortality_dict = {}
    fig, ax = plt.subplots()
    for z in survived['Zone'].unique():
        zone_survived_counts = survived[survived['Zone'] ==z].value_counts()
        zone_dead_counts = dead[dead['Zone'] == z].value_counts()
        mortality = zone_dead_counts / (zone_dead_counts + zone_survived_counts)
        mortality = mortality.sort_index().rolling(str(roll_window) + 'd').sum() / roll_window * 100
        mortality_dict[z] = mortality

    ax.plot(mortality.index, mortality, label=z)

    ax.set_xlabel('Time')
    ax.set_ylabel('Mortality Rate')
    ax.legend()

    return fig

def get_bleaching_rate(bleached_data, zone, axs):
    """
    Arguments:
    - bleached_data: dataframe which contains bleached data
    - zone
    - axs: matplotlib.axes, draw bleached chart
    Return: the bleaching rate for the specified zone
    """
    bleached_dict = {}

    # Sélection des données pour la zone donnée
    zone_data = bleached_data[bleached_data['Zone'] == zone]
    # Calcul du taux de blanchissement pour chaque data d'observation
    bleached_dict = zone_data.groupby(['ObsDate'])['Outcome'].apply(lambda x: (x == 'Bleached Corail').sum() / len(x) * 100)
    # Graphique de la zone donnée
    lines, = axs.plot_date(bleached_dict.index, bleached_dict, '-', label=zone)

    return lines

def get_growth_coral(growth_data, roll_window):
    """
    Function for having growth rate of a coral by its ID
    Return: curve for an ID coral
    """
    coral_ids = growth_data['FragmentID'].unique()
    figs = []
    i = 0
    while i < len(coral_ids):
        coral_id = coral_ids[i]
        coral_data = growth_data[growth_data['FragmentID'] == coral_id][['avgrowth']]
        growth_rate = coral_data.sort_index().rolling(int(roll_window)).mean() - 1
        time = coral_data.index
        fig, ax = plt.subplots()
        ax.plot(time, growth_rate)
        ax.set_xlabel('Time')
        ax.set_ylabel('Growth rate')
        ax.set_title(f'Coral {coral_id}')
        figs.append(fig)
        i += 1
    return figs
    



