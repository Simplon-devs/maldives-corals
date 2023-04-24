import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle

from sql import get_mortality, get_bleached_data

mainzones = ['Water Villas (LG)', 'Channel (KH)', 'Blue Holes (LG)', 'Parrot Reef (LG)', 'Water Villas (KH)', 'House Reef (KH)', 
         'Blu (LG)', 'Blu Deep (LG)', 'Dive Site (LG)', 'Coral Trail (LG)', 'Anchor Point (LG)', 'Al Barakat (LG)']
play = False

if play == True:
    # Téléchargement et prétraitement des modèles de mortalité
    survived, dead = get_mortality()
    with open('survived_dead.pkl', 'wb') as f:
        pickle.dump((survived, dead), f)

else:
    with open('survived_dead.pkl', 'rb') as f:
        survived, dead = pickle.load(f)
    roll_window = 120
    mortality_dict = {}
    lines = {}
    fig, axs = plt.subplots(2, sharex = True, figsize = [13,8])


    print(survived['Zone'].unique())

    # Calcul du taux de mortalité pour chaque zone
    for z in survived['Zone'].unique():
        zone_survived_counts = survived['Median'].value_counts()
        zone_dead_counts = dead['Median'].value_counts()
        mortality_dict[z] = zone_dead_counts / (zone_dead_counts + zone_survived_counts)
        mortality_dict[z] = mortality_dict[z].sort_index().rolling(str(roll_window) + 'd').sum() / roll_window * 100
        lines[z], = axs[0].plot_date(mortality_dict[z].index, mortality_dict[z], '-')

        # Entrainement du modèle de régression linéaire pour le taux de mortalité
        print(mortality_dict[z].values)

        X = pd.to_numeric(mortality_dict[z].index).values.reshape(-1,1)
        y = mortality_dict[z].values.reshape(-1,1)
        model = LinearRegression().fit(X,y)

        # Suivi Mlflow
        mlflow.set_experiment('Taux de mortalité')

        # Début traçabilité Mlflow
        with mlflow.start_run(run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")):
            mlflow.log_param("Zone", z)

            # Enregistrement des métriques
            mlflow.log_metric('R²', model.score(X,y))

            # Enregistrement de la figure de mortalité de chaque mainzones
            fig, ax = plt.subplots(figsize=(10, 6))
            x = pd.to_datetime(mortality_dict[z].index).to_pydatetime()
            ax.plot(x, np.array(mortality_dict[z].values), label = z)
            ax.set_xlabel("Temps")
            ax.set_ylabel("Taux de mortalité (%)")
            ax.set_title(f"Taux de mortalité dans la zone {z}")
            ax.text(0.95, 0.95, f"R²={model.score(X, y):.2f}", transform=ax.transAxes, ha="right", va="top")
            fig.autofmt_xdate() # éviter que les informations se chevauchent
            fig.savefig(f"mortality_{z}.png")
            mlflow.log_artifact(f"mortality_{z}.png")
            plt.close(fig)

        mlflow.end_run()