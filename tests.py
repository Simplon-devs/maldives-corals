import pandas as pd
from stats import get_growth_rate, get_mortality_rate
import matplotlib.pyplot as plt
import seaborn as sns

#Little script for making visualization woth other random datas ...

mainzones = ['Water Villas (LG)', 'Channel (KH)', 'Blue Holes (LG)', 'Parrot Reef (LG)', 'Water Villas (KH)', 'House Reef (KH)', 'Blu (LG)', 'Blu Deep (LG)', 'Dive Site (LG)', 'Coral Trail (LG)', 'Anchor Point (LG)', 'Al Barakat (LG)']

df = pd.read_csv('data_test.csv')

results = {}
for zone in mainzones:
    results[zone] = [get_growth_rate(df, 120, zone)]


results = pd.DataFrame(results).T
print(results)

sns.set(style='darkgrid')
sns.barplot(x=results.index, y=results[0])
plt.xticks(rotation=50, fontsize=7)
plt.title('Taux de croissance moyen par zone')
plt.show()