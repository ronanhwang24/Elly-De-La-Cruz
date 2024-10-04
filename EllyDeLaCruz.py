import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''         Elly de la cruz data            '''
elly = pd.read_csv("./ellydelacruz.csv")
elly = elly.drop(columns=['Runner', 'Fielder', 'pitcher_stealing_runs', 'Ball', 'Strike', 'Link'])
elly = elly.sort_values(by = "at_pitch_release")
elly = elly.sort_values(by = "SB/CS")
elly["Jump"] = np.nan
elly['Jump'] = elly['Jump'].astype(float)

'''         Fill Jump            ''' 
for i, row in elly.iterrows():
    elly.loc[i, 'Jump'] = elly.loc[i, 'at_pitch_release'] - elly.loc[i, 'lead_distance_gained']

'''         Print table with only Elly's Stolen Base occurances        '''
elly_sb = elly.copy()
elly_sb.set_index('SB/CS')
for i, row in elly_sb.iterrows():
    temp = row['SB/CS']
    if temp != 'SB' :
        elly_sb.drop(i, axis = 0, inplace = True)

'''         Print table with pitchout occurences when Elly steals        '''
elly_po = elly.copy()
elly_po.set_index('Notes')
for i, row in elly_po.iterrows():
    temp = row['Notes']
    base = row['SB/CS']
    if temp != 'Pitchout' or (temp == 'Pitchout' and base == 'SB'):
        elly_po.drop(i, axis = 0, inplace = True)

'''         Print table with only Elly's caught stealing occurances        '''
elly_cs = elly.copy()
elly_cs.set_index('SB/CS')
for i, row in elly_cs.iterrows():
    temp = row['SB/CS']
    if temp == 'SB':
        elly_cs.drop(i, axis = 0, inplace = True)


'''         Velocity of ball VS Distance from 1B at Pitch Release        '''
ax = elly_sb.plot(kind = "scatter", x='Velocity', y='at_pitch_release', label='Stolen Base', color='green', title = "Velocity of Pitch vs Distance from 1B at Pitch Release")
elly_cs.plot(kind = "scatter", x='Velocity', y='at_pitch_release', label='Caught Stealing', color='red', ax=ax)
elly_po.plot(kind = "scatter", x='Velocity', y='at_pitch_release', label='Pitchout', color='blue', ax=ax)
ax.set_xlabel("Velocity of Pitch (mph)")
ax.set_ylabel("Distance from 1B at Pitch Release (ft)")
plt.show()

'''         Handedness vs Lead Distance           '''
ax = elly_sb.plot(kind = "scatter", x='Handedness', y='lead_distance_gained', label='Stolen Base', color='green', title = "Handedness vs Lead Distance")
elly_cs.plot(kind = "scatter", x='Handedness', y='lead_distance_gained', label='Caught Stealing', color='red', ax=ax)
elly_po.plot(kind = "scatter", x='Handedness', y='lead_distance_gained', label='Pitchout', color='blue', ax=ax)
ax.set_xlabel("Handedness")
ax.set_ylabel("Lead Distance from 1B (ft)")
plt.show()

'''         Handedness vs Distance from 1B at pitch release           '''
ax = elly_sb.plot(kind = "scatter", x='Handedness', y='at_pitch_release', label='Stolen Base', color='green', title = "Handedness vs Distance from 1B at Pitch Release")
elly_cs.plot(kind = "scatter", x='Handedness', y='at_pitch_release', label='Caught Stealing', color='red', ax=ax)
elly_po.plot(kind = "scatter", x='Handedness', y='at_pitch_release', label='Pitchout', color='blue', ax=ax)
ax.set_xlabel("Handedness")
ax.set_ylabel("Distance from 1B at Pitch Release (ft)")
plt.show()
'''         Jump vs Stolen Base Outcome        '''
ax = elly_sb.plot(kind = "scatter", x='Jump', y='SB/CS', label='Stolen Base', color='green', title = "Jump vs Stolen Base Outcome")
elly_cs.plot(kind = "scatter", x='Jump', y='SB/CS', label='Caught Stealing', color='red', ax=ax)
elly_po.plot(kind = "scatter", x='Jump', y='SB/CS', label='Pitchout', color='blue', ax=ax)
ax.set_ylabel("Outcome")
plt.gca().invert_yaxis()
ax.set_xlabel("Jump (ft)")
plt.show()

