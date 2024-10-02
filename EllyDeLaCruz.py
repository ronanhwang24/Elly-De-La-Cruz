import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

'''
Helper function for find_percentile
Uses Binary search to find the index of the value in the given dataframe
'''
def binary_search_df(df, column, target):
    left, right = 0, len(df) - 1
    while left <= right:
        mid = (left + right) // 2
        mid_value = df.iloc[mid][column] 
        if mid_value == target:
            return mid  # Return the index of the target row
        elif mid_value > target:
            left = mid + 1
        else:
            right = mid - 1
    return right  # Return the index of where it should be if not found

'''     Finds index of where parameter velocity should be in velocity chart     '''
def find_percentilev(df, temp):
    # current_velo = allvelos.loc[0, 'Velo Chart']
    index = binary_search_df(df, 'Velo Chart', temp)
    return index

'''     Finds index of where parameter pop_2b_sba should be in pop_time chart     '''
def find_percentilep(df, temp):
    # current_velo = allvelos.loc[0, 'Velo Chart']
    index = binary_search_df(df, 'pop_2b_sba', temp)
    return index

'''     Turns full name e.g. Shohei Ohtani into Ohtani, S.      '''
def shorten_name(name):
    # Split the name into last name and first name
    last_name, first_name = name.split(', ')
    
    # Get the first letter of the first name
    first_initial = first_name[0] + '.'
    
    # Combine last name and first initial
    shortened_name = f"{last_name}, {first_initial}"
    
    return shortened_name

'''     Print the top 15 2B stolen base leaders     '''
dataset2 = pd.read_csv("./br_datah.csv")
dataset2 = dataset2.sort_values(by = "r_stolen_base_2b", ascending = [False])
print(dataset2.head(15))

print("\n \n \n ")
print("\n \n \n ")

'''         Elly de la cruz data            '''
elly = pd.read_csv("./ellydelacruz.csv")
elly = elly.drop(columns=['Runner', 'Fielder', 'pitcher_stealing_runs', 'Ball', 'Strike', 'Link'])
elly = elly.sort_values(by = "at_pitch_release")
elly = elly.sort_values(by = "SB/CS")
elly["Velo Percentile"] = np.nan
elly['Velo Percentile'] = elly['Velo Percentile'].astype(float)
elly["Jump"] = np.nan
elly['Jump'] = elly['Jump'].astype(float)
elly["Pop Time Percentile"] = np.nan
elly['Pop Time Percentile'] = elly['Pop Time Percentile'].astype(float)
elly["Jump Percentile"] = np.nan
elly['Jump Percentile'] = elly['Jump Percentile'].astype(float)
elly["Sum"] = np.nan
elly['Sum'] = elly['Sum'].astype(float)

'''         #All Velos Data          '''
allvelos = pd.read_csv("./allvelos.csv")
allvelos = allvelos.drop_duplicates(subset=['Velo Chart'])
allvelos = allvelos.sort_values(by = "Velo Chart", ascending = [False])
allvelos.reset_index(drop = True)

'''         Fill Jump            ''' 
for i, row in elly.iterrows():
    elly.loc[i, 'Jump'] = elly.loc[i, 'at_pitch_release'] - elly.loc[i, 'lead_distance_gained']

'''         Obtain the percentiles for each velo            ''' 
for i, row in elly.iterrows():
    temp = elly.loc[i, 'Velocity']                          #Change temp for user input velocity
    percentile = find_percentilev(allvelos,temp)
    total_rows = len(allvelos)
    final_p = ((total_rows - percentile) / total_rows)*100      #Change final_p for user input velocity
    elly.loc[i, 'Velo Percentile'] = 100 - final_p#  Row index 1, column 'Age'
# print(elly)

'''         Remove Occurences where Ellies gets 'FB' or 'PK'            ''' 
elly_f = elly.copy()
elly_f.set_index('SB/CS')
for i, row in elly.iterrows():
    temp = row['SB/CS']
    if temp == 'PK' or temp == 'FB':
        elly_f.drop(i, axis = 0, inplace = True)
# print(elly_f)


'''         Print table with only Elly's Stolen Base occurances        '''
elly_sb = elly.copy()
elly_sb.set_index('SB/CS')
for i, row in elly_sb.iterrows():
    temp = row['SB/CS']
    if temp != 'SB' :
        elly_sb.drop(i, axis = 0, inplace = True)
# print(elly_sb)

'''         Print table with pitchout occurences when Elly steals        '''
elly_po = elly.copy()
elly_po.set_index('Notes')
for i, row in elly_po.iterrows():
    temp = row['Notes']
    base = row['SB/CS']
    if temp != 'Pitchout' or (temp == 'Pitchout' and base == 'SB'):
        elly_po.drop(i, axis = 0, inplace = True)
# print(elly_po)


'''         Print table with only Elly's caught stealing occurances        '''
elly_cs = elly.copy()
elly_cs.set_index('SB/CS')
for i, row in elly_cs.iterrows():
    temp = row['SB/CS']
    if temp == 'SB':
        elly_cs.drop(i, axis = 0, inplace = True)
# print(elly_cs)


'''         Print table with Catcher's pop time's        '''
pop_time = pd.read_csv("./poptime.csv")
pop_time = pop_time.drop(columns=['player_id','age','maxeff_arm_2b_3b_sba', 'exchange_2b_3b_sba', 'pop_3b_sba_count', 'pop_3b_sba', 'pop_3b_cs', 'pop_3b_sb' ])
pop_time = pop_time.sort_values(by = "pop_2b_sba", ascending = [True])
pop_time.reset_index(drop = True, inplace=True)
pop_time["Percentile"] = np.nan
pop_time['Percentile'] = pop_time['Percentile'].astype(float)
total_rows = len(pop_time)

'''             Fill percentiles in pop_time            '''
for i, row in pop_time.iterrows():
    name = row['catcher']
    shortened_name = shorten_name(name)
    pop_time.loc[i, 'catcher'] = shortened_name

    temp = i                                                        #NEED TO FIGURE OUT HOW TO CHECK POP TIME WHILE COMPARING WITH OTHER POP TIMES
    total_rows = len(pop_time)
    final_p = ((total_rows - i )/ (total_rows + 1))*100
    pop_time.loc[i, 'Percentile'] = 100 - final_p
print(pop_time.head(15))

'''             Fill catcher percentiles in Elly            '''
for i, row in elly_f.iterrows():
    name = row['Catcher']
    temp_df = pop_time[pop_time['catcher'].str.contains(name, case=True)]
    if not temp_df.empty :
        catcher_percentile = temp_df.iloc[0]['Percentile']                  #Change temp for user input pop time
    else:
        catcher_percentile = 50.0000            #Set catcher pop time perceltile to average if catcher did not throw out Elly - Catcher did not appear in Elly dataframe
    
    elly_f.loc[i, 'Pop Time Percentile'] = catcher_percentile

'''             Fill Elly'sjump percentiles            '''
elly_f = elly_f.sort_values(by = "Jump", ascending = [False])
elly_f.reset_index(drop = True, inplace=True)
for i, row in elly_f.iterrows():

    temp = i
    total_rows = len(elly_f)
    jump_p = ((total_rows - i )/ (total_rows + 1))*100
    elly_f.loc[i, 'Jump Percentile'] = jump_p

'''             Get the sum of the jump percentiles, pop time percentiles and pitch velocity perceltiles            '''
for i, row in elly_f.iterrows():

    sum = elly_f.loc[i, 'Jump Percentile'] + elly_f.loc[i, 'Pop Time Percentile'] + elly_f.loc[i, 'Velo Percentile']            #CHANGE INPUT VALUES FOR POP TIME AND VELO HERE
    elly_f.loc[i, 'Sum'] = sum / 300
elly_f = elly_f.sort_values(by = "Sum", ascending = [False])
print(elly_f)

'''         Jump vs Stolen Base Outcome        '''
ax = elly_sb.plot(kind = "scatter", x='Jump', y='SB/CS', label='Stolen Base', color='green', title = "Jump vs Stolen Base Outcome")
elly_cs.plot(kind = "scatter", x='Jump', y='SB/CS', label='Caught Stealing', color='red', ax=ax)
elly_po.plot(kind = "scatter", x='Jump', y='SB/CS', label='Pitchout', color='blue', ax=ax)
ax.set_ylabel("Outcome")
plt.gca().invert_yaxis()
ax.set_xlabel("Jump (ft)")
plt.show()


'''         Velocity of ball VS Distance from 1B at pitch release        '''
ax = elly_sb.plot(kind = "scatter", x='Velocity', y='at_pitch_release', label='Stolen Base', color='green', title = "Velocity of Pitch vs Distance from 1B at pitch release")
elly_cs.plot(kind = "scatter", x='Velocity', y='at_pitch_release', label='Caught Stealing', color='red', ax=ax)
elly_po.plot(kind = "scatter", x='Velocity', y='at_pitch_release', label='Pitchout', color='blue', ax=ax)
ax.set_xlabel("Velocity of Pitch (mph)")
ax.set_ylabel("Distance from 1B at Pitch Release (ft)")
plt.show()

'''         Handedness vs Distance from 1B at pitch release           '''
ax = elly_sb.plot(kind = "scatter", x='Handedness', y='lead_distance_gained', label='Stolen Base', color='green', title = "Handedness vs Lead Distance")
elly_cs.plot(kind = "scatter", x='Handedness', y='lead_distance_gained', label='Caught Stealing', color='red', ax=ax)
elly_po.plot(kind = "scatter", x='Handedness', y='lead_distance_gained', label='Pitchout', color='blue', ax=ax)
ax.set_xlabel("Handedness")
ax.set_ylabel("Lead Distance from 1B (ft)")
plt.show()

'''             Create a new dataframe the sum of the 3 features and the stolen base/ caught stealing boolean            '''
logtable =  pd.DataFrame(columns = ["Sum / 3", "SB | CS"])
for i, row in elly_f.iterrows():
    VAL = row['Sum']
    VAL2 = row['SB/CS']
    if VAL2 == 'PK' or VAL2 == 'FB' or VAL2 == 'CS':
        VAL2 = "0"
    else:
        VAL2 = "1"
    temp_row = pd.DataFrame({'Sum / 3': [VAL], 'SB | CS': [VAL2]})
    logtable = pd.concat([logtable, temp_row], ignore_index=True)

print(logtable)

'''         Logistical Regression Model         '''
X = logtable[['Sum / 3']].values
y = logtable['SB | CS'].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X_scaled, y)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the actual data points
zeros_mask = (y == 0)
ones_mask = (y == 1)
plt.scatter(X[zeros_mask], np.zeros(np.sum(zeros_mask)), color='black', alpha=0.5, label='Actual 0s')
plt.scatter(X[ones_mask], np.ones(np.sum(ones_mask)), color='black', alpha=0.5, label='Actual 1s')

# Generate points for the curve
X_curve = np.linspace(0, 1, 100).reshape(-1, 1)
X_curve_scaled = scaler.transform(X_curve)
y_curve = model.predict_proba(X_curve_scaled)[:, 1]

# Plot the logistic curve
plt.plot(X_curve, y_curve, color='blue', label='Logistic Regression Curve')

# Customize the plot
plt.xlabel('Sum / 3')
plt.ylabel('Probability')
plt.title('Probability vs Sum / 3')
plt.grid(True, alpha=0.3)
plt.legend()
# Set y-axis limits explicitly
plt.ylim(0, 1)

# Show the plot
plt.show()

# Print model coefficients
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Coefficient: {model.coef_[0][0]:.4f}")


data = {'hours_studying': [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        'passed_exam': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]}

# Convert to DataFrame
df = pd.DataFrame(data)

# Feature (hours studying) and target (passed exam)
X = logtable[['Sum / 3']]
y = logtable[['SB | CS']]

# Create logistic regression model
model = LogisticRegression()

# Fit the model
model.fit(X, y)

# Generate a range of hours from 0 to 5 for prediction
X_test = np.linspace(0, 5, 300).reshape(-1, 1)

# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Plot the original data points
plt.scatter(logtable[['Sum / 3']], logtable[['SB | CS']], color='black', zorder=5)

# Plot the logistic regression curve
plt.plot(X_test, y_prob, color='blue', linewidth=2)

# Set labels and title
plt.xlabel('Features')
plt.ylabel('Probability')
plt.title('Probability vs Features')

# Show plot
plt.show()