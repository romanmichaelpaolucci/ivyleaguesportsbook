import streamlit as st
from datetime import datetime
import nfl_data_py as nfl
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binom
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import re
from scipy import stats
import ssl
import yaml
from yaml.loader import SafeLoader
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Ignore SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Your data downloading class, replace this with actual implementation
import pandas as pd
import nfl_data_py as nfl # nfl_data_py-0.3.1.tar.gz
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class NFLTeamTools:
    
    # compute the win/loss for the home team in general, the away team in general, and the home team against the away team
    def compute_winloss(self, team, home=True):
        '''
            team : str - city code of the team to find the spread for (Giants = NYG)
            
            home : bool - considering home games or away games
        
        '''
        
        # get data for years
        df = self.data
        
        # Filter by team, create binary wins and spread
        if home:
            team = df[df['home_team'] == team]
            team["_win"] = team["home_score"] > team["away_score"]
            team["spread"] = np.abs(team["home_score"] - team["away_score"])
        else:
            team = df[df['away_team'] == team]
            team["_win"] = team["away_score"] > team["home_score"]
            team["spread"] = np.abs(team["home_score"] - team["away_score"])
        
        # unique games played against each other
        df = df.drop_duplicates(subset='game_id')

        # compute the wins and losses for home or away
        s = {"win":0, "loss":0}

        # for each game get the outcome and add to w/l
        for game in team['game_id'].unique():
            if team[team["game_id"] == game]['_win'].values[0] == 1:
                s["win"] += 1
            else:
                s["loss"] += 1
                
        return s 
    
    def compute_spread(self, team, home=True):
        '''
            team : str - city code of the team to find the spread for (Giants = NYG)
            
            home : bool - considering home games or away games
        
        '''
        
        # get data for years
        df = self.data

        # Filter by team, create binary wins and spread
        if home:
            team = df[df['home_team'] == team]
            team["_win"] = team["home_score"] > team["away_score"]
            team["spread"] = np.abs(team["home_score"] - team["away_score"])
        else:
            team = df[df['away_team'] == team]
            team["_win"] = team["away_score"] > team["home_score"]
            team["spread"] = np.abs(team["home_score"] - team["away_score"])

        # compute the spread for wins and losses
        s = {"win":[], "loss":[]}

        # for each game get the outcome and spread
        for game in team['game_id'].unique():
            if team[team["game_id"] == game]['_win'].values[0] == 1:
                s["win"].append(team[team["game_id"] == game]['spread'].values[0])
            else:
                s["loss"].append(team[team["game_id"] == game]['spread'].values[0])

        return s
    
    def spread_dist_function(self, team, spread=0, win=True, home=True):
        '''
            Spread Distribution Function - returns P(SPREAD <= spread | win/loss)
            
            team : str - city code of the team to find the spread for (Giants = NYG)
            
            spread : float - game spread to find probability for
            
            win : bool - whether the team will win or lose
            
            home : bool - considering home games or away games
        
        '''
        
        # calculate spread for wins and losses
        spread_data = self.compute_spread(team, home)
        
        # find probability of spread given win/loss
        if win:
            c = 0
            # for all scores in wins
            for s in spread_data['win']:
                if s <= spread:
                    c+=1 # increment if score beats spread
            return c/len(spread_data['win'])
        else:
            c = 0
            # for all scores in losses
            for s in spread_data['loss']:
                if s <= spread:
                    c+=1 # increment if score beats spread
            return c/len(spread_data['loss'])
    
    def confidence_interval(self, group, confidence=0.95):
        # Calculate the mean and standard error
        mean = group.mean()
        sem = stats.sem(group)
        
        # Get the size of the group (number of observations)
        n = len(group)
        
        # Calculate the confidence interval
        margin = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
        
        # Calculate the spread (range of the lower and upper bound)
        spread = (mean + margin) - (mean - margin)
        
        return pd.Series([mean, mean - margin, mean + margin, n, spread], index=['Mean', 'Lower_Bound', 'Upper_Bound', 'Plays', 'Spread'])

    def get_top_rushers(self):
        g = self.data[['rusher_id', 'rushing_yards']].groupby('rusher_id').apply(self.confidence_interval).reset_index()
        g2 = self.data[['rusher_jersey_number', 'rusher_id', 'rusher']].drop_duplicates()
        merged_df = pd.merge(g, g2, on='rusher_id')
        merged_df['mean'] = merged_df['Mean'].astype('float')
        merged_df['95% lb'] = merged_df['Lower_Bound'].astype('float')
        merged_df['95% ub'] = merged_df['Upper_Bound'].astype('float')
        merged_df['spread'] = merged_df['Spread'].astype('float')
        merged_df['plays'] = merged_df['Plays'].astype('float')
        return merged_df[['rusher_jersey_number', 'rusher', 'plays', 'mean', '95% lb', '95% ub', 'spread']].sort_values('mean', ascending=False)
    
    def get_top_receivers(self):
        g = self.data[['receiver_id', 'receiving_yards']].groupby('receiver_id').apply(self.confidence_interval).reset_index()
        g2 = self.data[['receiver_id', 'receiver_jersey_number', 'receiver']].drop_duplicates()
        merged_df = pd.merge(g, g2, on='receiver_id')
        merged_df['mean'] = merged_df['Mean'].astype('float')
        merged_df['95% lb'] = merged_df['Lower_Bound'].astype('float')
        merged_df['95% ub'] = merged_df['Upper_Bound'].astype('float')
        merged_df['spread'] = merged_df['Spread'].astype('float')
        merged_df['plays'] = merged_df['Plays'].astype('float')
        return merged_df[['receiver_jersey_number', 'receiver', 'plays', 'mean', '95% lb', '95% ub', 'spread']].sort_values('mean', ascending=False)
    
    def __init__(self, years):
        self.data = nfl.import_pbp_data(years)

# Initialize Streamlit session state if it's not already initialized
if 'session_state' not in st.session_state:
    current_year = datetime.now().year
    st.session_state['session_state'] = {'start_date': current_year, 'end_date': current_year, 'display_date': str(current_year)}
    st.session_state['tools'] = NFLTeamTools([current_year])

def main():
    st.title('NFL Probability Dashboard')
    
    # Create a placeholder for the header
    header_placeholder = st.empty()

    # Populate it with the initial data range
    header_placeholder.markdown('<h1 id="headerID">Data for: ' + st.session_state['session_state']['display_date'] + '</h1>', unsafe_allow_html=True)
    
    # Sidebar for Date Range Selector with Dropdowns
    st.sidebar.header("Date Range Selector")
    start_year_options = list(range(2018, 2024))
    end_year_options = list(range(2018, 2024))

    start_date = st.sidebar.selectbox("Start Year", start_year_options, index=0)
    end_date = st.sidebar.selectbox("End Year", end_year_options, index=len(end_year_options) - 1)

    submit_clicked = st.sidebar.button('Submit')
    
    st.sidebar.header('About Me')
    # Display profile picture
    st.sidebar.image("roman.jpeg", caption='Roman Paolucci, Columbia MS Student', use_column_width=True)
        # Text description
    st.sidebar.markdown("""
        Hi there! I'm a Columbia engineering student developing a set of probability tools to aid in optimal betting on NFL games. The tools are free to use for now so enjoy!
    """)

    if submit_clicked:
        if st.session_state['session_state']['start_date'] != start_date or st.session_state['session_state']['end_date'] != end_date:
            # Update session state
            st.session_state['session_state']['start_date'] = start_date
            st.session_state['session_state']['end_date'] = end_date
            
            # Update display date
            if start_date == end_date:
                st.session_state['session_state']['display_date'] = str(start_date)
            else:
                st.session_state['session_state']['display_date'] = f"{start_date}-{end_date}"

            # Update header using placeholder
            header_placeholder.markdown('<h1 id="headerID">Data for: ' + st.session_state['session_state']['display_date'] + '</h1>', unsafe_allow_html=True)

            # Update tools variable with new data
            if start_date == end_date:
                st.session_state['tools'] = NFLTeamTools([start_date])
            else:
                st.session_state['tools'] = NFLTeamTools(list(range(start_date, end_date + 1)))

    # Define the available tabs
    tabs = ['Probability Tools', 'Rushing & Receiving Tools', 'Wager Tools', 'Machine Learning Tools']
    tab1, tabtp, tab2, tab3 = st.tabs(tabs)
  
    with tab1:
        probability_view()
        
    with tabtp:
        rushing_and_receiving_view()
  
    with tab2:
        wagers_view()
  
    with tab3:
        machine_learning_view()

def rushing_and_receiving_view():
    st.header('Rushing & Receiving Tools (Play by Play)')
    
    # Generate tables with your code (replace this with your actual code)
    top_player_receiving_yards_df = st.session_state['tools'].get_top_receivers()  # Replace with your actual data
    top_player_rushing_yards_df = st.session_state['tools'].get_top_rushers()  # Replace with your actual data
    
    # Adding search and filter functionality for "Top Player Receiving Yards" table
    st.subheader("Player Receiving Yards")
    filter_text_receiving = st.text_input("Search and filter (Receiving Yards):", key='receiving_filter')
    filtered_receiving_df = top_player_receiving_yards_df[
        top_player_receiving_yards_df.apply(lambda row: row.astype(str).str.contains(filter_text_receiving).any(), axis=1)]
    st.dataframe(filtered_receiving_df)
    
    st.write("---")  # Optional horizontal line for better separation

    # Adding search and filter functionality for "Top Player Rushing Yards" table
    st.subheader("Player Rushing Yards")
    filter_text_rushing = st.text_input("Search and filter (Rushing Yards):", key='rushing_filter')
    filtered_rushing_df = top_player_rushing_yards_df[
        top_player_rushing_yards_df.apply(lambda row: row.astype(str).str.contains(filter_text_rushing).any(), axis=1)]
    st.dataframe(filtered_rushing_df)

def probability_view():
    st.header('Probability View')
    
    df = st.session_state['tools'].data
    unique_home_teams = pd.unique(df['home_team'])
    unique_away_teams = pd.unique(df['away_team'])
    
    all_unique_teams = list(set(unique_home_teams) | set(unique_away_teams))
    
    col1, col2 = st.columns(2)
    selected_home_team = col1.selectbox("Select Home Team", sorted(all_unique_teams))
    selected_away_team = col2.selectbox("Select Away Team", sorted(all_unique_teams))
    
    home_winloss = st.session_state['tools'].compute_winloss(selected_home_team, True)
    away_winloss = st.session_state['tools'].compute_winloss(selected_away_team, False)
    
    total_home_games = home_winloss['win'] + home_winloss['loss']
    total_away_games = away_winloss['win'] + away_winloss['loss']
    
    home_win_prob = home_winloss['win'] / total_home_games
    home_loss_prob = home_winloss['loss'] / total_home_games
    
    away_win_prob = away_winloss['win'] / total_away_games
    away_loss_prob = away_winloss['loss'] / total_away_games
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader(f'Win/Loss Probability for {selected_home_team} (Home) [Total Games: {total_home_games}]')
        st.bar_chart({'Win': [home_win_prob], 'Loss': [home_loss_prob]})
        
    with chart_col2:
        st.subheader(f'Win/Loss Probability for {selected_away_team} (Away) [Total Games: {total_away_games}]')
        st.bar_chart({'Win': [away_win_prob], 'Loss': [away_loss_prob]})
    
    # New Section: Spread Over Value Probability
    st.subheader("Spread Over Value Probability")
    
    # Get spreads for selected teams using compute_spread
    home_spread = st.session_state['tools'].compute_spread(selected_home_team, True)
    away_spread = st.session_state['tools'].compute_spread(selected_away_team, False)
    
    # Create a slider to set the value against which the spread is compared
    spread_value = st.slider("Spread Value", min_value=0, max_value=50, value=0)
    
    # Create columns for side-by-side bar charts
    spread_chart_col1, spread_chart_col2 = st.columns(2)

    # Calculate probabilities of the spread being over the specified value
    home_win_over_prob = sum(1 for x in home_spread['win'] if x > spread_value) / len(home_spread['win']) if len(home_spread['win']) > 0 else 0
    home_loss_over_prob = sum(1 for x in home_spread['loss'] if x > spread_value) / len(home_spread['loss']) if len(home_spread['loss']) > 0 else 0
    away_win_over_prob = sum(1 for x in away_spread['win'] if x > spread_value) / len(away_spread['win']) if len(away_spread['win']) > 0 else 0
    away_loss_over_prob = sum(1 for x in away_spread['loss'] if x > spread_value) / len(away_spread['loss']) if len(away_spread['loss']) > 0 else 0

    # Create a bar chart for home team probabilities
    with spread_chart_col1:
        st.subheader(f'Spread Over Probabilities for {selected_home_team} (Home)')
        st.bar_chart({
            'Win Over': [home_win_over_prob], 
            'Loss Over': [home_loss_over_prob]
        })

    # Create a bar chart for away team probabilities
    with spread_chart_col2:
        st.subheader(f'Spread Over Probabilities for {selected_away_team} (Away)')
        st.bar_chart({
            'Win Over': [away_win_over_prob], 
            'Loss Over': [away_loss_over_prob]
        })

def wagers_view():
    st.header('3 Pick Parlay Wager Tool')
    # Three sliders for probabilities side by side
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prob1 = st.slider('Probability 1', min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    with col2:
        prob2 = st.slider('Probability 2', min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    with col3:
        prob3 = st.slider('Probability 3', min_value=0.0, max_value=1.0, value=0.7, step=0.01)

    # Three textboxes for Parlays, Odds, and Wager side by side
    col4, col5, col6 = st.columns(3)
    
    with col4:
        parlays = st.text_input('Parlays', '3')
    with col5:
        odds = st.text_input('Odds', '+100')
    with col6:
        wager = st.text_input('Wager', '100')

    if parlays and odds and wager:
        parlays = int(parlays)
        odds = float(odds.replace('+', ''))  # Remove the '+' sign if it's there
        wager = float(wager)

        # Convert American odds to decimal
        decimal_odds = odds / 100 + 1

        # Combined probability of winning a single parlay
        combined_prob = prob1 * prob2 * prob3

        # Generate profit data based on the binomial distribution
        x = range(parlays + 1)  # Number of successes
        y = [binom.pmf(k, parlays, combined_prob) for k in x]  # Probability of k successes

        profit = []
        for k in x:
            profit.append((k * (wager * decimal_odds)) - ((parlays - k) * wager))

        # Create a DataFrame to hold the data
        df = pd.DataFrame({
            'Successes': x,
            'Profit': profit,
            'Probability': y
        })

        # Create the bar chart with axis labels
        chart = st.bar_chart(df.set_index('Successes')['Profit'], use_container_width=True)
        st.write("X-axis: Number of Successes, Y-axis: Profit")

        # Show the probability when hovering (serves as a hover-over tooltip alternative)
        st.table(df)
    
def parse_wind(weather_str):
    # Extract wind speed from weather string
    try:
        wind_speed = int(weather_str.split('Wind:')[-1].split()[1])
    except:
        return 0 # assume zero windspeed if not recorder (indoors)
    return wind_speed    

def machine_learning_view():
    df = st.session_state['tools'].data[['kick_distance', 'field_goal_result', 'weather']].dropna()
    df = df[df['field_goal_result'].isin(["made", "missed"])]
    
    st.title("Field Goal Prediction with Support Vector Classifier")
    
    st.text("Research Question: Does wind have a significant impact on field goal completion?")

    # Convert 'made' to 1 and 'missed' to 0
    df = df[df['field_goal_result'].isin(['made', 'missed'])]
    df['weather'] = df['weather'].apply(parse_wind)
    df['wind'] = df['weather']
    df['field_goal_result'] = df['field_goal_result'].map({'made': 1, 'missed': 0})

    # Add a slider for minimum kick distance
    min_kick_distance = st.slider("Minimum Kick Distance", 0, 60, value=0)
    filtered_df = df[df['kick_distance'] >= min_kick_distance]

    # Prepare data for both models
    X_with_wind = filtered_df[['kick_distance', 'wind']]
    X_no_wind = filtered_df[['kick_distance']]
    y = filtered_df['field_goal_result']

    # Train-Test Split for both models
    X_train_with_wind, X_test_with_wind, y_train, y_test = train_test_split(X_with_wind, y, test_size=0.2, random_state=42)
    X_train_no_wind, X_test_no_wind = train_test_split(X_no_wind, test_size=0.2, random_state=42)

    # Model Training with Loading Spinner
    with st.spinner('Training Models...'):
        # With Wind
        model_with_wind = SVC(probability=True)
        model_with_wind.fit(X_train_with_wind, y_train)
        accuracy_with_wind = accuracy_score(y_test, model_with_wind.predict(X_test_with_wind))

        # Without Wind
        model_no_wind = SVC(probability=True)
        model_no_wind.fit(X_train_no_wind, y_train)
        accuracy_no_wind = accuracy_score(y_test, model_no_wind.predict(X_test_no_wind))

    # Display Accuracy
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart({'Accuracy with Wind': [accuracy_with_wind]}, use_container_width=True)
    with col2:
        st.bar_chart({'Accuracy without Wind': [accuracy_no_wind]}, use_container_width=True)

    # Scatter Plot
    st.subheader("Scatter Plot")
    plt.scatter(X_with_wind['kick_distance'], X_with_wind['wind'], c=y, cmap='coolwarm')
    plt.xlabel("Kick Distance")
    plt.ylabel("Wind")
    plt.colorbar(label='Field Goal Made (1) or Missed (0)')
    st.pyplot(plt)

    # Display Data Table with filters
    st.subheader('Data Table')
    cols_to_show = ["field_goal_result", "kick_distance", "wind"]
    table_filter = st.selectbox("Filter by Field Goal Result", ["All", "Made", "Missed"])
    if table_filter == "All":
        st.write(filtered_df[cols_to_show])
    elif table_filter == "Made":
        st.write(filtered_df[filtered_df['field_goal_result'] == 1][cols_to_show])
    else:
        st.write(filtered_df[filtered_df['field_goal_result'] == 0][cols_to_show])


if __name__ == '__main__':
    main()
