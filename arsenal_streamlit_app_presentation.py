import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importing Dataset

df = pd.read_csv("df_full_premierleague.csv")


# Code for the grqphs and dataframes
# Read the other python file for proper explanations of this code

df.drop(['Unnamed: 0', 'link_match'], axis=1, inplace=True)
df_arsenal = df.loc[(df['home_team'] == 'Arsenal') | (df['away_team'] == 'Arsenal')]
df_arsenal[['HomeGoals', 'AwayGoals']] = df_arsenal['result_full'].str.split('-', expand = True)
df_arsenal.drop('result_full', axis=1, inplace=True)
df_arsenal.drop('result_ht', axis=1, inplace=True)
df_arsenal = df_arsenal.astype({'HomeGoals': int, 'AwayGoals': int})
df_arsenal_home = df_arsenal.loc[(df['home_team'] == 'Arsenal')]
df_arsenal_away = df_arsenal.loc[(df['away_team'] == 'Arsenal')]
df_arsenal_home['GoalDifference'] = df_arsenal_home['HomeGoals'] - df_arsenal_home['AwayGoals']
df_arsenal_away['GoalDifference'] = df_arsenal_away['HomeGoals'] - df_arsenal_away['AwayGoals']
df_arsenal_home['Win'] = df_arsenal_home['GoalDifference'] > 0
df_arsenal_home['Win'] = df_arsenal_home['Win'].astype(int)
df_arsenal_home['Draw'] = df_arsenal_home['GoalDifference'] == 0
df_arsenal_home['Draw'] = df_arsenal_home['Draw'].astype(int)
df_arsenal_home['Loss'] = df_arsenal_home['GoalDifference'] < 0
df_arsenal_home['Loss'] = df_arsenal_home['Loss'].astype(int)
df_arsenal_away['Win'] = df_arsenal_away['GoalDifference'] < 0
df_arsenal_away['Win'] = df_arsenal_away['Win'].astype(int)
df_arsenal_away['Draw'] = df_arsenal_away['GoalDifference'] == 0
df_arsenal_away['Draw'] = df_arsenal_away['Draw'].astype(int)
df_arsenal_away['Loss'] = df_arsenal_away['GoalDifference'] > 0
df_arsenal_away['Loss'] = df_arsenal_away['Loss'].astype(int)
df_arsenal_home2 = df_arsenal_home.iloc[:,np.r_[2:4,-6:-0]] 
dataframes = [df_arsenal_home, df_arsenal_away]
df_arsenal = pd.concat(dataframes)
Totals = df_arsenal.groupby(['season'])[['Win', 'Draw', 'Loss']].sum()


# Creating functions for streamlit

def inject_CSS_table(xzy):
    # CSS to inject contained in a string
    hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

    # Display a static table
    st.table(xzy)


## Starting streamlit code
# Creating a sidebar

with st.sidebar:
    selected = option_menu(
        menu_title = "Arsenal FC Project",
        options = ["Objective", "The Data Story", "The Dataset", "Cleaning the Data", "Exploring the Data",
                  "Machine Learning", "Analysing the Results", "Understanding the Results", "Power BI", "Conclusion", "References"],
        icons = ["trophy", "book", "clipboard-data", "tools", "search", "pc-display-horizontal", "graph-up-arrow",
                "lightbulb", "terminal", "hand-thumbs-up", "sliders"]
    )
    
## Creating the pages
# "Objective" Page
    
if selected == "Objective":
        st.title("Arsenal FC: What Happened?")
        st.image('arsenalsad.webp')
        st.subheader('Objective')
        st.write('The objective of the project is to develop a data story that explains why the quality of Arsenal FC '
                 'deteriorated over the previous decade. A logical and coherent explanation must be created using official '
                 'data.')
        st.subheader('Description of Arsenal FC')
        st.write('Arsenal FC is an English Premier League club based in London. The club has won 13 league titles, '
                 'a record 14 FA Cups, and it has won the third-most amount of trophies in all of English football. '
                 'Arsène Wenger, a former Arsenal coach from 1996 to 2018, directed the team to a record seven FA Cups '
                 'and he also managed over their longest unbeaten league run of 49 straight games. (wikipedia)')

        
# "The Data Story" Page

elif selected == "The Data Story":        
        st.subheader('A Decline of One of the Greats')
        st.write('A datastory about the downfall of Arsenal FC.')
        st.image('arsenal_sad.jpg')
        st.write('Being an Arsenal football fan has brought some of the greatest feelings in sport. You’ve seen spectacular '
                 'goals, nail-biting finishes and most importantly you’ve felt the glory of winning premierships! '
                 'The club has a long history of triumpth ' 
                 'coming from the greatest footballers who’ve ever played the game. Orson Scott Card, author, once said, '
                 '"At some clubs success is accidental. At Arsenal it is compulsory".')
        st.write('However, in the last few years of the 2010s Arsenal had lost some of its shine. The team was losing '
                 'more games than usual and their wins were on a downward trend. The question is, **what happened?**')
        st.markdown('#')
        st.subheader('The Downfall')    
        st.write('First, lets look at how big the downfall of Arsenal has been in the past decade.')
        st.write('Below is a stacked bar chart of their win, loss, draw record for each season in the 2010s and also their '
                 'ranking at the end of the season.')
        st.image('rankingsresults.png')
        st.write('In the last two seasons Arsenal had won less games than in any other season of the decade. In fact in season '
                 '19/20 they won 42% less games than their peak season of 13/14 where they had won 24 games. As a result they '
                 'plummeted down the ladder to finish in 10th.')
        st.write('Throughout the decade their losses also grew. In the last four seasons their losses were 42 and it almost '
                 'matched the number of losses in the first six seasons of 46.')
        st.markdown('#')
        st.subheader('So What Happened?')
        st.write('Obviously Arsenal had struggled in the last few seasons of the decade, and the questions is why? '
                 'Instinctively you might think Arsenal weren’t scoring enough goals or maybe they weren’t producing as many players '
                 'that were up to premier league standard. Those would be good guesses but we wanted to know for sure. '
                 'So we crunched the data, fed the learning machines and have come up with some interesting results.')
        st.image('machinelearning.jpg')
        st.write('Believe it or not but the answer is Aresenal needed **more clearances**!')
        st.write('Confused? Let us explain.')
        st.markdown('#')   
        st.subheader('What are Clearances?') 
        st.image('clearance.png')
        st.write('A clearance is when a player kicks the ball away from their goal as a form of defence. The player may kick '
                 'the ball without the intention of passing it to a team mate, however it’s considered a successful play as '
                 'long as the ball has been advanced up the field and further from their goal. ')        
        st.write('Let’s look at Arsenals average clearances per game throughout the seasons.')
        st.image('arsenal_clearances.png')
        st.write('Since Arsenal’s season of 13/14 there has been an alarming downward trend in their average clearances '
                'per game. There was a decline of 49.5% since 13/14!')
        st.write('On its own this seems to be very alarming. However, the number of clearances a team performs does not tell the entire '
                 'story of a match nor of their season. For '
                 'example in season 11/12 Arsenal had a relatively low clearance rate of 20 clearances per game and they managed '
                 'to finish 3rd. Perhaps a reason for their success that year was that they had an incredibly strong mid-field '
                 'defense and they weren’t reliant on clearing the ball from deep in their territory. '
                 'So our question of why Arsenal declined in form won’t be answered by clearances alone. To find out the reasons, '
                 'we need to go deeper into the data to determine why Arsenal wasn’t clearing the ball as much in their later seasons. '
                 'We need to know where the ball was '
                 'actually going.')
        st.markdown('#') 
        st.subheader('Where was the Ball?') 
        st.write('There is strong evidence that shows that Arsenal’s opposition where able to evade their defence and attack more often. '
                'A clear example of this was the opposition shots on goal per game increased dramatically in the last couple of seasons '
                 'of the decade.')
        st.image('opposition_shots.png')
        st.write('Another example is Arsenal’s opposition corners per game also followed a similar increase.')
        st.image('opposition_corners.png')
        st.write('Opposition corners increased dramatically in the last year of the decade.')
        st.write('A corner allows the opposition a set shot close to the goal and according to TheAnalyst, 13.6% of goals in the '
                 'Premier League (measued across five seasons) come from corner situations. Very often a good corner is the '
                 'difference between victory and defeat.')
        st.write('For example, who can forget the Tottenham Hotspur v Arsenal game in the 19/20 season when the Spurs earned a '
                 'comeback win deep '
                'in the game. With nine minutes remaining the scores were tied 1-1 when Toby Alderweireld headed in the decisive goal '
                'from Son Heung-Min’s corner cross. It gave the Spurs a 2-1 scoreline which they held onto for a famous win.')
        st.write('It was a painful moment for Arsenal fans. They had the ascendancy until the Spurs produced beautiful cross and a quick '
                 'strike to snatch the win. '
                 'If only Arsenal was able to block the goal and produce a counter-strike, who knows how the game would have '
                 'finished. However, if we look at the '
                 'stats a different story can be read and we are reminded how important defense and clearances are. '
                 'In the game Arsenal had a huge 62.7% '
                 'possession of the ball, yet the Spurs were able to take fifteen shots on goal, which was '
                 'two more than Arsenal. They also had six corners to Arsenal’s five, and as you know their extra corner produced '
                 'the winning goal. '
                 'Considering how much possession Arsenal had '
                 'they should have dominated the attacking stats, but if we look at the clearance rate of both teams we can see '
                 'where Arsenal went wrong. The Spurs were able to clear the ball 21 times compared to Arsenal’s 15.')
        st.image('cornerphoto.webp')
        st.markdown('#') 
        st.subheader('Conclusion')
        st.write('So back to the original question, what happened?')
        st.write('In the last couple of years of the 2010s Arsenal weren’t been able to clear the ball from their defensive third '
                 'of the field as effectively as they had in the past. As a result their opposition had more chances getting closer to '
                 'their goal and from corner set-pieces and general attacking play they were able to take more shots.')
        st.write('The old adage of defense winning premierships rings true! If Arsenal want to stop their downfall then perhaps '
                 'they need to focus on this small but very important part of defense. The **humble clearance**!')
        
        
                 
# "The Dataset" Page

elif selected == "The Dataset":
        st.subheader('Description of the Dataset')
        st.write('The dataset was obtained from Kaggle. It consists of all English Premier League matches from '
                 '2010 to halfway through the 2021 season. The author of the dataset collected it using Selenium '
                 'web scraping from the official Premier League website. It has a total of 4070 rows (each '
                 'representing a match) and 113 columns (measuring different variables of the two teams in a match).')
        st.write('The vast majority of columns could be broken into two types: ')
        st.write('1. A variable from a match, eg: the number of passes from Arsenal or the number of tackles '
                 'from their opposition (approximately 25 columns) ')
        st.write('2. A rolling average of a team’s performance throughout the season at the beginning of a '
                 'match, eg: average number of tackles by Arsenal accumulated until the last match '
                 '(approximately 75 columns)')
        st.write('The first five rows from the original dataset: ')
        st.dataframe(df.head())
        

# "Cleaning the Data" Page

elif selected == "Cleaning the Data":
        st.subheader('Creating the Arsenal DataFrame')
        st.write("A quick review of the dataset showed several things needed to be fixed before analysis could start.")
        st.write('The first noticeable problem was that most of the rows didn’t feature Arsenal. The first step was to '
                 'create a new dataFrame of Arsenal games. Using loc, the rows which had ‘Arsenal’ in ‘home_team’ '
                 'or ‘away_team’ columns were taken. ')
               

        creating_arsenal_df = ("""
                               df_arsenal = df.loc[(df['home_team'] == 'Arsenal') | (df['away_team'] == 'Arsenal')] 
                               """)

        st.code(creating_arsenal_df)    
        st.write('The new Arsenal dataFrame looked like this:')
        
        df_arsenal = df.loc[(df['home_team'] == 'Arsenal') | (df['away_team'] == 'Arsenal')] 
        
        st.dataframe(df_arsenal.head())
        st.write('It had 407 rows and 114 columns. ')
        
        st.markdown('#')
        
        st.subheader('Removing unnecessary columns')
        st.write('Next, a lot of the unnecessary columns for the project were deleted.')
        st.write('The columns included: ')
        
        deleted_columns = ('Unnamed: 0', 'link_match','result_full','result_ht','goal_away_ft','sg_match_ft','goal_home_ht',
                           'sg_match_ht','clearances_avg_H','corners_avg_H','fouls_conceded_avg_H','offsides_avg_H','passes_avg_H',
                          'possession_avg_H','red_cards_avg_H','shots_avg_H','shots_on_target_avg_H','tackles_avg_H','touches_avg_H',
                           'yellow_cards_avg_H','goals_scored_ft_avg_H','goals_conced_ft_avg_H','sg_match_ft_acum_H',
                           'goals_scored_ht_avg_H','goals_conced_ht_avg_H','sg_match_ht_acum_H','clearances_avg_A','corners_avg_A',
                           'fouls_conceded_avg_A','offsides_avg_A','passes_avg_A','possession_avg_A','red_cards_avg_A',
                           'shots_avg_A','shots_on_target_avg_A','tackles_avg_A','touches_avg_A','yellow_cards_avg_A',
                           'goals_scored_ft_avg_A','goals_conced_ft_avg_A','sg_match_ft_acum_A','goals_scored_ht_avg_A',
                           'goals_conced_ht_avg_A','sg_match_ht_acum_A','performance_acum_A','clearances_avg_home',
                           'corners_avg_home','fouls_conceded_avg_home','offsides_avg_home','passes_avg_home','possession_avg_home',
                           'red_cards_avg_home','shots_avg_home','shots_on_target_avg_home','tackles_avg_home','touches_avg_home',
                           'yellow_cards_avg_home','goals_scored_ft_avg_home','goals_conced_ft_avg_home','sg_match_ft_acum_home',
                           'goals_scored_ht_avg_home','goals_conced_ht_avg_home','sg_match_ht_acum_home','performance_acum_home',
                           'clearances_avg_away','corners_avg_away','fouls_conceded_avg_away','offsides_avg_away','passes_avg_away',
                           'possession_avg_away','red_cards_avg_away','shots_avg_away','shots_on_target_avg_away','tackles_avg_away',
                           'touches_avg_away','yellow_cards_avg_away','goals_scored_ft_avg_away','goals_conced_ft_avg_away',
                           'sg_match_ft_acum_away','goals_scored_ht_avg_away','goals_conced_ht_avg_away','sg_match_ht_acum_away',
                           'performance_acum_away')       
        
        with st.expander('Deleted Columns'):
            inject_CSS_table(deleted_columns)
            
        st.write('The ‘Unnamed: 0’ and ‘link_match’ were deleted because they provide no useful information. '
                 'The ‘date’ column was deleted because the ‘season’ column was adequate. ')
        st.write('The accumulated average columns (approximately 75 columns) were deleted because they weren’t '
                 'ideal for machine learning models. I didn’t want statistics from previous matches influencing '
                 'the algorithm and the results for each individual match.')  

        st.markdown('#')
            
        st.subheader('Creating Win, Draw, Loss result columns')
        st.write('The next noticeable problem with the dataframe was there weren’t columns for Arsenal’s results of '
                 'each match. The column ‘result_full’ could be used to determine whether the home team or the '
                 'away team won the match, and then using the ‘home_team’, and ‘away_team’ columns it could be '
                 'determined whether Arsenal won, however that was inefficient. There needed to be columns which directly '
                 'showed whether Arsenal won, lost or drew the match. ')
        st.write('To solve this problem the Arsenal dataframe was broken into two different dataframes of Arsenal '
                 'Home Games and Arsenal Away Games. ')
        st.write('The code to find the home games: ')
        
        creating_df_arsenal_home = ("""
                               df_arsenal_home = df_arsenal.loc[(df['home_team'] == 'Arsenal')]
                               """)

        st.code(creating_df_arsenal_home) 
        
        st.write('For the new Arsenal Home Games dataframe a column called GoalDifference was created which '
                 'found the difference between the home team goals and the away team goals for each match. ')       

        creating_goal_difference = ("""
                               df_arsenal_home['GoalDifference'] = df_arsenal_home['HomeGoals'] - df_arsenal_home['AwayGoals']
                               """)
        st.code(creating_goal_difference) 
               

        st.write('Using GoalDifference if the column was positive it meant Arsenal won at home. If it was equal to '
                 'zero it meant they drew, and if it was negative they had lost. To track these results, three new '
                 'columns were created called, ‘Win’, ‘Draw’, ‘Loss’. These columns had binary results of 0 or 1. '
                 'If the result was satisfied a 1 was placed in the column, if it wasn’t then a 0 was placed. ')

        creating_winlossdraw = ("""
                               # Column for home wins
                               df_arsenal_home['Win'] = df_arsenal_home['GoalDifference'] > 0
                               df_arsenal_home['Win'] = df_arsenal_home['Win'].astype(int)
                               
                               # Column for home draws
                               df_arsenal_home['Draw'] = df_arsenal_home['GoalDifference'] == 0
                               df_arsenal_home['Draw'] = df_arsenal_home['Draw'].astype(int)
                               
                               # Column for home losses
                               df_arsenal_home['Loss'] = df_arsenal_home['GoalDifference'] < 0
                               df_arsenal_home['Loss'] = df_arsenal_home['Loss'].astype(int)
                               """)
        st.code(creating_winlossdraw)
        
        st.write('An example of ten rows for the Win, Loss and Draw columns for Arsenal home games:')
        
        st.dataframe(df_arsenal_home2.head(10))
        
        st.write('The same method was used for Arsenal Away Games, however with inverse logic. This meant if '
                 'the GoalDifference was positive then the opposition won their home game and Arsenal had lost their '
                 'away game, etc. ')
        st.write('Afterwards the two dataframes were merged again.')


# "Exploring the Data" Page
        
elif selected == "Exploring the Data":
        st.subheader('Understanding Arsenal’s performances')
        st.write('A quick check revealed there were no duplicate columns or columns with missing values. ')
        st.write('A groupby of the seasons was used to show Arsenal’s performances.' )
            
        creating_seasongroupby = ("""
                               # Finding the number of Wins, Loss, Draws per season by groupby
                               Totals = df_arsenal.groupby(['season'])[['Win', 'Draw', 'Loss']].sum()
                               """)
        
        st.dataframe(Totals)  
        st.write('I then noticed season 20/21 only had 27 games whereas every other season had a total of 38 games. At first I '
                 'thought that the limited games were due to covid restrictions but upon further research the season '
                 'did have 38 games. It turns out the dataset was collated during the season and thus was '
                 'incomplete. The incomplete season of was 20/21 was deleted so I could focus on the decade of the '
                 '2010s.')
        st.markdown('#')
        st.subheader('Creating graphs of performances')
        st.write('The results of each season was graphed in a stacked bar chart.')
        
        Totals.drop(['20/21'],inplace = True)
        stacked_winloss = pd.DataFrame(Totals)
               
        st.code(creating_seasongroupby)
        st.bar_chart(stacked_winloss)
        st.write('As we are aware from the data story there was a noticeable downwards trend of games won by Arsenal throughout the '
                 'decade. Also, in '
                 'each of the last three seasons they suffered more losses than in any other previous years.')
        st.write('I wanted to see if there were a major difference in wins for their home games and away games. '
                 'The ratio for home game wins was found for each season and then graphed.')
        
        creating_home_graphs = ("""
                               # Code of home win record by season
                               home_graph = home_season['Win'] / (home_season['Draw'] + home_season['Loss'] + home_season['Win'])
                               """)
        st.code(creating_home_graphs) 
        
        home_season = df_arsenal_home.groupby(['season'])[['Win', 'Draw', 'Loss']].sum()
        home_season.drop(['20/21'],inplace = True)
        home_graph = home_season['Win'] / (home_season['Draw'] + home_season['Loss'] + home_season['Win'])
        
        st.subheader('Ratio of Home Wins per Season')
        st.line_chart(home_graph)
        st.write('In season 19/20 they won a little over half of their games at home. This wasn’t a great result considering that for '
                 'the majority of the decade they averaged around a 65% win rate at home games for each season. Their best season was '
                 '17/18 where they won almost '
                 '80% of their home games. ')
        st.write('The same process was then done for Arsenal away games.')
        
        away_season = df_arsenal_away.groupby(['season'])[['Win', 'Draw', 'Loss']].sum()
        away_season.drop(['20/21'],inplace = True)
        away_graph = away_season['Win'] / (away_season['Draw'] + away_season['Loss'] + away_season['Win'])
        
        st.subheader('Ratio of Away Wins per Season')
        st.line_chart(away_graph)
        
        st.write('In their best away game season they were winning over half their games away from home. '
                 'In two of the last three seasons of the decade Arsenal had managed to win away from home only 20% of the time. ')
        st.write('The trend was clear for Arsenal. They had downward trajectories of wins during the last seasons '
                 'of the decade.')
        st.write('To discover the reason why, machine learning was used.')
        st.markdown('#')
        st.subheader('Arsenal Compared to Other Champions')     
        st.write('It is necessary to understand not only the performance of Arsenal itself but also the performance of the champions '
                 'in the last years. This analysis can best be represented using a polar chart, which is used to compare several '
                 'variables. This chart is composed of the variables goals scored, goals conceded, passes, points, shots and shots on '
                 'target. The variables were standardized on a scale of 0 to 3 in order to better represent them visually.')
        
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        df_teams=pd.read_csv('champions_arsenal_performance.csv')
        
        # Set Figure

        fig = make_subplots(rows=3, cols=2, specs=[[{'type': 'polar'}]*2]*3)
        
        # Adding the Subplots for Arsena!

        fig.add_trace(go.Scatterpolar(
              name = df_teams['Team'][5],
              r = df_teams.loc[5].drop('Team').values.flatten().tolist(),
              theta = df_teams.columns.tolist()[1:],line=dict(color="orange"),showlegend=False
            ), 1, 1)

        fig.add_trace(go.Scatterpolar(
              name = df_teams['Team'][5],
              r = df_teams.loc[5].drop('Team').values.flatten().tolist(),
              theta = df_teams.columns.tolist()[1:],line=dict(color="orange"),showlegend=False
            ), 1, 2)

        fig.add_trace(go.Scatterpolar(
              name = df_teams['Team'][5],
              r = df_teams.loc[5].drop('Team').values.flatten().tolist(),
              theta = df_teams.columns.tolist()[1:],line=dict(color="orange"),showlegend=False
            ), 2, 1)

        fig.add_trace(go.Scatterpolar(
              name = df_teams['Team'][5],
              r = df_teams.loc[5].drop('Team').values.flatten().tolist(),
              theta = df_teams.columns.tolist()[1:],line=dict(color="orange"),showlegend=False
            ), 2, 2)

        fig.add_trace(go.Scatterpolar(
              name = df_teams['Team'][5],
              r = df_teams.loc[5].drop('Team').values.flatten().tolist(),
              theta = df_teams.columns.tolist()[1:],line=dict(color="orange"),showlegend=True
            ), 3, 1)

        # Adding the Subplots for the champions

        # Manchester United
        fig.add_trace(go.Scatterpolar(
              name = df_teams['Team'][0],
              r = df_teams.loc[0].drop('Team').values.tolist(),
              theta = df_teams.columns.tolist()[1:],line=dict(color="red")
            ), 1, 1)

        # Manchester City
        fig.add_trace(go.Scatterpolar(
              name = df_teams['Team'][1],
              r = df_teams.loc[1].drop('Team').values.tolist(),
              theta = df_teams.columns.tolist()[1:],line=dict(color="deepskyblue")
            ), 1, 2)

        # Chelsea
        fig.add_trace(go.Scatterpolar(
              name = df_teams['Team'][2],
              r = df_teams.loc[2].drop('Team').values.flatten().tolist(),
              theta = df_teams.columns.tolist()[1:],line=dict(color="navy")
            ), 2, 1)

        # Leiscerter City
        fig.add_trace(go.Scatterpolar(
              name = df_teams['Team'][3],
              r = df_teams.loc[3].drop('Team').values.flatten().tolist(),
              theta = df_teams.columns.tolist()[1:],line=dict(color="cyan")
            ), 2, 2)

        # Liverpool
        fig.add_trace(go.Scatterpolar(
              name = df_teams['Team'][4],
              r = df_teams.loc[4].drop('Team').values.flatten().tolist(),
              theta = df_teams.columns.tolist()[1:],line=dict(color="darkred")
            ), 3, 1)

        # Adjusting the layout

        layout = go.Layout(
            autosize=False,
            width=1000,
            height=1000,

            xaxis= go.layout.XAxis(linecolor = 'black',
                                  linewidth = 1,
                                  mirror = True),

            yaxis= go.layout.YAxis(linecolor = 'black',
                                  linewidth = 1,
                                  mirror = True),

            margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=100,
                pad = 4
            )
        )

        fig.update_traces(fill='toself')

        fig.update_layout(
            autosize=False,
            width=1000,
            height=1000,
            yaxis=dict(
                title_text="Y-axis Title",
                ticktext=["Very long label", "long label", "3", "label"],
                tickvals=[1, 2, 3, 4],
                tickmode="array",
                titlefont=dict(size=30),
            )
        )

        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write('Arsenal’s offensive performance is very similar to the performance of the last Premier League champions. The number '
                 'of goals scored, and points scored is very similar between Arsenal, Liverpool, Manchester United and Chelsea.  '
                 'Manchester city is the team that clearly has the best score in each variable.')
        st.write('On the other hand, the defensive aspect is definitive to understand the performance of the teams. Despite having '
                 'similar statistics on the offensive performance, Arsenal concedes more goals than the champions. This translates to '
                 'arsenal’s defensive performance being below that of other teams.')
        


# "Machine Learning" Page

elif selected == "Machine Learning":
        st.subheader('Further Cleaning')
        st.write('To avoid confusion in the outcomes of the following models of whether Arsenal was the home team or the away team '
                 'only the dataset of their home games was used. The results were then tested '
                 'using the dataset for all Arsenal games to see whether the results were consistent.')
        st.write('The Arsenal Home dataset needed further cleaning before machine learning was used. More '
                 'columns were deleted such as the qualitative columns like home_team and away_team. '
                 'The goal columns and the newly created Win, Loss, Draw columns were deleted so the algorithms '
                 'wouldn’t be affected by the outcomes of the match. ')
        st.write('The remaining columns of the dataset were all numerical variables and they measured Arsenal’s '
                 'and the opposition’s performances in home matches. Eg: home_corners, away_offsides, etc.')
        st.write('This dataframe was used as the ‘data’ in train_test_split. For the ‘target’ the ‘Win’ column '
                 'was chosen.') 
                
        creating_datatarget = ("""
                               # Machine Learning Target
                               targethome = df_arsenal_home['Win']
                               """)
        st.code(creating_datatarget)        
        st.write('The Data dataframe was the following: ')
        
        datahome = df_arsenal_home.drop(['season', 'date', 'home_team', 'away_team', 'HomeGoals', 'AwayGoals', 'GoalDifference', 'Win',
                                 'Draw', 'Loss'], axis=1)
        datahome = datahome.iloc[:,0:24]
        
        st.dataframe(datahome.head())     
        
        st.write('The rational for choosing the Win column was that I wanted a column of binary results that '
                 'measured the success of Arsenal. I decided not to make a new column that was the combination of '
                 'wins and draws. The reason was that wins are more much valuable to success than draws are. '
                 'Wins are worth three times as many points than draws. If Arsenal drew every game of their season '
                 'then it would be a very bad result compared to previous seasons and they would be further down '
                 'the ladder. ')   
                
        st.markdown('#')
        st.subheader('Logistic Regression')
        st.write('The Arsenal home dataframe was scaled using StandardScaler.')
        st.write('With sklearn a logistic regression was used on the Target and Data to determine which variables '
                 'were the most correlated to Arsenal wins. ')
        st.image('Machinelearning2.png')   
        st.write('The variable most positive correlated to Arsenal winning at home was home_clearances, ie: '
                 'Arsenal kicking the ball up the field while they were in defence. ')
        st.write('The most negative correlated variable to Arsenal winning was away_clearances, ie: their '
                 'opposition kicking the ball up the field while they were in defence. ')
        st.write('Other noteworthy variables were shots_on_target and away_shots. ')
        st.write('A confusion matrix was used and the logistic regression score was determined to be **0.78**. '
                 'This was in a satisfactory range and consistent with industry standards. ')
        st.image('logistic_report.png', width=430)
        st.markdown('#')
        st.subheader('XGBCLASSIFIER')
        st.write('With sklearn a model of XGBClassifer was used on the Target and Data to determine which '
                 'variables were the most correlated to Arsenal wins. ')
        st.image('Machinelearning1.png')  
        st.write('In the XBG model the most correlated variables to Arsenal victories were away_corners. There '
                 'was also an overlapping result from the logistic regression and it was away_shots. ')
        st.write('A confusion matrix was used and the XGBClassifier score was determined to be **0.76**. '
                 'This was in a satisfactory range and consistent with industry standards. ')
        st.image('xgb_report.png', width=430)
        
        
# "Analysing the Results" Page

elif selected == "Analysing the Results":        
        st.subheader('Analysing the Results')
        st.write('The results of the models were surprising. I did not expect clearances to be such an important variable. '
                 'I needed to have some exploration on the data to understand the results '
                 'better. ')
        st.write('For the following analysis the dataframe of both Arsenal home and away games was used. ')
        st.subheader('Clearances')
        st.write('First the mean clearances per game were tracked for both Arsenal and their opposition throughout '
                 'the seasons in both Arsenal’s home and away games. ')
        
        clearances_home = df_arsenal_home.groupby(['season'])[['home_clearances', 'away_clearances']].mean()
        clearances_home.drop(['20/21'],inplace = True)
        clearances_away = df_arsenal_away.groupby(['season'])[['home_clearances', 'away_clearances']].mean()
        clearances_away.drop(['20/21'],inplace = True)
        clearances_home.rename(columns = {'home_clearances':'Arsenal1','away_clearances':'Opposition1'}, inplace = True)
        clearances_away.rename(columns = {'home_clearances':'Opposition2','away_clearances':'Arsenal2'}, inplace = True)
        clearances = pd.merge(clearances_home, clearances_away, how='outer', on='season')
        clearances['Arsenal'] = (clearances['Arsenal1'] + clearances['Arsenal2']) / 2
        clearances['Opposition'] = (clearances['Opposition1'] + clearances['Opposition2']) / 2
        clearances.drop(['Arsenal1','Arsenal2','Opposition1','Opposition2'],axis=1,inplace=True)
        st.dataframe(clearances) 
        st.write('They were then graphed.')
        st.line_chart(clearances)
        st.write('Looking at the graph we can see clearances for both Arsenal and their opposition had decreased throughout the years. '
                 'Since both sides had decreasing clearances it may suggest there was a general trend for all teams in the league. '
                 'However, the prompt wanted a focus on Arsenal and an explanation of why their team had '
                 'deteriorated over the decade so Arsenal’s diminishing clearances was the variable analysed. ')
        st.write('The models also suggested that away_corners and away_shots were two variables that were '
                 'correlated to wins and losses so they were tracked as well. ')
        st.markdown('#')
        
        st.subheader('Opposition Corners')
        st.write('The average of the opposition corners per game were grouped by season. ')
        
        corners_home = df_arsenal_home.groupby(['season'])[['away_corners', 'home_corners']].mean()
        corners_home.drop(['20/21'],inplace = True)
        corners_away = df_arsenal_away.groupby(['season'])[['home_corners', 'away_corners']].mean()
        corners_away.drop(['20/21'],inplace = True)
        corners = pd.merge(corners_away, corners_home, how='outer', on='season')
        corners['Opposition Corners'] = (corners['home_corners_x'] + corners['away_corners_y']) / 2
        corners.drop(['home_corners_x','away_corners_y'],axis=1,inplace=True)
        corners['Arsenal Corners'] = (corners['home_corners_y'] + corners['away_corners_x']) / 2
        corners.drop(['home_corners_y','away_corners_x'],axis=1,inplace=True)
        corners.drop(['Arsenal Corners'],axis=1,inplace=True)
        st.dataframe(corners) 
        st.write('They were then graphed.')
        st.line_chart(corners)
        st.write('There is a strong upwards trend for Arsenal’s opposition receiving more corners in their matches. ')
        st.markdown('#')
        
        st.subheader('Opposition Shots')
        st.write('Next, the averages of the opposition shots per game were tracked by '
                 'season. ')
        
        shots = df_arsenal_home.groupby(['season'])[['away_shots']].mean()
        shots.drop(['20/21'],inplace = True)
        shots2 = df_arsenal_away.groupby(['season'])[['home_shots']].mean()
        shots2.drop(['20/21'],inplace = True)
        opposition_shots = pd.merge(shots, shots2, how='outer', on='season')
        opposition_shots['Opposition Shots'] = (opposition_shots['away_shots'] + opposition_shots['home_shots']) / 2
        opposition_shots.drop(['away_shots','home_shots'],axis=1,inplace=True)
        st.dataframe(opposition_shots)
        st.write('They were then graphed.')
        st.line_chart(opposition_shots)
        
        st.write('And finally the opposition shots on target were collated and graphed. ')
        shots3 = df_arsenal_home.groupby(['season'])[['away_shots_on_target']].mean()
        shots3.drop(['20/21'],inplace = True)
        shots4 = df_arsenal_away.groupby(['season'])[['home_shots_on_target']].mean()
        shots4.drop(['20/21'],inplace = True)
        opp_shots_on_target = pd.merge(shots3, shots4, how='outer', on='season')
        opp_shots_on_target['Opposition Shots On Target'] = (opp_shots_on_target['away_shots_on_target'] + opp_shots_on_target['home_shots_on_target']) / 2
        opp_shots_on_target.drop(['away_shots_on_target','home_shots_on_target'],axis=1,inplace=True)
        st.dataframe(opp_shots_on_target)
        st.bar_chart(opp_shots_on_target)
        st.write('Again, there is a clear upwards trend for both variables. ')
        
        
# "Understanding the Results" Page

elif selected == "Understanding the Results":        
        st.subheader('Understanding the Results')
        st.write('Putting all of the data together gives a clearer story as to why Arsenal had less success '
                 'throughout the last couple of years of the decade. The models suggests that Arsenal was’t able to '
                 'clear the ball from their half as efficiently as before. ')
        st.write('All the graphs and data demonstrated that throughout the decade Arsenal had less and '
                 'less clearances per game. Interestingly, the dataset also demonstrated that their opposition had '
                 'followed a similar trend of decreasing clearances and this may suggest there was a wider trend '
                 'for all teams in the league. This may have occurred because in the premier league mangers are '
                 'always trying to find an evolution in play that brings an advantage over their opposition and '
                 'hence teams tactics and strategy constantly change. If a team is successful in finding an advantage '
                 'then it is only temporary because inevitably other teams will copy the change and play in a similar '
                 'style. With that said, whether or not the diminishing clearances was a league wide decision or an '
                 'Arsenal decision, one thing is known for sure, the change did not suit Arsenal. As the decade '
                 'progressed they won less and less games per season. ')
        st.image('photoclearance.jpg')
        st.subheader('#')
        st.subheader('Are clearances good or bad?')
        st.write('On its own a team’s diminishing clearances does not tell the entire story. For example, in season '
                 '11/12 Arsenal had a low clearance average per game and they finished 3rd. So there could be several '
                 'positive reasons for a low clearance rate. For example, Aresenal’s tackling efficiency could have '
                 'increased and they were able to steal the ball from their oppositions more often. Or perhaps their '
                 'mid-field defence improved and they were less reliant on clearances. However further testing on the '
                 'dataset demonstrated that in the last few seasons of the decade neither of these reasons were the case. '
                 'There is a strong indication that their diminishing clearances was a negative factor because their '
                 'opposition had significant increases in key variables for their victory. These variables included shots '
                 'on goal and an increase in corners.')
        st.write('So what happened? The data and charts suggests that Arsenal weren’t able to stop their opposition early '
                 'which meant they allowed their opponents more opportunity to get closer to their goal. It is logical to '
                 'suggest that the closer they got, the more shots they took and the more chances they had at scoring goals '
                 'and hence winning or drawing the game. This was obviously detrimental to Arsenal. ')
        st.write('The variables that reinforces this thesis are opposition corners and opposition shots on goal. '
                 'Firstly, an increase in corners says that Arsenal weren’t able to stop their opposition from attacking '
                 'because they were forced to kick the ball out of bounds close to their goal. When this happens a '
                 'corner is produced which is a set shot close to goal. Further research from TheAnalyst says that '
                 '13.6% of goals are scored from corner situations (tracked across five Premier League seasons for '
                 'all teams). The more corners Arsenal gave their opposition the more chances their opponents had '
                 'at scoring goals. Finally, the graphs show their opposition had a dramatic increase in shots and shots '
                 'on target. ')
        st.subheader('Why did it happen?')  
        st.image('wenger.webp')
        st.write('Why did this change in play occur? All of these variables had significant changes in season 18/19 '
                 'and 19/20. Looking at the history of Arsenal there were also major changes occurring within the '
                 'club. In May 2018 their manager of 22 years, Arsène Wenger (pictured above) retired from coaching and he was replaced '
                 'by Unia Emery. Unia only managed Arsenal FC for a year until he was replaced by Freddie Ljungberg '
                 'who also managed the team for a year. Three managers in three years may be reason why there was a '
                 'change in the team’s playing style. Perhaps the new managers wanted to stamp their new style '
                 'onto the team which meant the players weren’t focused on the basics like clearances. Or maybe the '
                 'consistent changes were distractions for the players and they weren’t as motivated as their were in '
                 'the first half of the decade. Whatever the reason, what we now know is that for Arsenal to improve they '
                 'should focus more on defence and clearances. As the old adage goes, “defence wins premierships!” ')
        

# "Power BI" Page

elif selected == "Power BI":  
        st.subheader('Power BI')
        st.write('Accompanying our streamlit is a Power BI file.')
        st.image('powerbi.png')
        st.write('The results of the Power BI analysis has determined that arsenal had a big issues in all of these '
                 'main features, clearances, goal scored, and possession, which means that arsenal need to invest more in better '
                 'players in those positions to be able to compete with Manchester city and Liverpool.')

        
# "Conclusion" Page
elif selected == "Conclusion":
        st.subheader('Conclusion')
        st.image('jersey.jpg')
        st.write('Arsenal FC have been a successful club for a long period of time. However in the last couple of years '
                 'of the 2010s they weren’t as successful. After analysing the data it seems that Arsenal would have '
                 'done better to have focused on their defence more. If they were able to stop their opposition from getting '
                 'closer to their goal by doing more clearances then their opposition would have had less shots and also less '
                 'corners. Obviously the less shots they took, the less chances they had at scoring goals and winning the '
                 'game.')
        st.write('I believe there is sufficient data and graphs for a coherent data story to be told and these surprising '
                 'results that have been found in a facit of the game that not many people consider will allow for interesting '
                 'discussion among Arsenal fans. ')
        
        
        
# "References" Page

elif selected == "References":  
        st.subheader('References')
        st.write('All information about Arsenal FC that was not in the original dataset was taken from the official Premier League '
                 'website and also wikipedia.')
        st.write('The article from TheAnalyst mentioned in the data story is called, ’Premier League Corners: Justifiable Excitement?’ '
                 'It was written by Matt Furniss. ')
           
