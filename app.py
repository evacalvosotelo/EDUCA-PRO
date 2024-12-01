import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load and preprocess data
file_path = 'StudentPerformanceFactors.csv'
data = pd.read_csv(file_path)
df = data.dropna()
df = df.rename(columns={'Exam_Score': 'Grade'})

df['Parental_Involvement'] = df['Parental_Involvement'].replace(
    {'Low': 1, 'Medium': 2, 'High': 3})
df['Motivation_Level'] = df['Motivation_Level'].replace(
    {'Low': 1, 'Medium': 2, 'High': 3})
df['Family_Income'] = df['Family_Income'].replace(
    {'Low': 1, 'Medium': 2, 'High': 3})
df['Teacher_Quality'] = df['Teacher_Quality'].replace(
    {'Low': 1, 'Medium': 2, 'High': 3})
df['Internet_Access'] = df['Internet_Access'].replace({'Yes': 1, 'No': 0})
df['Extracurricular_Activities'] = df['Extracurricular_Activities'].replace({
                                                                            'Yes': 1, 'No': 0})

# LINEAR REGRESSION MODEL
X = df[['Hours_Studied', 'Attendance', 'Sleep_Hours',
        'Motivation_Level', 'Parental_Involvement', 'Teacher_Quality']]
y = df['Grade']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# PERFORMANCE
y_test_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

predicted_vs_actual_fig = px.scatter(
    x=y_test,
    y=y_test_pred,
    labels={"x": "Actual Grades", "y": "Predicted Grades"},
    title="Actual vs Predicted Grades",
)
predicted_vs_actual_fig.add_trace(
    go.Scatter(x=y_test, y=y_test, mode="lines", name="Perfect Prediction")
)

# DASHBOARD
numeric_columns = df.select_dtypes(include=['number']).columns
correlation_matrix = df[numeric_columns].corr()

scatter_fig = px.scatter(df, x='Hours_Studied', y='Grade', labels={
                         'Hours_Studied': 'Hours of study per week'})

bar_parental_fig = px.bar(
    df.groupby('Parental_Involvement')['Grade'].mean().reset_index(),
    x='Parental_Involvement',
    y='Grade',
    text='Grade'
)
bar_parental_fig.update_traces(textposition='outside')
bar_parental_fig.update_layout(
    xaxis=dict(tickmode='array', tickvals=[
               1, 2, 3], ticktext=['Low', 'Medium', 'High']),
    xaxis_title="Parental Involvement"
)
### EXTRA ACTS ###
pie_extracurricular_fig = px.pie(
    df, names='Extracurricular_Activities', title='Participation in extracurricular activities')

# Box plot: Grade distribution based on participation in Extracurricular Activities
box_extracurricular_fig = px.box(
    df,
    x='Extracurricular_Activities',
    y='Grade',
    title='Grade distribution by extracurricular activity participation',
    labels={'Extracurricular_Activities': 'Participates in extracurricular activities', 'Grade': 'Grade'}
)


box_sleep_fig = px.box(df, x='Sleep_Hours', y='Grade', labels={
                       'Sleep_Hours': 'Hours of sleep per week'})

heatmap_fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    colorscale='Viridis'))
heatmap_fig.update_layout(title='Correlation Heatmap')

# APP
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Predict your grade',
                style={
                    'font-family': 'Arial, sans-serif',
                    'font-size': '18px',
                    'color': 'black',
                    'text-align': 'center'
                },
                selected_style={
                    'font-family': 'Arial, sans-serif',
                    'font-size': '18px',
                    'color': 'black',
                    'font-weight': 'bold',
                    'background-color': '#dcdcdc',
                    'font-weight': 'bold',
                    'text-align': 'center',
                },
                children=[
                    html.Div([
                        html.Div(
                            style={
                                'display': 'flex',
                                'justify-content': 'space-between',
                                'align-items': 'center',
                                'width': '100%',
                                'padding': '10px'
                            },
                            children=[
                                html.Img(src='assets/undraw_studying_re_deca.svg',
                                         style={
                                             'width': '200px',
                                             'margin': '0 auto',
                                         }),

                                html.Div([
                                    html.H3("Predict your grade", style={
                                        'text-align': 'center',
                                        'font-family': 'Arial, sans-serif',
                                        'font-size': '24px',
                                        'color': '#2C3E50',
                                        'margin-bottom': '10px'
                                    }),
                                    html.Span("Enter details and click the button to predict", style={
                                        'color': 'black',
                                        'font-style': 'italic',
                                        'font-weight': 'thin',
                                        'display': 'block',
                                        'text-align': 'center',
                                        'margin-bottom': '20px'
                                    }),

                                    html.Label("How many hours do you study per week?", style={
                                        'font-size': '16px', 'color': '#34495E'}),
                                    dcc.Input(id='input-hours-studied', type='number', min=0, placeholder="Hours studied",
                                              style={
                                                  'width': '250px',
                                                  'padding': '8px',
                                                  'border': '1px solid #BDC3C7',
                                                  'border-radius': '5px',
                                                  'display': 'block',
                                                  'margin-bottom': '10px',
                                                  'margin-top': '10px',
                                                  'margin-left': 'auto',
                                                  'margin-right': 'auto',
                                              }),

                                    html.Label("What is your attendance percentage?", style={
                                        'font-size': '16px', 'color': '#34495E'}),
                                    dcc.Input(id='input-attendance', type='number', min=0, max=100, placeholder="Attendance (%)",
                                              style={
                                                  'width': '250px',
                                                  'padding': '8px',
                                                  'border': '1px solid #BDC3C7',
                                                  'border-radius': '5px',
                                                  'display': 'block',
                                                  'margin-bottom': '10px',
                                                  'margin-top': '10px',
                                                  'margin-left': 'auto',
                                                  'margin-right': 'auto',
                                              }),

                                    html.Label("How many hours do you sleep per night on average?", style={
                                        'font-size': '16px', 'color': '#34495E'}),
                                    dcc.Input(id='input-sleep-hours', type='number', min=0, max=24, placeholder="Hours slept",
                                              style={
                                                  'width': '250px',
                                                  'padding': '8px',
                                                  'border': '1px solid #BDC3C7',
                                                  'border-radius': '5px',
                                                  'display': 'block',
                                                  'margin-bottom': '10px',
                                                  'margin-top': '10px',
                                                  'margin-left': 'auto',
                                                  'margin-right': 'auto',
                                              }),

                                    html.Label("How would you rate your motivation level?",
                                               style={
                                                   'font-size': '16px', 'color': '#34495E'}),
                                    dcc.Dropdown(
                                        id='input-motivation-level',
                                        options=[
                                            {'label': 'Low', 'value': 1},
                                            {'label': 'Medium', 'value': 2},
                                            {'label': 'High', 'value': 3}
                                        ],
                                        placeholder="Select your motivation level",
                                        style={'margin-bottom': '10px',
                                               'margin-top': '10px',
                                               'margin-left': 'auto',
                                               'margin-right': 'auto',
                                               'width': '250px'}
                                    ),

                                    html.Label("How involved are your parents in your education?", style={
                                        'font-size': '16px', 'color': '#34495E'}),
                                    dcc.Dropdown(
                                        id='input-parental-involvement',
                                        options=[
                                            {'label': 'Low', 'value': 1},
                                            {'label': 'Medium', 'value': 2},
                                            {'label': 'High', 'value': 3}
                                        ],
                                        placeholder="Select your parents involvement",
                                        style={'margin-bottom': '10px',
                                               'margin-top': '10px',
                                               'margin-left': 'auto',
                                               'margin-right': 'auto',
                                               'width': '250px'}
                                    ),

                                    html.Label("How would you rate your teacher's quality?", style={
                                        'font-size': '16px', 'color': '#34495E'}),
                                    dcc.Dropdown(
                                        id='input-teacher-quality',
                                        options=[
                                            {'label': 'Low', 'value': 1},
                                            {'label': 'Medium', 'value': 2},
                                            {'label': 'High', 'value': 3}
                                        ],
                                        placeholder="Select the quality of your teacher",
                                        style={'margin-bottom': '10px',
                                               'margin-top': '10px',
                                               'margin-left': 'auto',
                                               'margin-right': 'auto',
                                               'width': '250px'}),

                                    html.Button('Predict your grade out of 100', id='predict-button', n_clicks=0,
                                                style={
                                                    'background-color': '#2980B9',
                                                    'color': 'white',
                                                    'font-size': '14px',
                                                    'padding': '8px 12px',
                                                    'border': 'none',
                                                    'border-radius': '5px',
                                                    'cursor': 'pointer',
                                                    'display': 'block',
                                                    'margin': '10px auto'
                                                }),
                                    html.Div(id='output-prediction', style={
                                        'margin-top': '20px', 'font-size': '16px', 'font-weight': 'bold', 'color': '#27AE60', 'text-align': 'center'
                                    })
                                ], style={
                                    'background-color': '#ECF0F1',
                                    'padding-left': '20px',
                                    'border-radius': '10px',
                                    'width': '40%',
                                    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
                                }),

                                html.Img(src='assets/undraw_exams_re_4ios.svg',
                                         style={
                                             'width': '200px',
                                             'margin': '0 auto',
                                         }),
                            ]
                        )
                    ])
                ]),


        dcc.Tab(label='Dashboard and insights',
                style={
                    'font-family': 'Arial, sans-serif',
                    'font-size': '18px',
                    'color': 'black',
                    'text-align': 'center'
                },
                selected_style={
                    'font-family': 'Arial, sans-serif',
                    'font-size': '18px',
                    'color': 'black',
                    'font-weight': 'bold',
                    'background-color': '#dcdcdc',
                    'font-weight': 'bold',
                    'text-align': 'center',
                },
                children=[
                    html.Div([
                        html.H2("Hours studied vs. Grade"),
                        dcc.Graph(figure=scatter_fig),
                        html.P("There is a positive trend, suggesting that higher study hours are associated with higher grades.",
                               style={'textAlign': 'center', 'font-size': '18px', 'color': '#333', 'margin-top': '10px'})
                    ], style={'margin-bottom': '30px'}),

                    html.Div([
                        html.H2("Parental involvement vs. Grade"),
                        dcc.Graph(figure=bar_parental_fig),
                        html.P("Students with higher parental involvement tend to have slightly higher average grades, though the effect is modest.",
                               style={'textAlign': 'center', 'font-size': '18px', 'color': '#333', 'margin-top': '10px'})
                    ], style={'margin-bottom': '30px'}),

                    html.Div([
                        html.H2("Participation in extracurricular activities"),
                        html.Div([
                            dcc.Graph(figure=pie_extracurricular_fig,
                                      style={'width': '48%'}),
                            dcc.Graph(figure=box_extracurricular_fig,
                                      style={'width': '48%'})
                        ], style={'display': 'flex', 'justify-content': 'space-between'}),
                        html.P("Students participating in extracurricular activities have similar grade distributions compared to those who do not.",
                               style={'textAlign': 'center', 'font-size': '18px', 'color': '#333', 'margin-top': '10px'})
                    ], style={'margin-bottom': '30px'}),

                    html.Div([
                        html.H2("Sleep hours vs. Grade"),
                        dcc.Graph(figure=box_sleep_fig),
                        html.P("Most students have grades around the same range regardless of sleep hours, with few significant outliers.",
                               style={'textAlign': 'center', 'font-size': '18px', 'color': '#333', 'margin-top': '10px'})
                    ], style={'margin-bottom': '30px'}),

                    html.Div([
                        html.H2("Correlation heatmap"),
                        dcc.Graph(figure=heatmap_fig),
                        html.P("This heatmap shows that hours studied and attendance have the highest correlations with 'grade', though they are not very strong.",
                               style={'textAlign': 'center', 'font-size': '18px', 'color': '#333', 'margin-top': '10px'})
                    ], style={'margin-bottom': '30px'}),
                ]),

        dcc.Tab(label='Build your graph!',
                style={
                    'font-family': 'Arial, sans-serif',
                    'font-size': '18px',
                    'color': 'black',
                    'text-align': 'center'
                },
                selected_style={
                    'font-family': 'Arial, sans-serif',
                    'font-size': '18px',
                    'color': 'black',
                    'font-weight': 'bold',
                    'background-color': '#dcdcdc',
                    'font-weight': 'bold',
                    'text-align': 'center',
                },
                children=[
                    html.Div([
                        html.Label('Choose X-axis:',
                                   style={'font-size': '16px', 'color': '#34495E'}),
                        dcc.Dropdown(
                            id='x-axis-dropdown',
                            options=[{'label': col, 'value': col}
                             for col in X.columns],
                            placeholder='Select X-axis',
                            style={'width': '50%', 'margin-bottom': '20px'}
                        ),
                        html.Div([
                            html.Label(['*The Y-axis is fixed: Grade'],
                                       style={'font-style': 'italic', 'color': '#34495E'})
                        ], style={'padding-top': '10px'}),
                        dcc.Graph(id='custom-graph')
                    ], style={'padding': '20px'})
                ]),

        dcc.Tab(label='Precision of the model',
                style={
                    'font-family': 'Arial, sans-serif',
                    'font-size': '18px',
                    'color': 'black',
                    'text-align': 'center'
                },
                selected_style={
                    'font-family': 'Arial, sans-serif',
                    'font-size': '18px',
                    'color': 'black',
                    'font-weight': 'bold',
                    'background-color': '#dcdcdc',
                    'font-weight': 'bold',
                    'text-align': 'center',
                },
                children=[
                    html.Div([
                        html.H2("How precise is the model we use?", style={
                            "text-align": "center", 'margin-bottom': '20px'}),
                        html.Div([
                            html.P(f"Mean Squared Error (MSE): {mse:.2f}", style={
                                'font-size': '16px', 'color': '#34495E'}),
                            html.P(f"Root Mean Squared Error (RMSE): {rmse:.2f}", style={
                                'font-size': '16px', 'color': '#34495E'}),
                            html.P(f"Mean Absolute Error (MAE): {mae:.2f}", style={
                                'font-size': '16px', 'color': '#34495E'}),
                            html.P(f"R² Score: {r2:.2f}", style={
                                'font-size': '16px', 'color': '#34495E'})
                        ], style={
                            "font-size": "18px",
                            "line-height": "1.6",
                            "margin-bottom": "20px",
                            "text-align": "center"
                        }),
                        dcc.Graph(figure=predicted_vs_actual_fig)
                    ], style={'padding': '20px'})
                ])
    ])
])

# Callback to update the custom graph


@ app.callback(
    Output('custom-graph', 'figure'),
    [Input('x-axis-dropdown', 'value')]
)
def update_custom_graph(x_axis):
    if x_axis:
        return px.scatter(df, x=x_axis, y='Grade', title=f'{x_axis} vs Grade',
                          labels={'x': x_axis, 'y': 'Grade'})
    return go.Figure()

# Callback to make predictions


@ app.callback(
    Output('output-prediction', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        State('input-hours-studied', 'value'),
        State('input-attendance', 'value'),
        State('input-sleep-hours', 'value'),
        State('input-motivation-level', 'value'),
        State('input-parental-involvement', 'value'),
        State('input-teacher-quality', 'value')
    ]
)
def predict_grade(n_clicks, hours_studied, attendance, sleep_hours, motivation_level, parental_involvement, teacher_quality):
    if n_clicks > 0:
        if None in (hours_studied, attendance, sleep_hours, motivation_level, parental_involvement, teacher_quality):
            return html.Span("Please fill in all fields.",
                             style={'color': 'red',
                                    'font-weight': 'bold',
                                    })

        features = np.array([[hours_studied, attendance, sleep_hours,
                            motivation_level, parental_involvement, teacher_quality]])

        grade_prediction = model.predict(features)[0]

        return f"Predicted Grade: {grade_prediction:.2f}"

    return ""


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
