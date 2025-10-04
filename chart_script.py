import plotly.graph_objects as go
import pandas as pd

# Data for strategy comparison
data = {
    "strategy": ["RL Agent", "Fixed Price", "Competitive"],
    "revenue": [456492, 440900, 396830],
    "profit": [182597, 176360, 158732]
}

df = pd.DataFrame(data)

# Convert to thousands for better readability
df['revenue_k'] = df['revenue'] / 1000
df['profit_k'] = df['profit'] / 1000

# Create grouped bar chart
fig = go.Figure()

# Add Revenue bars
fig.add_trace(go.Bar(
    name='Revenue',
    x=df['strategy'],
    y=df['revenue_k'],
    marker_color='#1FB8CD',
    text=[f'${x:.1f}k' for x in df['revenue_k']],
    textposition='outside',
    offsetgroup=1
))

# Add Profit bars
fig.add_trace(go.Bar(
    name='Profit',
    x=df['strategy'],
    y=df['profit_k'],
    marker_color='#DB4545',
    text=[f'${x:.1f}k' for x in df['profit_k']],
    textposition='outside',
    offsetgroup=2
))

# Update layout
fig.update_layout(
    title='Strategy Performance Comparison',
    xaxis_title='Strategy',
    yaxis_title='Amount ($k)',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces to prevent clipping
fig.update_traces(cliponaxis=False)

# Save as both PNG and SVG
fig.write_image("strategy_performance.png")
fig.write_image("strategy_performance.svg", format="svg")

fig.show()