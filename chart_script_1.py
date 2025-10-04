import pandas as pd
import plotly.graph_objects as go
import json

# Load the data
data_json = {
  "sensitivity_data": [
    {"price": 50, "competitor_80": 800, "competitor_90": 1000, "competitor_100": 1200, "competitor_110": 1400, "competitor_120": 1600},
    {"price": 60, "competitor_80": 1200, "competitor_90": 1500, "competitor_100": 1800, "competitor_110": 2100, "competitor_120": 2400},
    {"price": 70, "competitor_80": 1500, "competitor_90": 1900, "competitor_100": 2300, "competitor_110": 2700, "competitor_120": 3100},
    {"price": 80, "competitor_80": 1700, "competitor_90": 2200, "competitor_100": 2700, "competitor_110": 3200, "competitor_120": 3700},
    {"price": 90, "competitor_80": 1800, "competitor_90": 2400, "competitor_100": 3000, "competitor_110": 3600, "competitor_120": 4200},
    {"price": 100, "competitor_80": 1850, "competitor_90": 2500, "competitor_100": 3200, "competitor_110": 3900, "competitor_120": 4600},
    {"price": 110, "competitor_80": 1800, "competitor_90": 2500, "competitor_100": 3300, "competitor_110": 4100, "competitor_120": 4900},
    {"price": 120, "competitor_80": 1700, "competitor_90": 2400, "competitor_100": 3200, "competitor_110": 4200, "competitor_120": 5100},
    {"price": 130, "competitor_80": 1550, "competitor_90": 2250, "competitor_100": 3100, "competitor_110": 4200, "competitor_120": 5200},
    {"price": 140, "competitor_80": 1350, "competitor_90": 2050, "competitor_100": 2950, "competitor_110": 4100, "competitor_120": 5200},
    {"price": 150, "competitor_80": 1100, "competitor_90": 1800, "competitor_100": 2750, "competitor_110": 3950, "competitor_120": 5100},
    {"price": 160, "competitor_80": 800, "competitor_90": 1500, "competitor_100": 2500, "competitor_110": 3700, "competitor_120": 4900},
    {"price": 170, "competitor_80": 450, "competitor_90": 1150, "competitor_100": 2200, "competitor_110": 3400, "competitor_120": 4600},
    {"price": 180, "competitor_80": 50, "competitor_90": 750, "competitor_100": 1850, "competitor_110": 3000, "competitor_120": 4200},
    {"price": 190, "competitor_80": -400, "competitor_90": 300, "competitor_100": 1450, "competitor_110": 2550, "competitor_120": 3700},
    {"price": 200, "competitor_80": -900, "competitor_90": -200, "competitor_100": 1000, "competitor_110": 2050, "competitor_120": 3100}
  ]
}

# Convert to DataFrame
df = pd.DataFrame(data_json["sensitivity_data"])

# Create the figure
fig = go.Figure()

# Brand colors in order
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C']

# Add traces for each competitor scenario
scenarios = [
    ('competitor_80', '80% Comp Price', colors[0]),
    ('competitor_90', '90% Comp Price', colors[1]),
    ('competitor_100', '100% Comp Price', colors[2]),
    ('competitor_110', '110% Comp Price', colors[3]),
    ('competitor_120', '120% Comp Price', colors[4])
]

for scenario, name, color in scenarios:
    fig.add_trace(go.Scatter(
        x=df['price'],
        y=df[scenario],
        mode='lines+markers',
        name=name,
        line=dict(color=color, width=3),
        marker=dict(size=6),
        hovertemplate='Price: $%{x}<br>Profit: $%{y:,.0f}<extra></extra>'
    ))

# Update layout
fig.update_layout(
    title='Price Sensitivity Analysis',
    xaxis_title='Price ($)',
    yaxis_title='Profit ($)',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    )
)

# Update axes
fig.update_xaxes(tickformat='$,.0f')
fig.update_yaxes(tickformat='$,.0f')

# Save as PNG and SVG
fig.write_image('chart.png')
fig.write_image('chart.svg', format='svg')

print("Chart created successfully!")