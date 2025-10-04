// Application data
const appData = {
  "business_metrics": {
    "total_products": 10500,
    "revenue_uplift_potential": "15-25%",
    "profit_improvement": "18-30%",
    "current_total_revenue": 540048.62,
    "optimization_coverage": "5 categories, 4 store types"
  },
  "model_performance": [
    {"model": "Linear Regression", "r2_score": 0.7565, "rmse": 24.39},
    {"model": "Ridge Regression", "r2_score": 0.7566, "rmse": 24.38},
    {"model": "Random Forest", "r2_score": 0.8158, "rmse": 21.21},
    {"model": "Gradient Boosting", "r2_score": 0.8192, "rmse": 21.01}
  ],
  "feature_importance": [
    {"feature": "Sales Moving Average", "importance": 63.18},
    {"feature": "Inventory Level", "importance": 12.23},
    {"feature": "Price Ratio", "importance": 5.16},
    {"feature": "Sales Lag", "importance": 3.55},
    {"feature": "Price Difference", "importance": 2.30},
    {"feature": "Seasonal Factor", "importance": 1.81},
    {"feature": "Day of Year", "importance": 1.80},
    {"feature": "Weather Impact", "importance": 1.59},
    {"feature": "Current Price", "importance": 1.36},
    {"feature": "Product Type", "importance": 1.12}
  ],
  "strategy_results": [
    {"strategy": "RL Agent", "revenue": 456492, "profit": 182597, "avg_price": 60.56, "avg_sales": 150.7},
    {"strategy": "Fixed Price", "revenue": 440900, "profit": 176360, "avg_price": 100.00, "avg_sales": 84.8},
    {"strategy": "Competitive", "revenue": 396830, "profit": 158732, "avg_price": 198.06, "avg_sales": 38.8}
  ],
  "price_elasticity": [
    {"category": "Electronics", "elasticity": -1.2, "calculated": 14.03},
    {"category": "Clothing", "elasticity": -1.8, "calculated": 16.5},
    {"category": "Home & Garden", "elasticity": -0.8, "calculated": 9.2},
    {"category": "Sports", "elasticity": -1.5, "calculated": 12.8},
    {"category": "Books", "elasticity": -0.6, "calculated": 7.1}
  ],
  "rl_training": [
    {"episode": 0, "reward": 312087, "revenue": 912891},
    {"episode": 50, "reward": 308000, "revenue": 920000},
    {"episode": 100, "reward": 306544, "revenue": 921124},
    {"episode": 150, "reward": 307200, "revenue": 921000},
    {"episode": 200, "reward": 306826, "revenue": 921432},
    {"episode": 250, "reward": 306500, "revenue": 921300},
    {"episode": 299, "reward": 306800, "revenue": 921410}
  ]
};

// Chart instances
let charts = {};

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
  initializeTabs();
  // Initialize charts after a small delay to ensure DOM is ready
  setTimeout(() => {
    initializeCharts();
    initializeOptimizer();
    initializeSensitivityAnalysis();
  }, 100);
});

// Tab Management - Fixed version
function initializeTabs() {
  const navTabs = document.querySelectorAll('.nav-tab');
  const tabContents = document.querySelectorAll('.tab-content');
  
  console.log('Found tabs:', navTabs.length);
  console.log('Found contents:', tabContents.length);

  navTabs.forEach(tab => {
    tab.addEventListener('click', (e) => {
      e.preventDefault();
      const targetTab = tab.getAttribute('data-tab');
      console.log('Switching to tab:', targetTab);
      
      // Remove active class from all tabs and contents
      navTabs.forEach(t => t.classList.remove('active'));
      tabContents.forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none';
      });
      
      // Add active class to clicked tab and corresponding content
      tab.classList.add('active');
      const targetContent = document.getElementById(targetTab);
      if (targetContent) {
        targetContent.classList.add('active');
        targetContent.style.display = 'block';
        console.log('Activated tab:', targetTab);
      } else {
        console.error('Target content not found:', targetTab);
      }
      
      // Resize charts when switching tabs
      setTimeout(() => {
        Object.values(charts).forEach(chart => {
          if (chart && typeof chart.resize === 'function') {
            chart.resize();
          }
        });
      }, 100);
    });
  });
  
  // Ensure only overview is visible initially
  tabContents.forEach((content, index) => {
    if (index === 0) {
      content.style.display = 'block';
    } else {
      content.style.display = 'none';
    }
  });
}

// Chart Initialization
function initializeCharts() {
  // Only create charts if their containers exist
  const modelChart = document.getElementById('modelPerformanceChart');
  if (modelChart) createModelPerformanceChart();
  
  const featureChart = document.getElementById('featureImportanceChart');
  if (featureChart) createFeatureImportanceChart();
  
  const elasticityChart = document.getElementById('elasticityChart');
  if (elasticityChart) createElasticityChart();
  
  const strategyChart = document.getElementById('strategyChart');
  if (strategyChart) createStrategyChart();
  
  const rlChart = document.getElementById('rlTrainingChart');
  if (rlChart) createRLTrainingChart();
  
  const sensitivityChart = document.getElementById('sensitivityChart');
  if (sensitivityChart) createSensitivityChart();
}

// Model Performance Chart
function createModelPerformanceChart() {
  const ctx = document.getElementById('modelPerformanceChart');
  if (!ctx) return;
  
  charts.modelPerformance = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: {
      labels: appData.model_performance.map(d => d.model),
      datasets: [{
        label: 'R² Score',
        data: appData.model_performance.map(d => d.r2_score),
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5'],
        borderColor: '#32b8cd',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#eee' }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          ticks: { color: '#aaa' },
          grid: { color: '#2a3f5f' }
        },
        x: {
          ticks: { color: '#aaa' },
          grid: { color: '#2a3f5f' }
        }
      }
    }
  });
}

// Feature Importance Chart
function createFeatureImportanceChart() {
  const ctx = document.getElementById('featureImportanceChart');
  if (!ctx) return;
  
  charts.featureImportance = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: {
      labels: appData.feature_importance.slice(0, 8).map(d => d.feature),
      datasets: [{
        label: 'Importance (%)',
        data: appData.feature_importance.slice(0, 8).map(d => d.importance),
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C', '#964325'],
        borderColor: '#32b8cd',
        borderWidth: 1
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#eee' }
        }
      },
      scales: {
        x: {
          ticks: { color: '#aaa' },
          grid: { color: '#2a3f5f' }
        },
        y: {
          ticks: { color: '#aaa' },
          grid: { color: '#2a3f5f' }
        }
      }
    }
  });
}

// Price Elasticity Chart
function createElasticityChart() {
  const ctx = document.getElementById('elasticityChart');
  if (!ctx) return;
  
  charts.elasticity = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: {
      labels: appData.price_elasticity.map(d => d.category),
      datasets: [
        {
          label: 'Theoretical Elasticity',
          data: appData.price_elasticity.map(d => Math.abs(d.elasticity)),
          backgroundColor: '#1FB8CD',
        },
        {
          label: 'Calculated Elasticity',
          data: appData.price_elasticity.map(d => d.calculated),
          backgroundColor: '#FFC185',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#eee' }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { color: '#aaa' },
          grid: { color: '#2a3f5f' }
        },
        x: {
          ticks: { color: '#aaa' },
          grid: { color: '#2a3f5f' }
        }
      }
    }
  });
}

// Strategy Comparison Chart
function createStrategyChart() {
  const ctx = document.getElementById('strategyChart');
  if (!ctx) return;
  
  charts.strategy = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: {
      labels: appData.strategy_results.map(d => d.strategy),
      datasets: [
        {
          label: 'Revenue ($K)',
          data: appData.strategy_results.map(d => d.revenue / 1000),
          backgroundColor: '#1FB8CD',
        },
        {
          label: 'Profit ($K)',
          data: appData.strategy_results.map(d => d.profit / 1000),
          backgroundColor: '#FFC185',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#eee' }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { 
            color: '#aaa',
            callback: function(value) {
              return '$' + value + 'K';
            }
          },
          grid: { color: '#2a3f5f' }
        },
        x: {
          ticks: { color: '#aaa' },
          grid: { color: '#2a3f5f' }
        }
      }
    }
  });
}

// RL Training Chart
function createRLTrainingChart() {
  const ctx = document.getElementById('rlTrainingChart');
  if (!ctx) return;
  
  charts.rlTraining = new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: {
      labels: appData.rl_training.map(d => d.episode),
      datasets: [{
        label: 'Training Reward',
        data: appData.rl_training.map(d => d.reward),
        borderColor: '#32b8cd',
        backgroundColor: 'rgba(50, 184, 205, 0.1)',
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#eee' }
        }
      },
      scales: {
        y: {
          ticks: { 
            color: '#aaa',
            callback: function(value) {
              return '$' + (value / 1000).toFixed(0) + 'K';
            }
          },
          grid: { color: '#2a3f5f' }
        },
        x: {
          ticks: { color: '#aaa' },
          grid: { color: '#2a3f5f' }
        }
      }
    }
  });
}

// Sensitivity Analysis Chart
function createSensitivityChart() {
  const ctx = document.getElementById('sensitivityChart');
  if (!ctx) return;
  
  // Generate sample sensitivity data
  const pricePoints = [];
  const profitData = {};
  const multipliers = [0.8, 0.9, 1.0, 1.1, 1.2];
  
  for (let price = 50; price <= 200; price += 10) {
    pricePoints.push(price);
  }
  
  multipliers.forEach(mult => {
    profitData[mult] = pricePoints.map(price => {
      // Simulate profit curve with optimal around $85-$95
      const optimalPrice = 90;
      const maxProfit = 12000;
      const distance = Math.abs(price - optimalPrice);
      const profit = Math.max(0, maxProfit - (distance * distance * mult * 2));
      return profit;
    });
  });
  
  const datasets = multipliers.map((mult, index) => ({
    label: `Competitor ${Math.round(mult * 100)}%`,
    data: profitData[mult],
    borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#5D878F', '#DB4545'][index],
    backgroundColor: `rgba(${index === 0 ? '31, 184, 205' : index === 1 ? '255, 193, 133' : index === 2 ? '180, 65, 60' : index === 3 ? '93, 135, 143' : '219, 69, 69'}, 0.1)`,
    fill: false,
    tension: 0.4
  }));
  
  charts.sensitivity = new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: {
      labels: pricePoints,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#eee' }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { 
            color: '#aaa',
            callback: function(value) {
              return '$' + (value / 1000).toFixed(1) + 'K';
            }
          },
          grid: { color: '#2a3f5f' }
        },
        x: {
          ticks: { 
            color: '#aaa',
            callback: function(value) {
              return '$' + value;
            }
          },
          grid: { color: '#2a3f5f' }
        }
      }
    }
  });
}

// Price Optimizer
function initializeOptimizer() {
  const form = document.getElementById('optimizerForm');
  const inventorySlider = document.getElementById('inventoryLevel');
  const seasonalSlider = document.getElementById('seasonalFactor');
  const inventoryValue = document.getElementById('inventoryValue');
  const seasonalValue = document.getElementById('seasonalValue');
  
  if (!form) return; // Exit if form doesn't exist
  
  // Update slider values
  if (inventorySlider && inventoryValue) {
    inventorySlider.addEventListener('input', () => {
      inventoryValue.textContent = inventorySlider.value;
    });
  }
  
  if (seasonalSlider && seasonalValue) {
    seasonalSlider.addEventListener('input', () => {
      seasonalValue.textContent = seasonalSlider.value;
    });
  }
  
  // Form submission
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    optimizePrice();
  });
}

function optimizePrice() {
  const category = document.getElementById('category')?.value;
  const currentPrice = parseFloat(document.getElementById('currentPrice')?.value || 0);
  const competitorPrice = parseFloat(document.getElementById('competitorPrice')?.value || 0);
  const inventoryLevel = parseFloat(document.getElementById('inventoryLevel')?.value || 50);
  const seasonalFactor = parseFloat(document.getElementById('seasonalFactor')?.value || 1);
  const marketingSpend = parseFloat(document.getElementById('marketingSpend')?.value || 1000);
  
  if (!category || !currentPrice || !competitorPrice) return;
  
  // Get category elasticity
  const categoryData = appData.price_elasticity.find(d => d.category === category);
  const elasticity = categoryData ? categoryData.elasticity : -1.2;
  
  // Optimization algorithm (simplified)
  const baseDemand = 100;
  const costRatio = 0.6; // Assume 60% cost ratio
  
  // Calculate optimal price considering various factors
  let optimalPrice = competitorPrice * 0.95; // Start slightly below competitor
  
  // Adjust for inventory (higher inventory = lower price)
  const inventoryFactor = 1 - (inventoryLevel - 50) * 0.002;
  optimalPrice *= inventoryFactor;
  
  // Adjust for seasonal factor
  optimalPrice *= seasonalFactor;
  
  // Adjust for marketing spend (higher marketing = higher sustainable price)
  const marketingFactor = 1 + (marketingSpend / 10000) * 0.05;
  optimalPrice *= marketingFactor;
  
  // Calculate demand using price elasticity
  const priceChange = (optimalPrice - currentPrice) / currentPrice;
  const demandChange = elasticity * priceChange;
  const predictedDemand = baseDemand * (1 + demandChange);
  
  // Calculate revenue and profit
  const expectedRevenue = optimalPrice * predictedDemand;
  const expectedProfit = (optimalPrice - (optimalPrice * costRatio)) * predictedDemand;
  const priceChangePercent = ((optimalPrice - currentPrice) / currentPrice) * 100;
  
  // Update results
  const elements = {
    recommendedPrice: document.getElementById('recommendedPrice'),
    predictedDemand: document.getElementById('predictedDemand'),
    expectedRevenue: document.getElementById('expectedRevenue'),
    expectedProfit: document.getElementById('expectedProfit'),
    priceChange: document.getElementById('priceChange')
  };
  
  if (elements.recommendedPrice) elements.recommendedPrice.textContent = `$${optimalPrice.toFixed(2)}`;
  if (elements.predictedDemand) elements.predictedDemand.textContent = `${predictedDemand.toFixed(0)} units`;
  if (elements.expectedRevenue) elements.expectedRevenue.textContent = `$${expectedRevenue.toFixed(0)}`;
  if (elements.expectedProfit) elements.expectedProfit.textContent = `$${expectedProfit.toFixed(0)}`;
  if (elements.priceChange) elements.priceChange.textContent = `${priceChangePercent >= 0 ? '+' : ''}${priceChangePercent.toFixed(1)}%`;
  
  // Add animation
  const resultsContainer = document.getElementById('optimizationResults');
  if (resultsContainer) {
    resultsContainer.classList.add('slide-up');
  }
}

// Sensitivity Analysis
function initializeSensitivityAnalysis() {
  const priceRangeMin = document.getElementById('priceRangeMin');
  const priceRangeMax = document.getElementById('priceRangeMax');
  const multiplierRadios = document.querySelectorAll('input[name="multiplier"]');
  
  // Range slider updates
  if (priceRangeMin) priceRangeMin.addEventListener('input', updatePriceRange);
  if (priceRangeMax) priceRangeMax.addEventListener('input', updatePriceRange);
  
  // Multiplier radio updates
  multiplierRadios.forEach(radio => {
    radio.addEventListener('change', updateSensitivityAnalysis);
  });
  
  // Initialize
  updatePriceRange();
  updateSensitivityAnalysis();
}

function updatePriceRange() {
  const minInput = document.getElementById('priceRangeMin');
  const maxInput = document.getElementById('priceRangeMax');
  const minSpan = document.getElementById('priceRangeMin');
  const maxSpan = document.getElementById('priceRangeMax');
  
  if (!minInput || !maxInput) return;
  
  const minValue = minInput.value;
  const maxValue = maxInput.value;
  
  // Update display values
  const minDisplay = document.querySelector('span[id="priceRangeMin"]');
  const maxDisplay = document.querySelector('span[id="priceRangeMax"]');
  if (minDisplay) minDisplay.textContent = minValue;
  if (maxDisplay) maxDisplay.textContent = maxValue;
  
  // Ensure min <= max
  if (parseInt(minValue) > parseInt(maxValue)) {
    if (event && event.target.id === 'priceRangeMin') {
      maxInput.value = minValue;
    } else {
      minInput.value = maxValue;
    }
  }
}

function updateSensitivityAnalysis() {
  const selectedRadio = document.querySelector('input[name="multiplier"]:checked');
  if (!selectedRadio) return;
  
  const multiplier = parseFloat(selectedRadio.value);
  
  // Update scenario analysis based on selected multiplier
  const optimalPrice = 85 + (multiplier - 1) * 15; // Adjust optimal price
  const maxProfit = 12500 - Math.abs(multiplier - 1) * 2000; // Adjust max profit
  const elasticity = -1.2 - (multiplier - 1) * 0.3; // Adjust elasticity
  
  let marketPosition = 'Competitive';
  if (multiplier < 0.9) marketPosition = 'Premium';
  else if (multiplier > 1.1) marketPosition = 'Discount';
  
  const elements = {
    optimalPrice: document.getElementById('optimalPrice'),
    maxProfit: document.getElementById('maxProfit'),
    priceElasticity: document.getElementById('priceElasticity'),
    marketPosition: document.getElementById('marketPosition')
  };
  
  if (elements.optimalPrice) elements.optimalPrice.textContent = `$${optimalPrice.toFixed(0)}`;
  if (elements.maxProfit) elements.maxProfit.textContent = `$${maxProfit.toLocaleString()}`;
  if (elements.priceElasticity) elements.priceElasticity.textContent = elasticity.toFixed(1);
  if (elements.marketPosition) elements.marketPosition.textContent = marketPosition;
}

// Utility Functions
function formatCurrency(value) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }).format(value);
}

function formatPercentage(value) {
  return `${(value * 100).toFixed(1)}%`;
}

// Window resize handler for charts
window.addEventListener('resize', () => {
  Object.values(charts).forEach(chart => {
    if (chart && typeof chart.resize === 'function') {
      chart.resize();
    }
  });
});

// Export functions for external use if needed
window.appFunctions = {
  optimizePrice,
  updateSensitivityAnalysis,
  formatCurrency,
  formatPercentage
};