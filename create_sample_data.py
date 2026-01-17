

import pandas as pd
import numpy as np
from datetime import datetime, timedelta



start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=x) for x in range(365)]

products = ['Milk', 'Eggs', 'Bread', 'Chicken', 'Rice', 'Tomatoes', 'Apples', 'Cheese']

retail_data = []
for product in products:
    base_price = np.random.uniform(2, 10)  
    trend = np.linspace(0, 2, 365)  
    
    for i, date in enumerate(dates):
        # Add some randomness
        noise = np.random.normal(0, 0.3)
        price = base_price + trend[i] + noise
        quantity = np.random.randint(1, 20)
        
        retail_data.append({
            'InvoiceDate': date,
            'Product': product,
            'UnitPrice': round(price, 2),
            'Quantity': quantity,
            'Country': 'USA'
        })

retail_df = pd.DataFrame(retail_data)
retail_df.to_csv('data/raw/retail_data.csv', index=False)
print(f"✅ Created retail_data.csv with {len(retail_df)} records")

# 2. CREATE REVIEWS/SENTIMENT DATA
print("Creating sentiment data...")

sentiment_templates = {
    'positive': [
        "Great deals at the store today!",
        "Prices are reasonable, happy with my shopping.",
        "Found good discounts on produce.",
        "Grocery shopping was pleasant today."
    ],
    'neutral': [
        "Prices seem normal, nothing special.",
        "Regular shopping trip, nothing changed.",
        "Average prices at the supermarket.",
        "Typical grocery prices this week."
    ],
    'negative': [
        "Everything is so expensive now!",
        "Can't believe how much eggs cost!",
        "Prices keep going up every week.",
        "Had to skip items because too expensive.",
        "Food inflation is getting out of control!"
    ]
}

sentiment_data = []
for i, date in enumerate(dates):
    sentiment_trend = -0.3 * (i / 365)  
    
    if sentiment_trend > -0.1:
        category = 'positive'
        score = np.random.uniform(0.1, 0.5)
    elif sentiment_trend > -0.2:
        category = 'neutral'
        score = np.random.uniform(-0.1, 0.1)
    else:
        category = 'negative'
        score = np.random.uniform(-0.5, -0.1)
    
    
    score += np.random.normal(0, 0.05)
    
    text = np.random.choice(sentiment_templates[category])
    
    sentiment_data.append({
        'date': date,
        'text': text,
        'sentiment_score': round(score, 3),
        'source': 'social_media'
    })

sentiment_df = pd.DataFrame(sentiment_data)
sentiment_df.to_csv('data/raw/sentiment_data.csv', index=False)
print(f"✅ Created sentiment_data.csv with {len(sentiment_df)} records")


print("Creating trends data...")

keywords = ['grocery prices', 'food inflation', 'eggs price', 'milk price']
trends_data = []

for keyword in keywords:
    base_interest = np.random.uniform(20, 40)
    trend_increase = np.linspace(0, 40, 365)  # Interest increases over time
    
    for i, date in enumerate(dates):
        noise = np.random.normal(0, 5)
        interest = base_interest + trend_increase[i] + noise
        interest = max(0, min(100, interest))  # Keep between 0-100
        
        trends_data.append({
            'date': date,
            'keyword': keyword,
            'search_interest': round(interest, 1)
        })

trends_df = pd.DataFrame(trends_data)
trends_df.to_csv('data/raw/trends_data.csv', index=False)
print(f"✅ Created trends_data.csv with {len(trends_df)} records")

print("\n🎉 All sample data created successfully!")
print("\nFiles created:")
print("  - data/raw/retail_data.csv")
print("  - data/raw/sentiment_data.csv")
print("  - data/raw/trends_data.csv")
print("\nYou're ready to start analyzing!")
