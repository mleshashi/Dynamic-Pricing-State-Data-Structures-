import numpy as np
from collections import deque
import math

def p(info, state=None):
    """
    Pricing function called every period.
    info: dict with competitor price, demand feedback, etc.
    state: dict holding compact state across calls
    """
    if state is None:
        state = {
            "competitor_prices": deque(maxlen=50),   # ring buffer for competitor prices
            "my_prices": deque(maxlen=50),           # ring buffer for my prices
            "demands": deque(maxlen=50),             # ring buffer for demands
            "mean_demand": 0.0,                      # online mean demand
            "variance_demand": 0.0,                  # online variance for demand
            "count": 0,                              # total observations
            "base_price": 10.0,                      # adaptive base price
            "last_profit": 0.0,                      # profit tracking
            "trend_signal": 0.0                      # competitor trend signal
        }

    # Extract inputs
    competitor_price = info.get("competitor_price", state["base_price"])
    demand = info.get("demand", None)
    my_last_price = info.get("my_last_price", state["base_price"])

    # Update state with new data
    if demand is not None:
        state["competitor_prices"].append(competitor_price)
        state["my_prices"].append(my_last_price)
        state["demands"].append(demand)
        state["count"] += 1
        
        # Online mean and variance update (Welford's algorithm)
        n = state["count"]
        delta = demand - state["mean_demand"]
        state["mean_demand"] += delta / n
        delta2 = demand - state["mean_demand"]
        state["variance_demand"] += delta * delta2
        
        # Update base price based on performance
        profit = (my_last_price - 5.0) * demand  # Assume cost = 5.0
        if n > 1:
            profit_change = profit - state["last_profit"]
            if profit_change > 0:
                state["base_price"] += 0.05  # Increase base if profit improved
            else:
                state["base_price"] -= 0.02  # Decrease base if profit declined
        state["last_profit"] = profit

    # Calculate competitor trend
    if len(state["competitor_prices"]) >= 3:
        recent_prices = list(state["competitor_prices"])[-3:]
        if recent_prices[-1] > recent_prices[0]:
            state["trend_signal"] = 0.3  # Upward trend
        elif recent_prices[-1] < recent_prices[0]:
            state["trend_signal"] = -0.3  # Downward trend
        else:
            state["trend_signal"] *= 0.8  # Decay trend signal

    # Enhanced pricing strategy
    demand_std = math.sqrt(state["variance_demand"] / max(1, state["count"] - 1))
    demand_signal = 1.0 if state["mean_demand"] > 5.0 else -1.0
    
    # Adaptive pricing based on multiple factors
    price_adjustment = (
        demand_signal * 0.4 +                    # Demand response
        state["trend_signal"] +                   # Competitor trend following
        min(0.2, demand_std / 10.0) * demand_signal  # Volatility adjustment
    )
    
    my_price = state["base_price"] + price_adjustment
    
    # Add small competitive offset
    if competitor_price > 0:
        competitive_gap = competitor_price - my_price
        my_price += 0.2 * competitive_gap  # Close gap partially
    
    # Dynamic bounds based on market conditions
    min_price = max(1.0, state["base_price"] * 0.7)
    max_price = min(20.0, state["base_price"] * 1.5)
    my_price = max(min_price, min(max_price, my_price))

    return my_price, state


# Test function for development
def test_pricing():
    """Simple test to verify pricing behavior."""
    state = None
    print("Testing pricing function:")
    
    for period in range(10):
        # Simulate market data
        competitor_price = 10.0 + period * 0.2
        demand = max(1, 8 - abs(period - 5))
        my_last_price = 10.5 if period > 0 else 10.0
        
        info = {
            "competitor_price": competitor_price,
            "demand": demand,
            "my_last_price": my_last_price
        }
        
        price, state = p(info, state)
        print(f"Period {period}: My Price={price:.2f}, Competitor={competitor_price:.2f}, Demand={demand}")


if __name__ == "__main__":
    test_pricing()