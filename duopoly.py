import numpy as np
from collections import deque
import json

def p(current_selling_season, information_dump_last_round=None, **feedback_data):
    """
    DPC-compliant pricing function for Duopoly competition.
    
    Args:
        current_selling_season (int): Current season/period number
        information_dump_last_round (str): Serialized state from previous round
        **feedback_data: Market feedback (competitor_price, demand, etc.)
    
    Returns:
        tuple: (price (float), information_dump_response (str))
    """
    
    # 1. DESERIALIZE STATE
    if information_dump_last_round is None or information_dump_last_round == "":
        # Cold start - initialize state
        state = {
            "season": 0,
            "competitor_prices": [],      # Ring buffer as list (bounded)
            "my_prices": [],             # Ring buffer as list (bounded) 
            "demands": [],               # Ring buffer as list (bounded)
            "profit_history": [],        # Ring buffer as list (bounded)
            
            # Online statistics (O(1) updates)
            "demand_mean": 0.0,
            "demand_m2": 0.0,           # For Welford's variance
            "demand_count": 0,
            
            "competitor_mean": 0.0,
            "competitor_m2": 0.0,
            "competitor_count": 0,
            
            # Adaptive parameters
            "base_price": 10.0,
            "profit_momentum": 0.0,      # Exponential moving average of profit changes
            "price_volatility": 1.0,     # Adaptive volatility measure
            
            # Trend tracking
            "competitor_trend": 0.0,     # Short-term trend signal
            "demand_trend": 0.0,         # Short-term demand trend
        }
    else:
        try:
            state = json.loads(information_dump_last_round)
        except (json.JSONDecodeError, TypeError):
            # Fallback to cold start on corrupted state
            state = {
                "season": 0,
                "competitor_prices": [], "my_prices": [], "demands": [], "profit_history": [],
                "demand_mean": 0.0, "demand_m2": 0.0, "demand_count": 0,
                "competitor_mean": 0.0, "competitor_m2": 0.0, "competitor_count": 0,
                "base_price": 10.0, "profit_momentum": 0.0, "price_volatility": 1.0,
                "competitor_trend": 0.0, "demand_trend": 0.0
            }
    
    # 2. EXTRACT MARKET FEEDBACK
    competitor_price = feedback_data.get('competitor_price', None)
    demand = feedback_data.get('demand', None) 
    my_last_price = feedback_data.get('my_last_price', state["base_price"])
    
    # 3. UPDATE STATE WITH BOUNDED RING BUFFERS (MAX 30 elements for memory efficiency)
    MAX_HISTORY = 30
    
    if competitor_price is not None:
        state["competitor_prices"].append(float(competitor_price))
        if len(state["competitor_prices"]) > MAX_HISTORY:
            state["competitor_prices"].pop(0)  # Ring buffer behavior
            
        # Online mean update for competitor prices (Welford's algorithm)
        state["competitor_count"] += 1
        n = state["competitor_count"]
        delta = competitor_price - state["competitor_mean"]
        state["competitor_mean"] += delta / n
        delta2 = competitor_price - state["competitor_mean"] 
        state["competitor_m2"] += delta * delta2
    
    if demand is not None and my_last_price is not None:
        state["demands"].append(float(demand))
        state["my_prices"].append(float(my_last_price))
        
        # Maintain ring buffers
        if len(state["demands"]) > MAX_HISTORY:
            state["demands"].pop(0)
        if len(state["my_prices"]) > MAX_HISTORY:
            state["my_prices"].pop(0)
            
        # Online mean and variance for demand (Welford's algorithm)
        state["demand_count"] += 1
        n = state["demand_count"]
        delta = demand - state["demand_mean"]
        state["demand_mean"] += delta / n
        delta2 = demand - state["demand_mean"]
        state["demand_m2"] += delta * delta2
        
        # Calculate and store profit
        cost = 5.0  # Assumed unit cost
        profit = (my_last_price - cost) * demand
        state["profit_history"].append(profit)
        if len(state["profit_history"]) > MAX_HISTORY:
            state["profit_history"].pop(0)
        
        # Update profit momentum (exponential moving average of changes)
        if len(state["profit_history"]) >= 2:
            profit_change = state["profit_history"][-1] - state["profit_history"][-2]
            alpha = 0.1  # Smoothing factor
            state["profit_momentum"] = alpha * profit_change + (1 - alpha) * state["profit_momentum"]
    
    # 4. CALCULATE TREND SIGNALS (using recent history)
    if len(state["competitor_prices"]) >= 3:
        recent = state["competitor_prices"][-3:]
        if recent[-1] > recent[0]:
            state["competitor_trend"] = min(1.0, state["competitor_trend"] + 0.2)
        elif recent[-1] < recent[0]: 
            state["competitor_trend"] = max(-1.0, state["competitor_trend"] - 0.2)
        else:
            state["competitor_trend"] *= 0.8  # Decay
            
    if len(state["demands"]) >= 3:
        recent = state["demands"][-3:]
        if recent[-1] > recent[0]:
            state["demand_trend"] = min(1.0, state["demand_trend"] + 0.2)
        elif recent[-1] < recent[0]:
            state["demand_trend"] = max(-1.0, state["demand_trend"] - 0.2) 
        else:
            state["demand_trend"] *= 0.8
    
    # 5. ADAPTIVE BASE PRICE UPDATE
    if state["demand_count"] > 0:
        # Adjust base price based on profit momentum and demand trends
        base_adjustment = 0.02 * state["profit_momentum"] + 0.01 * state["demand_trend"]
        state["base_price"] += base_adjustment
        state["base_price"] = max(6.0, min(15.0, state["base_price"]))  # Keep reasonable bounds
    
    # 6. CALCULATE PRICING DECISION
    price = state["base_price"]
    
    # Competitor-responsive pricing
    if competitor_price is not None and competitor_price > 0:
        competitor_gap = competitor_price - price
        price += 0.3 * competitor_gap  # Follow competitor partially
        
        # Add trend-following component
        price += 0.5 * state["competitor_trend"]
    
    # Demand-responsive adjustment
    if state["demand_count"] > 5:  # Need some history
        demand_variance = state["demand_m2"] / (state["demand_count"] - 1) if state["demand_count"] > 1 else 0
        demand_std = np.sqrt(max(0, demand_variance))
        
        # Price higher when demand is strong and stable
        if state["demand_mean"] > 6.0:
            price += 0.3
        elif state["demand_mean"] < 3.0:
            price -= 0.2
            
        # Reduce price volatility in uncertain conditions
        if demand_std > 3.0:
            price = 0.7 * price + 0.3 * state["base_price"]  # Move toward conservative base
    
    # 7. PRICE BOUNDS & EDGE CASE HANDLING
    price = max(1.0, min(20.0, price))  # Hard bounds for competition safety
    
    # Handle edge cases
    if np.isnan(price) or np.isinf(price):
        price = state["base_price"]
    
    # Update season counter
    state["season"] = current_selling_season
    
    # 8. SERIALIZE STATE FOR NEXT ROUND
    try:
        information_dump_response = json.dumps(state, separators=(',', ':'))  # Compact JSON
    except (TypeError, ValueError):
        # Fallback: minimal state to prevent complete failure
        minimal_state = {"season": current_selling_season, "base_price": float(price)}
        information_dump_response = json.dumps(minimal_state)
    
    return float(price), information_dump_response


# Helper function for local testing
def simulate_market_feedback():
    """Generate synthetic market data for testing."""
    import random
    return {
        'competitor_price': 8.0 + random.uniform(-2, 4),
        'demand': max(1, int(10 + random.gauss(0, 2))),
        'my_last_price': 9.0 + random.uniform(-1, 2)
    }


# Test function
def test_pricing_function():
    """Test the pricing function with synthetic data."""
    print("Testing DPC-compliant pricing function...")
    
    state_dump = None
    
    for season in range(1, 11):
        feedback = simulate_market_feedback()
        
        price, state_dump = p(
            current_selling_season=season,
            information_dump_last_round=state_dump,
            **feedback
        )
        
        print(f"Season {season}: Price={price:.2f}, Competitor={feedback['competitor_price']:.2f}, "
              f"Demand={feedback['demand']}, State_size={len(state_dump)} chars")
    
    print("Test completed successfully!")


if __name__ == "__main__":
    test_pricing_function()