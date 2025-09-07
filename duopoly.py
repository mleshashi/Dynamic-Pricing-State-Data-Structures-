"""
duopoly.py
Pricing bot for the Duopoly track of the Dynamic Pricing Competition (DPC).
Implements a DPC-compliant pricing function with lightweight state management.
"""

import numpy as np
from collections import deque
import json

# Pricing function called once per selling period
# Returns (price, information_dump_response)
def p(current_selling_season, information_dump_last_round=None, **feedback_data):
    """
    DPC-compliant pricing function for Duopoly competition.
    - Maintains bounded history using ring buffers (max 30 elements)
    - Updates online statistics (mean, variance) in O(1) time
    - Handles cold starts and corrupted state gracefully
    - Keeps price within sensible bounds for safety
    """
    # Helper to create default state for cold start or recovery
    def create_default_state():
        return {
            "season": 0,
            "competitor_prices": [],      # Last 30 competitor prices
            "my_prices": [],              # Last 30 own prices
            "demands": [],                # Last 30 demand values
            "profit_history": [],         # Last 30 profit values
            # Online stats for demand and competitor prices
            "demand_mean": 0.0,
            "demand_m2": 0.0,
            "demand_count": 0,
            "competitor_mean": 0.0,
            "competitor_m2": 0.0,
            "competitor_count": 0,
            # Adaptive pricing parameters
            "base_price": 10.0,           # Initial base price
            "profit_momentum": 0.0,       # EMA of profit changes
            "price_volatility": 1.0,      # Not used, placeholder
            # Trend signals
            "competitor_trend": 0.0,      # Short-term competitor trend
            "demand_trend": 0.0,          # Short-term demand trend
        }

    # 1. Deserialize state or cold start
    if information_dump_last_round is None or information_dump_last_round == "":
        state = create_default_state()
    else:
        try:
            state = json.loads(information_dump_last_round)
            # Fill missing keys if any
            default_state = create_default_state()
            for key in default_state:
                if key not in state:
                    state[key] = default_state[key]
        except (json.JSONDecodeError, TypeError):
            state = create_default_state()

    # 2. Extract market feedback
    competitor_price = feedback_data.get('competitor_price', None)
    demand = feedback_data.get('demand', None)
    my_last_price = feedback_data.get('my_last_price', state["base_price"])

    # 3. Update state with bounded ring buffers (max 30 elements)
    MAX_HISTORY = 30

    if competitor_price is not None:
        state["competitor_prices"].append(float(competitor_price))
        if len(state["competitor_prices"]) > MAX_HISTORY:
            state["competitor_prices"].pop(0)
        # Online mean/variance update (Welford's algorithm)
        state["competitor_count"] += 1
        n = state["competitor_count"]
        delta = competitor_price - state["competitor_mean"]
        state["competitor_mean"] += delta / n
        delta2 = competitor_price - state["competitor_mean"]
        state["competitor_m2"] += delta * delta2

    if demand is not None and my_last_price is not None:
        state["demands"].append(float(demand))
        state["my_prices"].append(float(my_last_price))
        if len(state["demands"]) > MAX_HISTORY:
            state["demands"].pop(0)
        if len(state["my_prices"]) > MAX_HISTORY:
            state["my_prices"].pop(0)
        # Online mean/variance for demand
        state["demand_count"] += 1
        n = state["demand_count"]
        delta = demand - state["demand_mean"]
        state["demand_mean"] += delta / n
        delta2 = demand - state["demand_mean"]
        state["demand_m2"] += delta * delta2
        # Calculate profit (simple margin)
        cost = 5.0  # Unit cost
        profit = (my_last_price - cost) * demand
        state["profit_history"].append(profit)
        if len(state["profit_history"]) > MAX_HISTORY:
            state["profit_history"].pop(0)
        # Exponential moving average for profit momentum
        if len(state["profit_history"]) >= 2:
            profit_change = state["profit_history"][-1] - state["profit_history"][-2]
            alpha = 0.1  # Smoothing factor (low = slow, high = fast)
            state["profit_momentum"] = alpha * profit_change + (1 - alpha) * state["profit_momentum"]

    # 4. Calculate trend signals (last 3 values)
    if len(state["competitor_prices"]) >= 3:
        recent = state["competitor_prices"][-3:]
        if recent[-1] > recent[0]:
            state["competitor_trend"] = min(1.0, state["competitor_trend"] + 0.2)
        elif recent[-1] < recent[0]:
            state["competitor_trend"] = max(-1.0, state["competitor_trend"] - 0.2)
        else:
            state["competitor_trend"] *= 0.8  # Decay toward zero
    if len(state["demands"]) >= 3:
        recent = state["demands"][-3:]
        if recent[-1] > recent[0]:
            state["demand_trend"] = min(1.0, state["demand_trend"] + 0.2)
        elif recent[-1] < recent[0]:
            state["demand_trend"] = max(-1.0, state["demand_trend"] - 0.2)
        else:
            state["demand_trend"] *= 0.8

    # 5. Adaptive base price update
    if state["demand_count"] > 0:
        # Base price nudged by profit momentum and demand trend
        base_adjustment = 0.02 * state["profit_momentum"] + 0.01 * state["demand_trend"]
        state["base_price"] += base_adjustment
        # Keep base price in reasonable bounds
        state["base_price"] = max(6.0, min(15.0, state["base_price"]))

    # 6. Calculate pricing decision
    price = state["base_price"]
    # Respond to competitor price
    if competitor_price is not None and competitor_price > 0:
        competitor_gap = competitor_price - price
        price += 0.3 * competitor_gap  # Follow competitor partially
        price += 0.5 * state["competitor_trend"]  # Add trend-following
    # Respond to demand
    if state["demand_count"] > 5:
        demand_variance = state["demand_m2"] / (state["demand_count"] - 1) if state["demand_count"] > 1 else 0
        demand_std = np.sqrt(max(0, demand_variance))
        if state["demand_mean"] > 6.0:
            price += 0.3  # Demand is strong
        elif state["demand_mean"] < 3.0:
            price -= 0.2  # Demand is weak
        # If demand is volatile, be more conservative
        if demand_std > 3.0:
            price = 0.7 * price + 0.3 * state["base_price"]
    # 7. Price bounds and edge case handling
    price = max(1.0, min(20.0, price))  # Hard bounds for safety
    if np.isnan(price) or np.isinf(price):
        price = state["base_price"]
    # Update season
    state["season"] = current_selling_season
    # 8. Serialize state for next round
    try:
        information_dump_response = json.dumps(state, separators=(',', ':'))
    except (TypeError, ValueError):
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