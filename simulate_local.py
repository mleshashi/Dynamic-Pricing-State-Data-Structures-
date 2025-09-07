"""
simulate_local.py
A toy simulator for the DPC duopoly pricing function.
Shows state evolution over multiple periods.

Columns:
- Price: Your chosen price for the period
- Competitor: Simulated competitor price
- Demand: Simulated demand
- Base: Adaptive base price
- Trend: Competitor price trend signal
- StateSize: Length of serialized state
"""
import random
import json
import sys
from duopoly import p

def main():
    # Allow custom number of periods via command-line argument
    periods = 20
    if len(sys.argv) > 1:
        try:
            periods = int(sys.argv[1])
        except ValueError:
            print("Invalid argument for number of periods. Using default: 20.")
    print(f"Simulating {periods} periods of pricing...")
    print("Season | Price | Competitor | Demand | Base | Trend | StateSize")
    state_dump = None
    for season in range(1, periods + 1):
        feedback = {
            'competitor_price': 8.0 + random.uniform(-2, 4),
            'demand': max(1, int(10 + random.gauss(0, 2))),
            'my_last_price': 9.0 + random.uniform(-1, 2)
        }
        price, state_dump = p(
            current_selling_season=season,
            information_dump_last_round=state_dump,
            **feedback
        )
        try:
            state = json.loads(state_dump)
            print(f"{season:6d} | {price:5.2f} | {feedback['competitor_price']:10.2f} | {feedback['demand']:6d} | "
                  f"{state['base_price']:5.2f} | {state['competitor_trend']:6.2f} | {len(state_dump):9d}")
        except (json.JSONDecodeError, KeyError):
            print(f"{season:6d} | {price:5.2f} | {feedback['competitor_price']:10.2f} | {feedback['demand']:6d} | ERROR in state serialization.")
    print("Simulation complete.")

if __name__ == "__main__":
    main()
