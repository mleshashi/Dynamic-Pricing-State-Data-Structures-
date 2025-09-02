import unittest
import json
import numpy as np
from duopoly import p

class TestDPCDuopoly(unittest.TestCase):
    
    def test_cold_start_handling(self):
        """Test function behavior with no previous state (cold start)."""
        price, state_dump = p(
            current_selling_season=1,
            information_dump_last_round=None,
            competitor_price=10.0,
            demand=5
        )
        
        # Should return valid price and state
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)
        self.assertLess(price, 25)
        self.assertIsInstance(state_dump, str)
        
        # State should be valid JSON
        state = json.loads(state_dump)
        self.assertEqual(state['season'], 1)
        self.assertIn('base_price', state)

    def test_ring_buffer_bounds(self):
        """Test that ring buffers maintain bounded size."""
        state_dump = None
        MAX_PERIODS = 50  # Test more periods than MAX_HISTORY (30)
        
        for season in range(1, MAX_PERIODS + 1):
            price, state_dump = p(
                current_selling_season=season,
                information_dump_last_round=state_dump,
                competitor_price=10.0 + season * 0.1,
                demand=5 + (season % 3),
                my_last_price=9.0 + season * 0.05
            )
            
            # Check that state is valid
            state = json.loads(state_dump)
            
            # Ring buffers should not exceed MAX_HISTORY (30)
            self.assertLessEqual(len(state['competitor_prices']), 30)
            self.assertLessEqual(len(state['demands']), 30)
            self.assertLessEqual(len(state['my_prices']), 30)
            self.assertLessEqual(len(state['profit_history']), 30)

    def test_online_statistics_welford(self):
        """Test online mean and variance calculations (Welford's algorithm)."""
        demands = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
        competitor_prices = [8, 9, 10, 11, 12, 9, 10, 11, 10, 9]
        
        state_dump = None
        
        for i, (demand, comp_price) in enumerate(zip(demands, competitor_prices)):
            price, state_dump = p(
                current_selling_season=i + 1,
                information_dump_last_round=state_dump,
                competitor_price=comp_price,
                demand=demand,
                my_last_price=10.0
            )
        
        state = json.loads(state_dump)
        
        # Test demand statistics
        expected_demand_mean = np.mean(demands)
        self.assertAlmostEqual(state['demand_mean'], expected_demand_mean, places=3)
        
        if state['demand_count'] > 1:
            expected_demand_var = np.var(demands, ddof=1)
            calculated_demand_var = state['demand_m2'] / (state['demand_count'] - 1)
            self.assertAlmostEqual(calculated_demand_var, expected_demand_var, places=3)
        
        # Test competitor price statistics  
        expected_comp_mean = np.mean(competitor_prices)
        self.assertAlmostEqual(state['competitor_mean'], expected_comp_mean, places=3)

    def test_price_bounds_edge_cases(self):
        """Test price bounds and edge case handling."""
        test_cases = [
            # (competitor_price, demand, expected_bounds)
            (0.1, 1, (1.0, 20.0)),      # Very low competitor price
            (50.0, 20, (1.0, 20.0)),    # Very high competitor price  
            (10.0, 0, (1.0, 20.0)),     # Zero demand
            (10.0, 100, (1.0, 20.0)),   # Very high demand
            (None, None, (1.0, 20.0)),  # Missing data
        ]
        
        for comp_price, demand, (min_bound, max_bound) in test_cases:
            kwargs = {}
            if comp_price is not None:
                kwargs['competitor_price'] = comp_price
            if demand is not None:
                kwargs['demand'] = demand
                kwargs['my_last_price'] = 10.0
            
            price, _ = p(
                current_selling_season=1,
                information_dump_last_round=None,
                **kwargs
            )
            
            self.assertGreaterEqual(price, min_bound, 
                                  f"Price {price} below bound with comp={comp_price}, demand={demand}")
            self.assertLessEqual(price, max_bound,
                               f"Price {price} above bound with comp={comp_price}, demand={demand}")
            self.assertFalse(np.isnan(price) or np.isinf(price))

    def test_state_serialization_robustness(self):
        """Test state serialization handles corrupted data gracefully."""
        # Test with corrupted JSON state
        corrupted_states = [
            '{"incomplete": json',     # Invalid JSON
            '{"season": "not_a_number"}',  # Wrong data type
            '',                        # Empty string
            'not json at all',         # Not JSON
            None                       # None value
        ]
        
        for corrupted_state in corrupted_states:
            price, new_state = p(
                current_selling_season=5,
                information_dump_last_round=corrupted_state,
                competitor_price=10.0,
                demand=5
            )
            
            # Should recover gracefully
            self.assertIsInstance(price, float)
            self.assertGreater(price, 0)
            self.assertIsInstance(new_state, str)
            
            # New state should be valid JSON
            state_dict = json.loads(new_state)
            self.assertIn('season', state_dict)

    def test_trend_detection_mechanism(self):
        """Test competitor and demand trend detection."""
        state_dump = None
        
        # Create upward trend in competitor prices with more pronounced changes
        upward_trend_prices = [6.0, 8.0, 10.0, 12.0, 14.0]
        
        for i, comp_price in enumerate(upward_trend_prices):
            price, state_dump = p(
                current_selling_season=i + 1,
                information_dump_last_round=state_dump,
                competitor_price=comp_price,
                demand=5,
                my_last_price=10.0
            )
        
        state = json.loads(state_dump)
        # After upward competitor trend, trend signal should be positive
        self.assertGreater(state['competitor_trend'], 0.1,
                          f"Expected significantly positive trend, got {state['competitor_trend']}")
        
        # Now test downward trend with more pronounced changes
        downward_trend_prices = [12.0, 10.0, 8.0, 6.0, 4.0]
        
        for i, comp_price in enumerate(downward_trend_prices):
            price, state_dump = p(
                current_selling_season=i + 6,
                information_dump_last_round=state_dump,
                competitor_price=comp_price,
                demand=5,
                my_last_price=10.0
            )
        
        state = json.loads(state_dump)
        # After downward trend, trend signal should be negative
        self.assertLess(state['competitor_trend'], -0.1, 
                       f"Expected significantly negative trend, got {state['competitor_trend']}")

    def test_performance_constraints(self):
        """Test that function meets performance requirements."""
        import time
        
        state_dump = None
        times = []
        
        # Test 100 calls to check average performance
        for season in range(1, 101):
            start_time = time.time()
            
            price, state_dump = p(
                current_selling_season=season,
                information_dump_last_round=state_dump,
                competitor_price=10.0 + (season % 10),
                demand=5 + (season % 7),
                my_last_price=9.0 + (season % 5)
            )
            
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Check performance constraints from DPC requirements
        self.assertLess(avg_time, 0.2, f"Average time {avg_time:.4f}s exceeds 0.2s limit")
        self.assertLess(max_time, 5.0, f"Max time {max_time:.4f}s exceeds 5s limit")
        
        print(f"Performance: Avg={avg_time:.4f}s, Max={max_time:.4f}s")

    def test_memory_efficiency(self):
        """Test memory usage stays bounded."""
        state_dump = None
        state_sizes = []
        
        # Run for many periods to test memory bounds
        for season in range(1, 200):  
            price, state_dump = p(
                current_selling_season=season,
                information_dump_last_round=state_dump,
                competitor_price=10.0,
                demand=5,
                my_last_price=10.0
            )
            
            state_sizes.append(len(state_dump))
        
        # State size should stabilize and not grow indefinitely
        recent_avg = sum(state_sizes[-20:]) / 20  # Last 20 periods
        early_avg = sum(state_sizes[50:70]) / 20   # Earlier periods
        
        # Memory should not grow significantly over time due to ring buffers
        growth_ratio = recent_avg / early_avg if early_avg > 0 else 1
        self.assertLess(growth_ratio, 1.5, 
                       f"State size growing too fast: {growth_ratio:.2f}x growth")
        
        print(f"Memory: Early avg={early_avg:.0f} chars, Recent avg={recent_avg:.0f} chars")


if __name__ == "__main__":
    unittest.main(verbosity=2)