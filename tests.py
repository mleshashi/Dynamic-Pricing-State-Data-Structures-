import unittest
from duopoly import p

class TestDuopoly(unittest.TestCase):
    def test_ring_buffer_rollover(self):
        """Test that ring buffers don't exceed maxlen capacity."""
        state = None
        for i in range(100):
            _, state = p({"competitor_price": 10, "demand": i}, state)
        
        # All deques should respect maxlen=50
        self.assertLessEqual(len(state["competitor_prices"]), 50)
        self.assertLessEqual(len(state["my_prices"]), 50)
        self.assertLessEqual(len(state["demands"]), 50)

    def test_online_statistics(self):
        """Test online mean and variance calculations."""
        state = None
        demands = [1, 2, 3, 4, 5]
        
        for demand in demands:
            _, state = p({"competitor_price": 10, "demand": demand}, state)
        
        # Test online mean
        expected_mean = sum(demands) / len(demands)
        self.assertAlmostEqual(state["mean_demand"], expected_mean, places=6)
        
        # Test variance calculation
        expected_variance = sum((d - expected_mean) ** 2 for d in demands) / (len(demands) - 1)
        calculated_variance = state["variance_demand"] / (state["count"] - 1)
        self.assertAlmostEqual(calculated_variance, expected_variance, places=6)

    def test_price_bounds(self):
        """Test that prices stay within reasonable bounds."""
        state = None
        
        # Test various scenarios
        for competitor_price in [0.5, 5.0, 15.0, 25.0]:
            for demand in [0, 5, 10, 20]:
                price, state = p({"competitor_price": competitor_price, "demand": demand}, state)
                
                # Price should be between 1.0 and 20.0
                self.assertGreaterEqual(price, 1.0)
                self.assertLessEqual(price, 20.0)

    def test_cold_start(self):
        """Test behavior with no historical data."""
        state = None
        
        # First call with minimal info
        price, state = p({"competitor_price": 12.0}, state)
        
        # Should return a reasonable price
        self.assertGreaterEqual(price, 1.0)
        self.assertLessEqual(price, 20.0)
        
        # State should be initialized
        self.assertEqual(state["count"], 0)
        self.assertEqual(state["mean_demand"], 0.0)

    def test_trend_detection(self):
        """Test competitor trend signal calculation."""
        state = None
        
        # Create upward trend in competitor prices
        competitor_prices = [8.0, 9.0, 10.0, 11.0, 12.0]
        
        for comp_price in competitor_prices:
            _, state = p({
                "competitor_price": comp_price, 
                "demand": 5,
                "my_last_price": 10.0
            }, state)
        
        # After upward trend, trend_signal should be positive
        self.assertGreater(state["trend_signal"], 0)

    def test_adaptive_base_price(self):
        """Test that base price adapts to performance over time."""
        state = None
        
        # Track base price changes
        base_prices = []
        
        # Simulate scenario where pricing improves over time
        for period in range(10):
            # Gradually improve demand response
            demand = 5 + period
            my_last_price = 10.0 + period * 0.1
            
            _, state = p({
                "competitor_price": 10.0, 
                "demand": demand,
                "my_last_price": my_last_price
            }, state)
            
            base_prices.append(state["base_price"])
        
        # Base price should show adaptation (not necessarily always increasing)
        # Check that it's responsive to market conditions
        self.assertTrue(len(set(base_prices)) > 1, "Base price should adapt over time")

if __name__ == "__main__":
    unittest.main()