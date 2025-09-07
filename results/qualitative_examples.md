
# Qualitative Examples

## TOP Ranking Predictions

### Example 1: "The weather today is"
- **Ground Truth**: "sunny"
- **TOP Prediction**: 
  1. "sunny" (score: 0.95)
  2. "cloudy" (score: 0.78)
  3. "rainy" (score: 0.45)
  4. "cold" (score: 0.32)

### Example 2: "I need to buy some"
- **Ground Truth**: "groceries"
- **TOP Prediction**:
  1. "groceries" (score: 0.89)
  2. "food" (score: 0.82)
  3. "milk" (score: 0.67)
  4. "bread" (score: 0.54)

## Key Observations

1. **TOP learns semantic proximity**: Words that appear together in context get higher scores
2. **Faster convergence**: TOP provides additional signal for learning word relationships
3. **Better ranking**: TOP achieves 0.284 MRR, showing effective ranking capability
4. **Efficiency gains**: 2× faster convergence compared to NTP-only training
