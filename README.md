This is a Python implementation of the methodology described in the paper:

[**Low-Rank Bandit Methods for High-Dimensional Dynamic Pricing**](https://arxiv.org/abs/1801.10242)
Jonas Mueller, Vasilis Syrgkanis, Matt Taddy.  *NeurIPS* (2019)


Each of the main functions in **LowRankPricing.py** proposes prices for all the products in each round, based on the demands observed from the previous round.

**example.py** shows how one can use these functions for dynamic pricing (in an environment with simulated demands where the true optimal prices and resulting regret can be determined).