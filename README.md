# Bubble Bursting Implementations and Dynamics In Recommender Systems
Welcome to the repository for Jannik and Madison's Final Project for COMS 6998: Fair and Robust Algorithms!

## TL;DR
Recommender systems (RSs), which offer personalized suggestions to users, power many of today's social media, e-commerce, and entertainment services. However, they have been known to create filter bubbles, a term used to describe the phenomenon wherein users are only shown content that aligns with their preconceived tastes and opinions. Past research has shown filter bubbles to be related to ideological extremism and systematic bias against marginalized groups. We mitigate such effects by introducing new notions of exploration into our recommender system (RS) model and measuring the trade-offs between accuracy and other metrics, such as diversity, novelty, and serendipity. We also examine the effect of filter bubble-mitigation strategies on user behavior homogenization by analyzing inter- versus intra-bubble homogeneity dynamics.

The full paper can be found here: [Bubble Bursting Implementations and Dynamics In Recommender Systems](/Bubble_Bursting_RS_Paper.pdf)

## Implementation
To replicate experiments in our study, you can run the experiments using [run_experiment.py](/run_experiment.py). 
A simple experiment with a myopic recommender system can be run using the following command
```bash
python3 run_experiment.py
```
You can customize this using the parameters in run_experiment.py. You can find the possible configurations using
```bash
python3 run_experiment.py -h
```

You can also easily interact with and see the outputs of such an experiment in interactive_simulation.py.

## Conceptual Description
### Simulation Environment
To observe the long-term effects of changing user preferences due to recommender systems, we use the agent-based T-RECS simulation environment. The general architecture of the T-RECs environment can be seen below. Importantly, we influence the user preferences over time by changing the recommended items. A users' preferences "drift" towards the characteristics of the items they select.

![image](https://user-images.githubusercontent.com/29771705/214890603-16e474cd-de43-433e-a844-7c8a5c300e4b.png)

### Results
In the following, we show a few examples of our results. The two figures below show the trade-off between accuracy and exploration by means of MSE, recall, diversity, novelty, and serendipity.

First, this trade-off is indicated by the MSE and diversity curves. Cosine similarity (λ = 0.01) has the highest error and the highest diversity. Cosine regularization with λ = 0.1 is the opposite example, MSE is the lowest for most timesteps and diversity is continually increasing (albeit at a low absolute value). This can make it a good long-term focused algorithm. Additionally, a higher λ value indicates both higher novelty convergence and higher serendipity. It is
however important to note that cosine similarity regularization displays numerical instability and thus
needs further development.
Second, entropy regularization outperforms the accuracy of the myopic policy in both recall and MSE. This supports the argument that greedy optimization might not yield the highest accuracy and that exploration is valuable also from an accuracy perspective.
Third, we note that top-k re-ranking does not increase any of the exploration metrics and thus does
not achieve our desired outcome.

![image](https://user-images.githubusercontent.com/29771705/214891657-4f080d66-5b16-4551-b973-d0b1ab5ca261.png)

![image](https://user-images.githubusercontent.com/29771705/214891656-65b8285c-a208-402a-a756-0c033ffb071d.png)
