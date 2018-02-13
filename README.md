# nba-salary-predictor

A script that predicts the salary of an NBA player given a series of figures ranging from the player's statistics from the previous season to his social media activity and following.

The predictor was created using data from just 230 players, which prevents supremely accurate results. However, the following accuracies were found:

Logistic Regression Model: MSE of approximately 3.5
DEEP Neural Network: MSE of approximately 3.8 

This accuracy is relatively good, and would allow players to gauge how much they will be paid in upcoming free agency deals when their contracts expire.
A good utilization of such a script would be when players are negotiating contracts with teams, as they could use such a model to bargain their wages.
Another possible utilization could be for a player exiting his rookie contract, as he would never have been in the free agency market before. The model thus provides an indication as to which tier of contracts he should be looking at, and thus allows him to understand whether he should take his current team's contract, should they offer one.

However, due to the unpredictable nature of the NBA market and the cap nature of teams, the models are ultimately unlikely to be able to nail down concretely accurate salaries for players.
