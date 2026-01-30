# Venue Features
The venue_features.py script is used to calculate the following venue featurs:
- Park Factor: Scores a parks offense/defense friendliness. < 1 means it is unfriendly to offense, > 1 is friendly to offense. (best for mlb)
- Home Venue Advantage: Score for the advantage a venue gives in score to the home team. (best for non-mlb)


## Calculate Park Factor (PF)
Park Factor is a score for a venue that captures the scoring advantage/disadvantage of playing at a venue. This is most useful for baseball where multiple factors go into a venue being "hitter friendly".

PF is a seasonal value for each venue. It is based on the previous 3 seasons and has a recency weighting. League mean score is 100, over 100 is hitter friendly, under 100 is pitcher friendly.

The formula is:
$$PF = 100 \cdot \left( \frac{\frac{\text{Runs Scored (Home)} + \text{Runs Allowed (Home)}}{\text{Games (Home)}}}{\frac{\text{Runs Scored (Away)} + \text{Runs Allowed (Away)}}{\text{Games (Away)}}} \right)$$

This is the average total runs scored at that venue per game (by the home team and opposing teams), over the average total runs scored away from that venue in games played by the home team for that venue. Take that value and multiple by 100 (so that 100 equals no effect/neutral).

This is calculated for every venue for every season to get that venue's effect for the season. The feature used for inference is the weighted average of park effect for the last couple seasons.

## Home Venue Advantage (HVA)
HVA is a seasonally (previous N seasons for predicted season) weighted point-differential boost for a team when playing at their home venue. This is essentially 