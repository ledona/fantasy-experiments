# Calculate Park Factor (PF)
Park Factor is a score for a venue that captures the scoring advantage/disadvantage of playing at a venue. This is most useful for baseball where multiple factors go into a venue being "hitter friendly".

PF is a seasonal value for each venue. It is based on the previous 3 seasons and has a recency weighting. League mean score is 100, over 100 is hitter friendly, under 100 is pitcher friendly.

The formula is:
$$PF = 100 \cdot \left( \frac{\frac{\text{Runs Scored (Home)} + \text{Runs Allowed (Home)}}{\text{Games (Home)}}}{\frac{\text{Runs Scored (Away)} + \text{Runs Allowed (Away)}}{\text{Games (Away)}}} \right)$$

This is, a venue, the average total runs scored at that venue per game (by the home team and opposing teams), over the average total runs scored away from that venue in games played by the home team for that venue. Take that value and multiple by 100 (so that 100 becomes the neutral point).

This is calculated for every venue for every season for that season's venue PF. The PF _feature_ for a venue+season is then the weighted average of previous seasons for that venue.