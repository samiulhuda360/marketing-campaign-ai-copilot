# AI analyst evaluation

12 business questions with gold SQL (`eval/questions.jsonl`), scored by execution accuracy (see `campaign_copilot/evaluate.py`).

| Model | Execution accuracy | Median time per question |
|---|---|---|
| `openai/gpt-4.1-mini` | **100%** (12/12) | 2.0 s |
| `qwen/qwen3.8-27b:free` | **100%** (12/12) | 3.2 s |

### `openai/gpt-4.1-mini`

| | Question | Agent's answer (first sentence) | Queries | Seconds |
|---|---|---|---|---|
| ✓ | How many customers are there in total? | There are a total of 2,236 customers in the database. | 1 | 2.2 |
| ✓ | What share of customers accepted the last campaign offer? | The share of customers who accepted the last campaign offer is 14.9% | 1 | 1.9 |
| ✓ | Which segment has the highest average predicted score? | The segment "Meat buyers, big spenders" has the highest average predicted score with an average score of 0.307 | 1 | 2.3 |
| ✓ | How many customers are in the top score decile? | There are 224 customers in the top score decile (Decile = 1) | 1 | 1.9 |
| ✓ | What is the average income of customers who responded last time compared with those who did not? | The average income of customers who responded to the last campaign was $60,210, while the average income of those who did not respond was $5 | 1 | 2.0 |
| ✓ | Which education level has the highest past response rate? | The education level with the highest past response rate is PhD, with a past response rate of 20.8% | 1 | 2.3 |
| ✓ | How many customers have complained in the last two years? | There are 20 customers who have complained in the last two years | 1 | 1.6 |
| ✓ | What is the total wine spend of customers in the top score decile? | The total wine spend of customers in the top score decile is $159,496 | 1 | 2.1 |
| ✓ | How many customers have not bought anything for more than 90 days? | There are 197 customers who have not bought anything for more than 90 days | 1 | 1.7 |
| ✓ | If we contact the 200 customers with the highest scores, how many responders should we expect? | If you contact the 200 customers with the highest scores, you should expect approximately 104 responders based on their predicted probabilit | 1 | 1.8 |
| ✓ | Which marital status group spends the most on average? | The marital status group that spends the most on average is Widows, with an average spend of $739 | 1 | 2.0 |
| ✓ | What is the average number of catalogue purchases in each segment? | Here is the average number of catalogue purchases in each segment:  / Segment                        / Avg Catalogue Purchases / /---------- | 1 | 2.4 |

### `qwen/qwen3.8-27b:free`

| | Question | Agent's answer (first sentence) | Queries | Seconds |
|---|---|---|---|---|
| ✓ | How many customers are there in total? | There are **2,236 customers** in total in the database | 1 | 2.1 |
| ✓ | What share of customers accepted the last campaign offer? | **14.9%** of customers accepted the last campaign offer — 334 out of 2,236 customers responded | 1 | 3.6 |
| ✓ | Which segment has the highest average predicted score? | **"Meat buyers, big spenders"** has the highest average predicted score at **30.7%** (0.307), ahead of "Deal seekers, families" at 17.6% | 1 | 11.1 |
| ✓ | How many customers are in the top score decile? | There are **224 customers** in the top score decile (Decile = 1), the top 10% most likely to respond to the next offer | 1 | 1.8 |
| ✓ | What is the average income of customers who responded last time compared with those who did not? | Customers who responded to the last campaign had a higher average income than those who did not:  / Last campaign response / Customers / Avg | 1 | 33.2 |
| ✓ | Which education level has the highest past response rate? | **PhD** has the highest past response rate at **20.8%** (101 of 485 customers responded to the last campaign) | 1 | 2.4 |
| ✓ | How many customers have complained in the last two years? | **20 customers** have complained in the last two years | 1 | 2.2 |
| ✓ | What is the total wine spend of customers in the top score decile? | Customers in the top score decile (Decile = 1, the top 10% most likely to respond) total **$159,496** in wine spend across 224 customers | 1 | 26.6 |
| ✓ | How many customers have not bought anything for more than 90 days? | **197 customers** have not bought anything for more than 90 days (Recency > 90 days) | 1 | 2.4 |
| ✓ | If we contact the 200 customers with the highest scores, how many responders should we expect? | Contacting the 200 highest-scored customers should yield about **104 responders** (sum of their predicted response probabilities = 104.3, i. | 1 | 3.2 |
| ✓ | Which marital status group spends the most on average? | **Widow** customers spend the most on average: **$739** per customer (2-year total spend), ahead of Divorced ($613), Single ($607), and Marr | 1 | 25.1 |
| ✓ | What is the average number of catalogue purchases in each segment? | Average catalogue purchases by segment (2-year window):  / Segment / Avg catalogue purchases / Customers / /---/---/---/ / Meat buyers, big  | 1 | 3.1 |
