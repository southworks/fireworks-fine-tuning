# Cost and deployment

## Fireworks deployment configuration supplied for this project

- Accelerator: 1 x NVIDIA B200 180 GB.
- Precision: BF16.
- Autoscaling range: zero to one GPU.
- Listed dedicated-deployment range: USD 0 to USD 13.00 per hour.

At continuous maximum allocation, 730 hours at USD 13/hour would be approximately USD 9,490/month. Scale-to-zero can materially reduce this when traffic is intermittent, but actual cost depends on active GPU time, cold starts, throughput, batching, and utilization.

## What can and cannot be concluded

The benchmark demonstrates an accuracy advantage. It does **not**, by itself, prove a universal cost advantage over GPT-4.1 mini. A fair comparison must use the same workload and include:

- monthly requests and input/output tokens;
- average and peak concurrency;
- measured tokens per second;
- GPU active hours and idle policy;
- serverless or API token prices at the time of measurement;
- training cost amortized across expected lifetime volume.

## Scenario template

| Input | Value |
|---|---:|
| Requests/month | TBD |
| Mean input tokens | TBD |
| Mean output tokens | TBD |
| Peak requests/second | TBD |
| Qwen GPU active hours/month | TBD |
| Qwen training cost | Not present in source snapshot |
| GPT-4.1 mini price snapshot | TBD with date/source |

The deployment price above was supplied with the project brief and should be verified against the current Fireworks quote before an external presentation.
