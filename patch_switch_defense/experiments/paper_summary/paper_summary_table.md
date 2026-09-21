| Metric | all_switch | local_switch |
|---|---|---|
| Recovery rate (unconditional, same images/detection) | 96.5% | 89.4% |
| System accuracy (attacked eval set) | 66.0% | 65.3% |
| Clean false-positive cost (delta vs P16 clean) | +4.0% | +0.7% |
| Naive joint attack complete-defeat rate | 18.4% [9.2%,33.4%] | 16.7% [8.3%,30.6%] |
| Adaptive evasion worst-case (calibrated_threshold, representative) | 15.8% [7.4%,30.4%] | 21.4% [11.7%,35.9%] (ref. clean_max 11.9%) |
| Pipeline extra cost (escalate 시, batch=1) | 6.91 ms | 2.58 ms |
