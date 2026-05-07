# Dataset Documentation

## Dataset: CIC-DDoS2019

**Source:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick
**Cleaned version used:** [dhoogla/cicddos2019 on Kaggle](https://www.kaggle.com/datasets/dhoogla/cicddos2019)
**Original raw data:** also downloaded locally for reference (not used for training — see below)
**License:** Public research use

---

## Why we use the Kaggle cleaned version

The raw CIC-DDoS2019 CSVs contain critical data quality issues that break deep learning training:

| Issue | Raw | Kaggle cleaned |
| ----- | --- | -------------- |
| Inf values (in 100k sample) | 8,306 | 0 |
| Null values | present | 0 |
| Identifier columns (Flow ID, IPs, Timestamp) | present → data leakage | removed |
| Duplicate rows | present | removed |
| File size | ~29 GB total | ~32 MB total (parquet) |
| Format | CSV (inconsistent spacing) | Parquet (typed, compressed) |

Inf values in `Flow Bytes/s` and `Flow Packets/s` are caused by CICFlowMeter dividing by zero-duration flows. They propagate through PyTorch as NaN loss, silently corrupting all model weights.

---

## Classes

**Binary classification:** all attack labels → `ddos`, benign → `normal`

| Original Label | Binary | Count |
| -------------- | ------ | ----- |
| Benign | normal | 97,831 |
| DrDoS_DNS | ddos | 3,669 |
| DrDoS_LDAP | ddos | 1,440 |
| DrDoS_MSSQL | ddos | 6,212 |
| DrDoS_NetBIOS | ddos | 598 |
| DrDoS_NTP | ddos | 121,368 |
| DrDoS_SNMP | ddos | 2,717 |
| DrDoS_UDP | ddos | 10,420 |
| LDAP | ddos | 1,906 |
| MSSQL | ddos | 8,523 |
| NetBIOS | ddos | 644 |
| Portmap | ddos | 685 |
| Syn | ddos | 49,373 |
| TFTP | ddos | 98,917 |
| UDP | ddos | 18,090 |
| UDP-lag / UDPLag | ddos | 8,927 |
| WebDDoS | ddos | 51 |
| **Total** | | **431,371** |

---

## Features Used (17)

Selected from 78 available CICFlowMeter features based on temporal DDoS relevance:

```text
Flow Duration, Total Fwd Packets, Total Backward Packets,
Fwd Packets Length Total, Bwd Packets Length Total,
Fwd Packet Length Mean, Bwd Packet Length Mean,
Flow Bytes/s, Flow Packets/s,
Fwd IAT Mean, Bwd IAT Mean,
SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count,
Avg Packet Size, Avg Fwd Segment Size
```

---

## Preprocessing Steps Applied

1. Load all 17 Kaggle parquet files
2. Normalize label names (handle spelling variants)
3. Map to binary: `normal` (0) / `ddos` (1)
4. Keep only 17 target features + label
5. Drop rows with NaN or Inf (none expected after Kaggle cleaning)
6. Stratified 70/15/15 train/val/test split (random seed 42)
7. StandardScaler fitted on train only, applied to val and test
8. Sliding window (seq_len=10) applied for LSTM/GRU sequences

**Important:** The Kaggle dataset includes train/test split files by capture day. We **ignore** that pre-built split and re-split from scratch stratified by binary label. This ensures all attack types appear proportionally in every split.

---

## Acquisition

1. Download from [Kaggle](https://www.kaggle.com/datasets/dhoogla/cicddos2019)
2. Place all `.parquet` files in `data/processed/kaggle/archive/`
3. Run `notebooks/02_preprocessing.ipynb` to generate `data/processed/splits/`

---

## Connection to Lab Research

This project is the classifier comparison component of the GANDD-Bridge research
(`github.com/calvinkatoroy/wazuh-gan-ddos-research`). That research uses a Random Forest
discriminator inside Wazuh SIEM — this project benchmarks whether LSTM/GRU can
replace or improve upon that RF baseline on real-world captured traffic.
