# PQBENCH: Empirical Latency Characterization of Post-Quantum Cryptography on Cloud Infrastructure

**Author:** Kanthi Phoosorn — Mae Fah Luang University, Thailand  
**Date:** March 2026  

## 🔑 Key Findings
| Algorithm | FIPS | Mean Latency | vs RSA-2048 |
|---|---|---|---|
| ECDSA-P256 | — | 0.012 ms | 2,477× faster |
| ML-KEM-512 | 203 | 0.695 ms | 42.8× faster ✅ |
| ML-KEM-768 | 203 | 1.348 ms | 22.0× faster ✅ |
| SLH-DSA-sha2-128f | 205 | 1.654 ms | 18.0× faster ✅ |
| ML-DSA-2 | 204 | 1.980 ms | 15.0× faster ✅ |
| ML-KEM-1024 | 203 | 2.149 ms | 13.8× faster ✅ |
| ML-DSA-5 | 204 | 6.777 ms | 4.4× faster ✅ |
| RSA-2048 | — | 29.73 ms | baseline |
| SLH-DSA-sha2-128s | 205 | 106.1 ms | 3.6× slower ❌ |

## 📄 Paper
- [Full paper (Markdown)](paper_v3.md)
- [Full paper (PDF)](paper_v2.pdf)

## 🛠️ Reproduce
```bash
pip install pycryptodome numpy scipy matplotlib
python benchmark_v2.py
```

## 📁 Files
- `paper_v3.md` — Full IEEE-format paper
- `paper_v2.pdf` — PDF version
- `benchmark_v2.py` — Benchmark code
- `results_v2.json` — Raw results (10 seeds)
- `charts/` — Figures

## 🔗 Author
- LinkedIn: [kanthi-phoosorn-238644392](https://linkedin.com/in/kanthi-phoosorn-238644392)
- GitHub: [KanthiPhoosorn](https://github.com/KanthiPhoosorn)
- Portfolio: [kanthi-cloud-portfolio.s3-website-ap-southeast-2.amazonaws.com](http://kanthi-cloud-portfolio.s3-website-ap-southeast-2.amazonaws.com)
```

MIT License

Copyright (c) 2026 Kanthi Phoosorn

Permission is hereby granted, free of charge, to any 
person obtaining a copy of this software and associated 
documentation files, to deal in the Software without 
restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF 
ANY KIND.
ANY KIND.
