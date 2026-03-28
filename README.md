# PQBENCH: Post-Quantum Cryptography Benchmarking

**Author:** Kanthi Phoosorn — Mae Fah Luang University, Thailand
**Date:** March 2026

## Key Findings
| Algorithm | Mean Latency | vs RSA-2048 |
|---|---|---|
| ECDSA-P256 | 0.012 ms | 2,477× faster |
| ML-KEM-512 | 0.695 ms | 42.8× faster ✅ |
| ML-DSA-2 | 1.980 ms | 15.0× faster ✅ |
| SLH-DSA-sha2-128f | 1.654 ms | 18.0× faster ✅ |
| RSA-2048 | 29.73 ms | baseline |
| SLH-DSA-sha2-128s | 106.1 ms | 3.6× slower ❌ |

## Reproduce
pip install pycryptodome numpy scipy matplotlib
python benchmark_v2.py

## Links
- LinkedIn: linkedin.com/in/kanthi-phoosorn-238644392
- GitHub: github.com/KanthiPhoosorn
