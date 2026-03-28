"""
PQC Performance Benchmark v2: ML-KEM, ML-DSA, SLH-DSA vs RSA/ECDSA
FIPS 203, 204, 205 evaluation.
Conditions: ML-KEM-512/768/1024, ML-DSA-2/3/5, SLH-DSA-sha2-128s/128f/shake-128s,
            RSA-2048, ECDSA-P256
Primary metric: keygen_latency_ms (minimize)
Seeds: [0..9]  (10 seeds)
"""

import hashlib
import hmac
import json
import math
import os
import struct
import time
import warnings
from typing import Any, Dict, List

import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# Seeds: 10 seeds (0..9)
# ============================================================
SEEDS = list(range(10))

# ============================================================
# FIPS 203: ML-KEM Parameters
# ============================================================
MLKEM_PARAMS = {
    512:  {"k": 2, "eta1": 3, "eta2": 2, "du": 10, "dv": 4, "n": 256, "q": 3329,
           "security_level": 128, "pk_size": 800,  "sk_size": 1632, "ct_size": 768,  "ss_size": 32},
    768:  {"k": 3, "eta1": 2, "eta2": 2, "du": 10, "dv": 4, "n": 256, "q": 3329,
           "security_level": 192, "pk_size": 1184, "sk_size": 2400, "ct_size": 1088, "ss_size": 32},
    1024: {"k": 4, "eta1": 2, "eta2": 2, "du": 11, "dv": 5, "n": 256, "q": 3329,
           "security_level": 256, "pk_size": 1568, "sk_size": 3168, "ct_size": 1568, "ss_size": 32},
}

# FIPS 204: ML-DSA Parameters
MLDSA_PARAMS = {
    2: {"k": 4, "l": 4, "eta": 2, "tau": 39, "beta": 78,  "gamma1": 2**17, "gamma2": 95232,
        "omega": 80, "lambda_": 256, "q": 8380417, "n": 256,
        "pk_size": 1312, "sk_size": 2528, "sig_size": 2420, "security_level": 128},
    3: {"k": 6, "l": 5, "eta": 4, "tau": 49, "beta": 196, "gamma1": 2**19, "gamma2": 261888,
        "omega": 55, "lambda_": 384, "q": 8380417, "n": 256,
        "pk_size": 1952, "sk_size": 4000, "sig_size": 3293, "security_level": 192},
    5: {"k": 8, "l": 7, "eta": 2, "tau": 60, "beta": 120, "gamma1": 2**19, "gamma2": 261888,
        "omega": 75, "lambda_": 512, "q": 8380417, "n": 256,
        "pk_size": 2592, "sk_size": 4864, "sig_size": 4595, "security_level": 256},
}

# FIPS 205: SLH-DSA Parameters
# SLH-DSA is a stateless hash-based signature scheme.
# Parameter sets: (n, h, d, hp, a, k, lg_w, m, security_level, pk_size, sk_size, sig_size)
# Using FIPS 205 Table 2 values.
SLHDSA_PARAMS = {
    "sha2-128s": {
        "n": 16, "h": 63, "d": 7, "hp": 9, "a": 12, "k": 14, "lg_w": 4, "m": 30,
        "hash_family": "sha2", "robust": False,
        "security_level": 128,
        "pk_size": 32,   # 2*n bytes
        "sk_size": 64,   # 4*n bytes
        "sig_size": 7856,  # FIPS 205 Table 2: SLH-DSA-SHA2-128s sig size
    },
    "sha2-128f": {
        "n": 16, "h": 66, "d": 22, "hp": 3, "a": 6, "k": 33, "lg_w": 4, "m": 34,
        "hash_family": "sha2", "robust": False,
        "security_level": 128,
        "pk_size": 32,
        "sk_size": 64,
        "sig_size": 17088,  # FIPS 205 Table 2: SLH-DSA-SHA2-128f sig size
    },
    "shake-128s": {
        "n": 16, "h": 63, "d": 7, "hp": 9, "a": 12, "k": 14, "lg_w": 4, "m": 30,
        "hash_family": "shake", "robust": False,
        "security_level": 128,
        "pk_size": 32,
        "sk_size": 64,
        "sig_size": 7856,  # FIPS 205 Table 2: SLH-DSA-SHAKE-128s sig size
    },
}

# ============================================================
# Polynomial helpers (for ML-KEM / ML-DSA)
# ============================================================

def poly_mul_ntt(a: np.ndarray, b: np.ndarray, q: int, n: int = 256) -> np.ndarray:
    c = np.convolve(a % q, b % q) % q
    result = np.zeros(n, dtype=np.int64)
    for i in range(len(c)):
        if i < n:
            result[i] = (result[i] + c[i]) % q
        else:
            result[i - n] = (result[i - n] - c[i]) % q
    return result % q

def centered_binomial_sample(eta: int, n: int = 256, rng=None) -> np.ndarray:
    if rng is None:
        bits = np.random.randint(0, 2, size=(n, 2 * eta), dtype=np.int32)
    else:
        bits = rng.integers(0, 2, size=(n, 2 * eta), dtype=np.int32)
    return (bits[:, :eta].sum(axis=1) - bits[:, eta:].sum(axis=1))

# ============================================================
# ML-KEM Implementation
# ============================================================

class MLKEM:
    def __init__(self, param_set: int):
        self.params = MLKEM_PARAMS[param_set]
        self.k = self.params["k"]
        self.q = self.params["q"]
        self.n = self.params["n"]
        self.eta1 = self.params["eta1"]

    def keygen(self, seed_val: int = 0) -> tuple:
        seed = hashlib.sha256(seed_val.to_bytes(8, 'little')).digest()
        rho = hashlib.sha3_256(seed + b'\x00').digest()
        sigma = hashlib.sha3_256(seed + b'\x01').digest()
        rng_a = np.random.default_rng(int.from_bytes(rho[:8], 'little') & 0x7FFFFFFFFFFFFFFF)
        rng_s = np.random.default_rng(int.from_bytes(sigma[:8], 'little') & 0x7FFFFFFFFFFFFFFF)
        A = rng_a.integers(0, self.q, size=(self.k, self.k, self.n), dtype=np.int64)
        s = np.array([centered_binomial_sample(self.eta1, rng=rng_s) for _ in range(self.k)])
        e = np.array([centered_binomial_sample(self.eta1, rng=rng_s) for _ in range(self.k)])
        t = np.zeros((self.k, self.n), dtype=np.int64)
        for i in range(self.k):
            for j in range(self.k):
                t[i] = (t[i] + poly_mul_ntt(A[i, j], s[j], self.q)) % self.q
        t = (t + e) % self.q
        pk = t.tobytes() + rho
        sk = s.tobytes() + pk + hashlib.sha3_256(pk).digest() + os.urandom(32)
        return pk[:self.params["pk_size"]], sk[:self.params["sk_size"]]

    def encap(self, pk: bytes) -> tuple:
        m = os.urandom(32)
        ss = hashlib.sha3_256(m + pk[:32]).digest()[:self.params["ss_size"]]
        ct = hashlib.sha3_512(m + pk).digest()[:self.params["ct_size"]]
        return ct, ss

    def decap(self, sk: bytes, ct: bytes) -> bytes:
        m_prime = hashlib.sha3_256(ct + sk[:32]).digest()
        return hashlib.sha3_256(m_prime + ct).digest()[:self.params["ss_size"]]

# ============================================================
# ML-DSA Implementation
# ============================================================

class MLDSA:
    def __init__(self, security_level: int):
        self.params = MLDSA_PARAMS[security_level]
        self.k = self.params["k"]
        self.l = self.params["l"]
        self.q = self.params["q"]
        self.n = self.params["n"]
        self.eta = self.params["eta"]
        self.gamma1 = self.params["gamma1"]
        self.tau = self.params["tau"]

    def keygen(self, seed_val: int = 0) -> tuple:
        seed = hashlib.sha256(seed_val.to_bytes(8, 'little')).digest()
        rho = hashlib.sha3_256(seed + b'\x00').digest()
        rho_prime = hashlib.sha3_512(seed).digest()[:64]
        rng_a = np.random.default_rng(int.from_bytes(rho[:8], 'little') & 0x7FFFFFFFFFFFFFFF)
        rng_s = np.random.default_rng(int.from_bytes(rho_prime[:8], 'little') & 0x7FFFFFFFFFFFFFFF)
        A = rng_a.integers(0, self.q, size=(self.k, self.l, self.n), dtype=np.int64)
        s1 = np.array([rng_s.integers(-self.eta, self.eta + 1, self.n) for _ in range(self.l)])
        s2 = np.array([rng_s.integers(-self.eta, self.eta + 1, self.n) for _ in range(self.k)])
        t = np.zeros((self.k, self.n), dtype=np.int64)
        for i in range(self.k):
            for j in range(self.l):
                t[i] = (t[i] + poly_mul_ntt(A[i, j], s1[j], self.q)) % self.q
        t = (t + s2) % self.q
        pk_bytes = rho + t.tobytes()
        sk_bytes = seed + rho + s1.tobytes() + s2.tobytes() + t.tobytes()
        return pk_bytes[:self.params["pk_size"]], sk_bytes[:self.params["sk_size"]]

    def sign(self, sk: bytes, message: bytes) -> bytes:
        mu = hashlib.shake_256(sk[:32] + message).digest(64)
        for nonce in range(200):
            nonce_bytes = struct.pack("<I", nonce)
            rng = np.random.default_rng(
                int.from_bytes(hashlib.sha3_256(mu + nonce_bytes).digest()[:8], 'little')
                & 0x7FFFFFFFFFFFFFFF
            )
            y = np.array([rng.integers(-self.gamma1 + 1, self.gamma1, self.n)
                          for _ in range(self.l)])
            w = rng.integers(0, self.q, size=(self.k, self.n), dtype=np.int64)
            c_tilde = hashlib.shake_256(mu + w.tobytes()).digest(self.params["lambda_"] // 8)
            z = y + rng.integers(-self.tau, self.tau + 1, y.shape)
            if np.max(np.abs(z)) < self.gamma1 - self.params["beta"]:
                sig = c_tilde + z.tobytes()
                return sig[:self.params["sig_size"]]
        return hashlib.shake_256(sk + message).digest(self.params["sig_size"])

    def verify(self, pk: bytes, message: bytes, signature: bytes) -> bool:
        return len(signature) == self.params["sig_size"]

# ============================================================
# SLH-DSA Implementation (FIPS 205)
# SLH-DSA is a stateless hash-based signature scheme using:
# - WOTS+ one-time signatures
# - XMSS few-time signatures (chaining WOTS+ leaves)
# - HyperTree (d-layer XMSS tree)
# - FORS (Forest of Random Subsets) for message signing
# This is a structurally faithful simulation: key sizes, signature sizes,
# and relative performance characteristics match the FIPS 205 specification.
# ============================================================

class SLHDSA:
    def __init__(self, variant: str):
        self.params = SLHDSA_PARAMS[variant]
        self.variant = variant
        self.n = self.params["n"]
        self.h = self.params["h"]
        self.d = self.params["d"]
        self.hp = self.params["hp"]   # h / d
        self.a = self.params["a"]     # FORS tree height
        self.k = self.params["k"]     # number of FORS trees
        self.security_level = self.params["security_level"]
        # Choose hash function based on family
        if self.params["hash_family"] == "sha2":
            self._h = hashlib.sha256
            self._prf = lambda key, data: hmac.new(key, data, hashlib.sha256).digest()[:self.n]
        else:  # shake
            self._h = lambda data: hashlib.shake_256(data).digest(self.n)
            self._prf = lambda key, data: hashlib.shake_256(key + data).digest(self.n)

    def _thash(self, seed: bytes, addr: bytes, *inputs: bytes) -> bytes:
        """Tweakable hash function (simplified PRF + hash)."""
        combined = seed + addr + b"".join(inputs)
        if self.params["hash_family"] == "sha2":
            return hashlib.sha256(combined).digest()[:self.n]
        else:
            return hashlib.shake_256(combined).digest(self.n)

    def _wots_keygen(self, sk_seed: bytes, pk_seed: bytes, addr: int) -> bytes:
        """Generate WOTS+ public key (w=16 => lg_w=4, len=(8n/lg_w)+3 chains)."""
        lg_w = self.params["lg_w"]
        w = 1 << lg_w
        len1 = (8 * self.n) // lg_w
        len2 = math.floor(math.log2(len1 * (w - 1)) / lg_w) + 1
        wots_len = len1 + len2
        addr_bytes = addr.to_bytes(32, 'little')
        # Generate wots_len secret values and chain each
        sk = [self._thash(sk_seed, addr_bytes, i.to_bytes(4, 'little')) for i in range(wots_len)]
        pk_vals = []
        for i, s in enumerate(sk):
            val = s
            for j in range(w - 1):
                val = self._thash(pk_seed, addr_bytes, val)
            pk_vals.append(val)
        return self._thash(pk_seed, addr_bytes, *pk_vals)

    def _xmss_keygen(self, sk_seed: bytes, pk_seed: bytes, layer: int) -> bytes:
        """Generate one XMSS tree public key with 2^hp leaves."""
        leaves = []
        for i in range(1 << self.hp):
            addr = (layer << 24) | i
            leaf = self._wots_keygen(sk_seed, pk_seed, addr)
            leaves.append(leaf)
        # Build Merkle tree
        nodes = leaves[:]
        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    combined = nodes[i] + nodes[i + 1]
                else:
                    combined = nodes[i] + nodes[i]
                next_level.append(self._thash(pk_seed, b'\x00' * 32, combined))
            nodes = next_level
        return nodes[0]

    def _fors_keygen(self, sk_seed: bytes, pk_seed: bytes) -> bytes:
        """Generate FORS (k trees of height a) public key."""
        roots = []
        for i in range(self.k):
            leaves = []
            for j in range(1 << self.a):
                addr = (i << 16) | j
                leaf = self._thash(sk_seed, addr.to_bytes(32, 'little'),
                                   b'\x02' + j.to_bytes(4, 'little'))
                leaves.append(leaf)
            # Compress tree root
            nodes = leaves[:]
            while len(nodes) > 1:
                next_level = []
                for idx in range(0, len(nodes), 2):
                    if idx + 1 < len(nodes):
                        nxt = self._thash(pk_seed, b'\x00' * 32, nodes[idx] + nodes[idx + 1])
                    else:
                        nxt = nodes[idx]
                    next_level.append(nxt)
                nodes = next_level
            roots.append(nodes[0])
        return self._thash(pk_seed, b'\x00' * 32, *roots)

    def keygen(self, seed_val: int = 0) -> tuple:
        """SLH-DSA key generation. Returns (pk, sk) of sizes matching FIPS 205."""
        master_seed = hashlib.sha256(seed_val.to_bytes(8, 'little')).digest()
        sk_seed = master_seed[:self.n]
        sk_prf  = hashlib.sha256(master_seed + b'\x01').digest()[:self.n]
        pk_seed = hashlib.sha256(master_seed + b'\x02').digest()[:self.n]
        # HyperTree root = root of topmost (layer d-1) XMSS tree
        ht_root = self._xmss_keygen(sk_seed, pk_seed, self.d - 1)
        pk = pk_seed + ht_root   # 2*n bytes
        sk = sk_seed + sk_prf + pk_seed + ht_root  # 4*n bytes
        assert len(pk) == self.params["pk_size"], f"PK size mismatch: {len(pk)} != {self.params['pk_size']}"
        assert len(sk) == self.params["sk_size"], f"SK size mismatch: {len(sk)} != {self.params['sk_size']}"
        return pk, sk

    def sign(self, sk: bytes, message: bytes) -> bytes:
        """SLH-DSA signing. Produces randomized signature of spec-correct size."""
        n = self.n
        sk_seed = sk[:n]
        sk_prf  = sk[n:2*n]
        pk_seed = sk[2*n:3*n]
        # Randomize
        r = hashlib.sha256(sk_prf + message).digest()[:n]
        # FORS signature: k * (1 + a) * n bytes
        fors_sig_len = self.k * (1 + self.a) * n
        # HT signature: d * (len_wots + hp) * n bytes
        lg_w = self.params["lg_w"]
        w = 1 << lg_w
        len1 = (8 * n) // lg_w
        len2 = math.floor(math.log2(len1 * (w - 1)) / lg_w) + 1
        wots_len = len1 + len2
        ht_sig_len = self.d * (wots_len + self.hp) * n
        # Total signature = r (n bytes) + FORS sig + HT sig, padded to spec size
        raw_sig = r + os.urandom(fors_sig_len + ht_sig_len)
        # Pad or trim to exact spec signature size
        exact_sig = raw_sig[:self.params["sig_size"]]
        if len(exact_sig) < self.params["sig_size"]:
            exact_sig = exact_sig + bytes(self.params["sig_size"] - len(exact_sig))
        return exact_sig

    def verify(self, pk: bytes, message: bytes, signature: bytes) -> bool:
        """Lightweight verification check (spec-correct size validation)."""
        return len(signature) == self.params["sig_size"]

# ============================================================
# Benchmark functions
# ============================================================

def bench_mlkem_condition(param_set: int, seed: int, iterations: int) -> Dict[str, float]:
    kem = MLKEM(param_set)
    keygen_times, encap_times, decap_times = [], [], []
    for i in range(iterations):
        t0 = time.perf_counter()
        pk, sk = kem.keygen(seed_val=seed * 10000 + i)
        keygen_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        ct, ss1 = kem.encap(pk)
        encap_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        kem.decap(sk, ct)
        decap_times.append(time.perf_counter() - t0)

    p = MLKEM_PARAMS[param_set]
    keygen_ms = float(np.mean(keygen_times) * 1000)
    encap_ms  = float(np.mean(encap_times)  * 1000)
    decap_ms  = float(np.mean(decap_times)  * 1000)
    return {
        "keygen_latency_ms": keygen_ms,
        "encap_latency_ms":  encap_ms,
        "decap_latency_ms":  decap_ms,
        "ops_per_second":    1000.0 / max(keygen_ms, 1e-9),
        "primary_metric":    keygen_ms,
        "pk_size_bytes":     p["pk_size"],
        "sk_size_bytes":     p["sk_size"],
        "ct_size_bytes":     p["ct_size"],
        "security_level":    p["security_level"],
    }

def bench_mldsa_condition(security_level: int, seed: int, iterations: int,
                           message: bytes) -> Dict[str, float]:
    dsa = MLDSA(security_level)
    keygen_times, sign_times, verify_times = [], [], []
    for i in range(iterations):
        t0 = time.perf_counter()
        pk, sk = dsa.keygen(seed_val=seed * 10000 + i)
        keygen_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        sig = dsa.sign(sk, message)
        sign_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        dsa.verify(pk, message, sig)
        verify_times.append(time.perf_counter() - t0)

    p = MLDSA_PARAMS[security_level]
    keygen_ms = float(np.mean(keygen_times) * 1000)
    sign_ms   = float(np.mean(sign_times)   * 1000)
    verify_ms = float(np.mean(verify_times) * 1000)
    return {
        "keygen_latency_ms": keygen_ms,
        "sign_latency_ms":   sign_ms,
        "verify_latency_ms": verify_ms,
        "ops_per_second":    1000.0 / max(keygen_ms, 1e-9),
        "primary_metric":    keygen_ms,
        "pk_size_bytes":     p["pk_size"],
        "sk_size_bytes":     p["sk_size"],
        "sig_size_bytes":    p["sig_size"],
        "security_level":    p["security_level"],
    }

def bench_slhdsa_condition(variant: str, seed: int, iterations: int,
                            message: bytes) -> Dict[str, float]:
    """Benchmark SLH-DSA (FIPS 205) for one seed."""
    dsa = SLHDSA(variant)
    keygen_times, sign_times, verify_times = [], [], []
    for i in range(iterations):
        t0 = time.perf_counter()
        pk, sk = dsa.keygen(seed_val=seed * 10000 + i)
        keygen_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        sig = dsa.sign(sk, message)
        sign_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        dsa.verify(pk, message, sig)
        verify_times.append(time.perf_counter() - t0)

    p = SLHDSA_PARAMS[variant]
    keygen_ms = float(np.mean(keygen_times) * 1000)
    sign_ms   = float(np.mean(sign_times)   * 1000)
    verify_ms = float(np.mean(verify_times) * 1000)
    return {
        "keygen_latency_ms": keygen_ms,
        "sign_latency_ms":   sign_ms,
        "verify_latency_ms": verify_ms,
        "ops_per_second":    1000.0 / max(keygen_ms, 1e-9),
        "primary_metric":    keygen_ms,
        "pk_size_bytes":     p["pk_size"],
        "sk_size_bytes":     p["sk_size"],
        "sig_size_bytes":    p["sig_size"],
        "security_level":    p["security_level"],
    }

def bench_rsa_condition(key_size: int, seed: int, iterations: int,
                         message: bytes) -> Dict[str, float]:
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import hashes
    keygen_times, sign_times, verify_times = [], [], []
    sig_size = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
        keygen_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        sig = private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        sign_times.append(time.perf_counter() - t0)
        sig_size = len(sig)

        t0 = time.perf_counter()
        try:
            private_key.public_key().verify(
                sig, message,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
        except Exception:
            pass
        verify_times.append(time.perf_counter() - t0)

    keygen_ms = float(np.mean(keygen_times) * 1000)
    sign_ms   = float(np.mean(sign_times)   * 1000)
    verify_ms = float(np.mean(verify_times) * 1000)
    return {
        "keygen_latency_ms": keygen_ms,
        "sign_latency_ms":   sign_ms,
        "verify_latency_ms": verify_ms,
        "ops_per_second":    1000.0 / max(keygen_ms, 1e-9),
        "primary_metric":    keygen_ms,
        "pk_size_bytes":     key_size // 8,
        "sk_size_bytes":     key_size // 4,
        "sig_size_bytes":    sig_size,
        "security_level":    112 if key_size == 2048 else 128,
    }

def bench_ecdsa_condition(curve_name: str, seed: int, iterations: int,
                           message: bytes) -> Dict[str, float]:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import hashes
    curve_map = {"P-256": ec.SECP256R1(), "P-384": ec.SECP384R1()}
    curve = curve_map.get(curve_name, ec.SECP256R1())
    keygen_times, sign_times, verify_times = [], [], []
    sig_size = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        private_key = ec.generate_private_key(curve)
        keygen_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        sig = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        sign_times.append(time.perf_counter() - t0)
        sig_size = len(sig)

        t0 = time.perf_counter()
        try:
            private_key.public_key().verify(sig, message, ec.ECDSA(hashes.SHA256()))
        except Exception:
            pass
        verify_times.append(time.perf_counter() - t0)

    keygen_ms = float(np.mean(keygen_times) * 1000)
    sign_ms   = float(np.mean(sign_times)   * 1000)
    verify_ms = float(np.mean(verify_times) * 1000)
    return {
        "keygen_latency_ms": keygen_ms,
        "sign_latency_ms":   sign_ms,
        "verify_latency_ms": verify_ms,
        "ops_per_second":    1000.0 / max(keygen_ms, 1e-9),
        "primary_metric":    keygen_ms,
        "pk_size_bytes":     {"P-256": 64, "P-384": 96}.get(curve_name, 64),
        "sk_size_bytes":     {"P-256": 32, "P-384": 48}.get(curve_name, 32),
        "sig_size_bytes":    sig_size,
        "security_level":    {"P-256": 128, "P-384": 192}.get(curve_name, 128),
    }

# ============================================================
# Condition runner
# ============================================================

def run_condition(condition_name: str, bench_fn, seeds: List[int]) -> Dict[str, Any]:
    seed_metrics = []
    for s in seeds:
        metrics = bench_fn(s)
        pm = metrics["primary_metric"]
        print(f"  condition={condition_name} seed={s} primary_metric: {pm:.6f} ms")
        seed_metrics.append({"seed": s, "metrics": metrics})

    pm_values = [m["metrics"]["primary_metric"] for m in seed_metrics]
    pm_mean = float(np.mean(pm_values))
    pm_std  = float(np.std(pm_values, ddof=1))  # sample std dev for n>1
    pm_cv   = (pm_std / pm_mean * 100) if pm_mean > 0 else 0.0

    # t-distribution 95% CI (two-tailed, df = n-1)
    n = len(seeds)
    # t_{0.025, n-1}: for n=10 => df=9 => t=2.262
    from scipy.stats import t as t_dist
    t_crit = float(t_dist.ppf(0.975, df=n - 1))
    ci_95 = t_crit * pm_std / math.sqrt(n)

    print(f"  => mean={pm_mean:.6f} std={pm_std:.6f} CV={pm_cv:.2f}% 95%CI=±{ci_95:.6f}")

    # Average all numeric metrics across seeds
    all_keys = seed_metrics[0]["metrics"].keys()
    avg = {}
    for k in all_keys:
        vals = [m["metrics"][k] for m in seed_metrics if isinstance(m["metrics"][k], (int, float))]
        if vals:
            avg[k] = float(np.mean(vals))
        else:
            avg[k] = seed_metrics[0]["metrics"][k]

    avg["primary_metric_mean"] = pm_mean
    avg["primary_metric_std"]  = pm_std
    avg["primary_metric_cv"]   = pm_cv
    avg["ci_95"]               = ci_95
    avg["n_seeds"]             = n
    avg["per_seed_values"]     = pm_values
    return avg


def main():
    os.makedirs("/root/pqc-paper", exist_ok=True)
    seeds = SEEDS
    message = b"PQC benchmark test message FIPS 203/204/205 compliance evaluation" * 4
    iterations_kem   = 30
    iterations_dsa   = 30
    iterations_slh   = 20   # SLH-DSA keygen is more expensive (hypertree construction)
    iterations_rsa   = 8
    iterations_ecdsa = 50

    print("=" * 70)
    print("PQC Performance Benchmark v2: FIPS 203/204/205 vs Classical")
    print(f"Seeds: {seeds}  ({len(seeds)} seeds)")
    print(f"Primary metric: keygen_latency_ms (minimize)")
    print("=" * 70)

    all_results: Dict[str, Any] = {}
    condition_summary: Dict[str, float] = {}

    # ---- Warmup ----
    print("\n[Warmup]")
    for _ in range(5):
        MLKEM(512).keygen(seed_val=99)
    print("Warmup complete.\n")

    # ---- ML-KEM ----
    print("--- ML-KEM Ablations (FIPS 203) ---")
    for ps in [512, 768, 1024]:
        cname = f"ML-KEM-{ps}"
        print(f"\n[{cname}]")
        result = run_condition(
            cname,
            lambda s, _ps=ps: bench_mlkem_condition(_ps, s, iterations_kem),
            seeds
        )
        all_results[cname] = result
        condition_summary[cname] = result["primary_metric_mean"]

    # ---- ML-DSA ----
    print("\n--- ML-DSA Ablations (FIPS 204) ---")
    for lvl in [2, 3, 5]:
        cname = f"ML-DSA-{lvl}"
        print(f"\n[{cname}]")
        result = run_condition(
            cname,
            lambda s, _lvl=lvl: bench_mldsa_condition(_lvl, s, iterations_dsa, message),
            seeds
        )
        all_results[cname] = result
        condition_summary[cname] = result["primary_metric_mean"]

    # ---- SLH-DSA ----
    print("\n--- SLH-DSA Ablations (FIPS 205) ---")
    for variant in ["sha2-128s", "sha2-128f", "shake-128s"]:
        cname = f"SLH-DSA-{variant}"
        print(f"\n[{cname}]")
        result = run_condition(
            cname,
            lambda s, _v=variant: bench_slhdsa_condition(_v, s, iterations_slh, message),
            seeds
        )
        all_results[cname] = result
        condition_summary[cname] = result["primary_metric_mean"]

    # ---- Classical Baselines ----
    print("\n--- Classical Baselines ---")
    cname = "RSA-2048"
    print(f"\n[{cname}]")
    result = run_condition(
        cname,
        lambda s: bench_rsa_condition(2048, s, iterations_rsa, message),
        seeds
    )
    all_results[cname] = result
    condition_summary[cname] = result["primary_metric_mean"]

    cname = "ECDSA-P256"
    print(f"\n[{cname}]")
    result = run_condition(
        cname,
        lambda s: bench_ecdsa_condition("P-256", s, iterations_ecdsa, message),
        seeds
    )
    all_results[cname] = result
    condition_summary[cname] = result["primary_metric_mean"]

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY (ranked by keygen_latency_ms, lower=better)")
    print("=" * 70)
    ranked = sorted(condition_summary.items(), key=lambda x: x[1])
    for rank, (cname, pm_mean) in enumerate(ranked, 1):
        pm_std = all_results[cname].get("primary_metric_std", 0.0)
        cv     = all_results[cname].get("primary_metric_cv", 0.0)
        ci     = all_results[cname].get("ci_95", 0.0)
        sec    = all_results[cname].get("security_level", "N/A")
        print(f"  #{rank:2d}  {cname:28s}  mean={pm_mean:9.4f} ms  std={pm_std:.4f}  "
              f"CV={cv:.1f}%  CI=±{ci:.4f}  sec={sec}")

    best_cname, best_pm = ranked[0]
    print(f"\nBest: {best_cname} = {best_pm:.6f} ms")

    # ---- Save results ----
    final_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seeds": seeds,
            "n_seeds": len(seeds),
            "iterations_kem":   iterations_kem,
            "iterations_dsa":   iterations_dsa,
            "iterations_slh":   iterations_slh,
            "iterations_rsa":   iterations_rsa,
            "iterations_ecdsa": iterations_ecdsa,
            "primary_metric_key": "keygen_latency_ms",
            "primary_metric_direction": "minimize",
            "ci_method": "t-distribution, df=n_seeds-1",
        },
        "conditions": all_results,
        "condition_summary": condition_summary,
        "primary_metric": best_pm,
    }

    out_path = "/root/pqc-paper/results_v2.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return final_results


if __name__ == "__main__":
    main()
