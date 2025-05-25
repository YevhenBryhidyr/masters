import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import shared_memory, Barrier
from scipy.signal import sawtooth


def generate_weights_f(m):
    sigma = m
    x = np.arange(-m, m + 1)
    weights = np.exp(-x ** 2 / (2 * sigma ** 2))
    weights /= weights.sum()
    return weights


def generate_signal(fs, duration, f, type):
    t = np.linspace(0, duration, fs, endpoint=False)
    if type == "синус":
        clean_signal = np.sin(2 * np.pi * f * t)
    elif type == "косинус":
        clean_signal = np.cos(2 * np.pi * f * t)
    elif type == "східчастий":
        x = np.arange(fs)
        clean_signal = np.where(x < fs // 2, 0, 1)
    elif type == "пилкоподібний":
        clean_signal = sawtooth(2 * np.pi * f * t)
    elif type == "трикутний":
        clean_signal = sawtooth(2 * np.pi * f * t, 0.5)
    elif type == "квадратний":
        clean_signal = np.sign(np.sin(2 * np.pi * f * t))

    return clean_signal, t


def generate_noise(rnd, fs, type, std_dev=0.2):
    if not rnd:
        np.random.seed(42)
    if type == "нормальний":
        noise = np.random.normal(0, std_dev, fs)
    elif type == "імпульсний":
        num_impulses = int(std_dev)
        amplitude_range = (-3, 3)

        impulse_indices = np.random.choice(fs, size=num_impulses, replace=False)
        impulse_values = np.random.uniform(*amplitude_range, size=num_impulses)

        noise = np.zeros(fs)
        noise[impulse_indices] = impulse_values

    return noise


def calc_one_var(args):
    t, k1, m, f, constant, shm_name, n = args

    shm = shared_memory.SharedMemory(name=shm_name)
    x = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)

    x_prev = x.copy()
    x_new = x.copy()

    for j in range(1, k1+1):
        for i in range(max(1, m*(j-k1)+t) - 1, min(n, m*(k1-j)+t)):
            p = 0
            for s in range(-m, m+1):
                if 0 <= i + s < n:
                    xis = x_prev[i+s]
                else:
                    constant = x_prev[i]
                    xis = constant
                p = p + xis * f[s + m]
            x_new[i] = p
        x_prev = x_new.copy()

    shm.close()
    return x_prev[t-1]


def calc_one_var_limited(args):
    start, end, m, f, constant, k1, shm_in, shm_out, n, bar, r, p = args
    shm_in = shared_memory.SharedMemory(name=shm_in)
    shm_out = shared_memory.SharedMemory(name=shm_out)
    x_in = np.ndarray((n,), dtype=np.float64, buffer=shm_in.buf)
    x_out = np.ndarray((n,), dtype=np.float64, buffer=shm_out.buf)
    t = r + 1

    base, extra = divmod(n, p)
    if t <= extra:
        chunk_start = (t - 1) * (base + 1)
        chunk_end = chunk_start + (base + 1)
    else:
        chunk_start = extra * (base + 1) + (t - extra - 1) * base
        chunk_end = chunk_start + base

    for j in range(k1):
        lo = max(1, m * (j - k1) + chunk_start + 1) - 1
        hi = min(n, m * (k1 - j) + chunk_end)
        for i in range(lo, hi):
            acc = 0.0
            for s in range(-m, m + 1):
                idx = i + s
                if 0 <= idx < n:
                    acc += x_in[idx] * f[s + m]
                else:
                    constant = x_in[i]
                    acc += constant * f[s + m]
            x_out[i] = acc

        bar.wait()

        if start == 0:
            x_in[:], x_out[:] = x_out, x_in

        bar.wait()

    shm_in.close()
    shm_out.close()


def show_results(clean_signal, noisy_signal, x_seq, x_par, t):
    plt.figure(figsize=(10, 4))
    plt.plot(t, noisy_signal, label='Сигнал з шумом')
    plt.plot(t, clean_signal, label='Чистий сигнал', linewidth=2)
    plt.plot(t, x_seq, label='Відфільтрований сигнал', color='red', linewidth=2)
    plt.legend()
    plt.xlabel('Час [с]')
    plt.ylabel('Амплітуда')
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(t, x_par, label='Відфільтрований сигнал (паралельне виконання)', color='blue', linewidth=1)
    plt.legend()
    plt.xlabel('Час [с]')
    plt.ylabel('Амплітуда')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def find_mse(x_seq, x_par, clean_signal):
    mse = np.mean((x_seq - clean_signal) ** 2)
    mse_par = np.mean((x_par - clean_signal) ** 2)

    return mse, mse_par


def find_snr(clean_signal, noisy_filtered_signal):
    signal_power = np.mean(clean_signal ** 2)
    noisy_filtered_power = np.mean((clean_signal - noisy_filtered_signal) ** 2)

    snr = 10 * np.log10(signal_power / noisy_filtered_power)

    return snr


def is_equal(x_seq, x_par):
    return np.allclose(x_seq, x_par, atol=1e-10)


class DFP:
    def __init__(self, p, x):
        self.p = p
        self.x = x.copy()

    def standard_sequential(self, k1, m, f):
        start_time = time.perf_counter_ns()
        n = len(self.x)
        constant = 0
        x_prev = self.x.copy()
        x_new = self.x.copy()
        for j in range(k1):
            for i in range(n):
                p = 0
                for s in range(-m, m+1):
                    if 0 <= i + s < n:
                        xis = x_prev[i + s]
                    else:
                        constant = x_prev[i]
                        xis = constant
                    p = p + xis * f[s + m]

                x_new[i] = p
            x_prev = x_new.copy()

        end_time = time.perf_counter_ns()
        time_for_seq = end_time - start_time

        return x_prev, time_for_seq

    def standard_parallel(self, k1, m, f, constant=0):
        f = np.asarray(f, dtype=float)
        n = len(self.x)
        x_par = self.x.copy()

        shm = shared_memory.SharedMemory(create=True, size=x_par.nbytes)
        shm_array = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
        shm_array[:] = x_par[:]
        shm_name = shm.name

        start_time_par = time.perf_counter_ns()
        with multiprocessing.Pool(processes=None) as pool:
            args = ((t, k1, m, f, constant, shm_name, n) for t in range(1, n+1))

            x_par = pool.map(calc_one_var, args)


        end_time_par = time.perf_counter_ns()
        time_for_par = end_time_par - start_time_par

        shm.close()
        shm.unlink()

        return x_par, time_for_par

    def limited_parallel(self, x, k, m, f, constant=0, P=None):
        x = np.asarray(x, dtype=np.float64)
        f = np.asarray(f, dtype=np.float64)
        n = len(x)
        P = P or multiprocessing.cpu_count()

        shm_a = shared_memory.SharedMemory(create=True, size=x.nbytes)
        shm_b = shared_memory.SharedMemory(create=True, size=x.nbytes)
        x_prev = np.ndarray(x.shape, dtype=np.float64, buffer=shm_a.buf)
        x_curr = np.ndarray(x.shape, dtype=np.float64, buffer=shm_b.buf)
        x_prev[:] = x

        bar = Barrier(P)

        base, extra = divmod(n, P)
        ranges = []
        s = 0
        for r in range(P):
            e = s + base + (1 if r < extra else 0)
            ranges.append((s, e, r))
            s = e

        procs = []
        start_time = time.perf_counter_ns()

        for (s, e, r) in ranges:
            args = (s, e, m, f, constant, k, shm_a.name, shm_b.name, n, bar, r, P)
            p = multiprocessing.Process(target=calc_one_var_limited, args=(args, ))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        result = np.ndarray(x.shape, dtype=np.float64, buffer=shm_a.buf).copy()

        end_time = time.perf_counter_ns()

        shm_a.close()
        shm_a.unlink()
        shm_b.close()
        shm_b.unlink()

        time_for_lim_par = end_time - start_time
        return result, time_for_lim_par
