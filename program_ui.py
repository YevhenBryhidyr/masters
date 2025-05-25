import tkinter as tk
import tkinter.ttk
from tkinter import Tk, Label, Entry, Button
import dfp as dfp


class MainWindow(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("Задача цифрової фільтрації")
        self.geometry("1200x900")

        self.x_seq = None
        self.x_par = None
        self.x_lim_par = None
        self.time_seq = None
        self.time_par = None
        self.clean_signal = None
        self.noisy_signal = None
        self.t = None

        Label(self, text="Введіть розмір вибірки (на секунду):").pack()
        self.n_input = Entry(self, width=35)
        self.n_input.pack()
        Label(self, text="Введіть тривалість сигналу (в секундах):").pack()
        self.duration_input = Entry(self, width=35)
        self.duration_input.pack()
        Label(self, text="Введіть к-сть процесів (стандартно к-сть ядер*2):").pack()
        self.p_input = Entry(self, width=35)
        self.p_input.pack()
        Label(self, text="Введіть кількість етапів переобчислень:").pack()
        self.k_input = Entry(self, width=35)
        self.k_input.pack()
        Label(self, text="Введіть розмір рухомого вікна:").pack()
        self.m_input = Entry(self, width=35)
        self.m_input.pack()

        self.lim_par = tk.BooleanVar(value=False)
        check_lim_par = tk.Checkbutton(self, text="Обмежений паралелізм", variable=self.lim_par)
        check_lim_par.pack()

        Label(self, text="Виберіть тип сигналу:").pack()
        self.signal_chooser = tkinter.ttk.Combobox(state="readonly", values=["синус", "косинус", "східчастий",
                                                                             "пилкоподібний", "трикутний", "квадратний"])
        self.signal_chooser.current(0)
        self.signal_chooser.pack()
        Label(self, text="Введіть частоту сигналу:").pack()
        self.freq = Entry(self, width=35)
        self.freq.pack()

        self.signal_rnd = tk.BooleanVar(value=False)
        check = tk.Checkbutton(self, text="Випадкова генерація шуму", variable=self.signal_rnd)
        check.pack()

        Label(self, text="Виберіть тип шуму:").pack()
        self.noise_chooser = tkinter.ttk.Combobox(state="readonly", values=["нормальний", "імпульсний"])
        self.noise_chooser.current(0)
        self.noise_chooser.pack()
        Label(self, text="Введіть силу шуму / кількість імпульсних завад:").pack()
        self.std_dev = Entry(self, width=35)
        self.std_dev.pack(pady=5)
        Button(self, text="Провести фільтрацію сигналу", command=self.solve_problem).pack(pady=20)
        Button(self, text="Показати графічні результати", command=self.show_results).pack()

        self.label_time_for_default = Label(self, text="")
        self.label_time_for_parallel = Label(self, text="")
        self.label_faster_method = Label(self, text="")
        self.label_speedup = Label(self, text="")
        self.label_mse = Label(self, text="")
        self.label_mse_par = Label(self, text="")
        self.label_snr_before = Label(self, text="")
        self.label_snr_after = Label(self, text="")
        self.label_equal = Label(self, text="")

    def solve_problem(self):
        p = int(self.p_input.get())
        k = int(self.k_input.get())
        m = int(self.m_input.get())
        n = int(self.n_input.get())
        duration = float(self.duration_input.get())
        f = float(self.freq.get())
        std_dev = float(self.std_dev.get())
        signal_type = self.signal_chooser.get()
        noise_type = self.noise_chooser.get()
        rnd_noise = self.signal_rnd.get()
        is_lim_par = self.lim_par.get()

        f_arr = dfp.generate_weights_f(m)

        n = int(n * duration)

        clean_signal, t = dfp.generate_signal(n, duration, f, signal_type)
        noise = dfp.generate_noise(rnd_noise, n, noise_type, std_dev)

        noisy_signal = clean_signal + noise

        self.clean_signal = clean_signal
        self.noisy_signal = noisy_signal
        self.t = t

        dfpobj = dfp.DFP(p, noisy_signal)

        self.x_seq, time_for_seq = dfpobj.standard_sequential(k, m, f_arr)

        if is_lim_par:
            self.x_par, time_for_par = dfpobj.limited_parallel(noisy_signal, k, m, f_arr, p)
        else:
            self.x_par, time_for_par = dfpobj.standard_parallel(k, m, f_arr)

        time_seq_sec = time_for_seq / 1_000_000_000
        time_par_sec = time_for_par / 1_000_000_000

        mse, mse_par = dfp.find_mse(self.x_seq, self.x_par, self.clean_signal)

        snr_before = dfp.find_snr(self.clean_signal, self.noisy_signal)
        snr_after = dfp.find_snr(self.clean_signal, self.x_seq)

        is_res_equal = dfp.is_equal(self.x_seq, self.x_par)

        self.time_seq = time_seq_sec
        self.time_par = time_par_sec

        self.label_time_for_default.config(text=f'Час послідовного виконання: {time_seq_sec} с')
        self.label_time_for_default.pack(pady=5)
        self.label_time_for_parallel.config(text=f'Час паралельного виконання: {time_par_sec} с')
        self.label_time_for_parallel.pack(pady=5)
        self.label_speedup.config(text=f'Прискорення: {time_seq_sec / time_par_sec}')
        self.label_speedup.pack(pady=5)

        if time_seq_sec > time_par_sec:
            self.label_faster_method.config(text="Паралельне виконання швидше")
        else:
            self.label_faster_method.config(text="Послідовне виконання швидше")
        self.label_faster_method.pack(pady=5)

        self.label_mse.config(text=f'Середньоквадратична похибка послідовний алгоритм: {mse}')
        self.label_mse.pack(pady=5)
        self.label_mse_par.config(text=f'Середньоквадратична похибка паралельний алгоритм: {mse_par}')
        self.label_mse_par.pack(pady=5)
        self.label_snr_before.config(text=f'Співвідношення сигнал/шум до фільтрації: {snr_before}')
        self.label_snr_before.pack(pady=5)
        self.label_snr_after.config(text=f'Співвідношення сигнал/шум після фільтрації: {snr_after}')
        self.label_snr_after.pack(pady=5)
        res_equal_resp = "Так" if is_res_equal else "Ні"
        self.label_equal.config(text=f'Результати послідовного і паралельного виконання сходяться?\n {res_equal_resp}')
        self.label_equal.pack(pady=5)

    def show_results(self):
        dfp.show_results(self.clean_signal, self.noisy_signal, self.x_seq, self.x_par, self.t)


if __name__ == '__main__':
    root = MainWindow()
    root.mainloop()
