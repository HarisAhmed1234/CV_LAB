import numpy as np
import pywt
import matplotlib.pyplot as plt

# Anomaly detection in sensor data using wavelet transform
np.random.seed(42)
n = 1000
t = np.arange(n)
signal = np.random.normal(0, 1, n)
signal[[150, 400, 650, 900]] += [6, -6, 7, -7]  # injected anomalies

# Wavelet decomposition and soft thresholding of detail coefficients
wavelet, level = "db4", 4
coeffs = pywt.wavedec(signal, wavelet, level=level)
sigma = np.median(np.abs(coeffs[-1])) / 0.6745
thr = sigma * np.sqrt(2 * np.log(n))
coeffs_t = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
denoised = pywt.waverec(coeffs_t, wavelet)[:n]

# Residuals; points beyond 2 standard deviations are flagged
residual = signal - denoised
limit = 2 * np.std(residual)
anomalies = np.where(np.abs(residual) > limit)[0]
print("Anomalies detected:", len(anomalies))

fig, ax = plt.subplots(2, 1, figsize=(12, 7))
ax[0].plot(t, signal, label="Sensor Data")
ax[0].plot(t, denoised, "--", label="Denoised Signal")
ax[0].set_title("Sensor Data and Denoised Signal")
ax[0].legend(loc="lower right")

ax[1].plot(t, residual, "r", label="Residuals")
ax[1].scatter(anomalies, residual[anomalies], c="g", s=20, label="Anomalies", zorder=3)
ax[1].set_title("Residuals and Detected Anomalies")
ax[1].legend(loc="lower right")

plt.tight_layout()
plt.show()
