import cv2
import numpy as np
from scipy.optimize import curve_fit

def brightness_model_poly(phi, a0, a1, a2, a3, a4, a5=0):

    return a0 + a1 * phi + a2 * phi**2 + a3 * phi**3 + a4 * phi**4 + a5 * phi**5

dphi = 0.00028667 * 2

img_list = [f"tool1/warp5/img_{i:03d}.png" for i in range(98)]

profiles = []

for img_path in img_list[20:-20]:

    img = cv2.imread(img_path)
    if img is None:
        print(f"skip: {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    profiles.append(np.mean(img, axis=0))

profiles = np.asarray(profiles)
brightness_mean = np.mean(profiles, axis=0)
brightness_std = np.std(profiles, axis=0)
brightness = brightness_mean
W = brightness.shape[0]
cx = (W - 1) / 2
phi = (np.arange(W) - cx) * dphi

a0 = brightness.max() * 0.9
a1 = -a0 / (np.max(phi)**2)
a2 = 0.0
a3 = 0.0
a4 = 0.0
a5 = 0.0

p0 = [a0, a1, a2, a3, a4, a5]
popt, pcov = curve_fit(
    brightness_model_poly,
    phi,
    brightness,
    p0=p0,
    maxfev=20000
)
fit_curve = brightness_model_poly(phi, *popt)
disp_curve = brightness_model_poly(phi - np.deg2rad(1), *popt) + 2

rmse = np.sqrt(np.mean((brightness - fit_curve) ** 2))

print(f"a0      = {popt[0]:.4f}")
print(f"a1      = {popt[1]:.4f}")
print(f"a2      = {popt[2]:.4f}")
print(f"a3      = {popt[3]:.4f}")
print(f"a4      = {popt[4]:.4f}")
print(f"a5      = {popt[5]:.4f}")