import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_dilation
from skimage.transform import resize


def reconstruct_height_from_xgradient(
        grad_x,
        vertical_weight=0.01,
        reg_lambda=2e-6,
        laplacian_weight=0.0
        ):

    H, W = grad_x.shape

    # FFT
    P = np.fft.fft2(grad_x)
    wx = 2 * np.pi * np.fft.fftfreq(W)
    wy = 2 * np.pi * np.fft.fftfreq(H)

    WX, WY = np.meshgrid(wx, wy)
    denom = WX**2 + vertical_weight * WY**2 + reg_lambda + laplacian_weight*(WX**2 + WY**2)**2
    denom[0, 0] = 1.0
    Z_fft = (-1j * WX * P) / denom
    Z_fft[0, 0] = 0
    height = np.real(np.fft.ifft2(Z_fft))

    return height

def suppress_mask_area(img, mask, ksize=5, stretch=30):
    img = img.astype(np.float32)
    dilate_mask = binary_dilation(mask, iterations=5)
    H, W = img.shape
    compressed_img = resize(img, (H // stretch, W))
    
    out = median_filter(
        compressed_img,
        size=(ksize, ksize),
        mode='reflect'
    )
    out = resize(out, (H, W))
    out = np.where(dilate_mask, out, img)
    return out  

if __name__ == "__main__":
    a0 = 207.9776
    a1 = 109.2414
    a2 = -4950.6671
    a3 = -10329.0707
    a4 = 54801.7878
    a5 = 239825.3354
    
    radius = 0.5
    windowScale = 0.015
    pixelToMM = 0.0002846
    resample_pixelToMM = 0.00028667
    scanWidth = np.sqrt(radius) * windowScale
    scanAngle = 2 * np.arcsin(scanWidth / 2.0 / radius)
    sector_half_num = 8
    img_list = [f"tool1/stitch1/sector_{sector:02d}.png" for sector in range(sector_half_num * 2 - 1)]
    phi_list = np.linspace(-scanAngle*(sector_half_num-1), scanAngle*(sector_half_num-1), sector_half_num * 2 - 1)
    print("phi_list:", phi_list * 180 / np.pi, "degrees")

    delta_gray = []
    for img_path in img_list:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        correct_img = img - np.mean(img)
        delta_gray.append(correct_img)

    # (9,H,W)
    delta_gray = np.stack(delta_gray, axis=0).astype(np.float32)

    _, H, W = delta_gray.shape

    s = -(a1
        + 2 * a2 * phi_list
        + 3 * a3 * phi_list**2
        + 4 * a4 * phi_list**3
        + 5 * a5 * phi_list**4).astype(np.float32)

    X = np.column_stack([
        s,
        np.ones_like(s)
    ]).astype(np.float32)

    # (2,9)
    X_pinv = np.linalg.inv(X.T @ X) @ X.T

    # (9,N)
    Y = delta_gray.reshape(len(s), -1)

    # (2,N)
    coef = X_pinv @ Y

    theta = coef[0].reshape(H, W)
    shift = coef[1].reshape(H, W)
    pred = (X @ coef).reshape(len(s), H, W)
    residual = delta_gray - pred
    residual_rms = np.sqrt(
        np.mean(residual**2, axis=0)
    )

    print("residual rms:",
        residual_rms.mean())
    
    mask = residual_rms > 10

    theta = suppress_mask_area(theta, mask)
    slope = np.tan(theta) * resample_pixelToMM * 1000
    height_map = reconstruct_height_from_xgradient(slope, vertical_weight=0.01, reg_lambda=2e-6, laplacian_weight=0.0)
    