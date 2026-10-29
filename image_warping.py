import cv2
import numpy as np


def build_cone_remap(nominal_radius, actual_radius, theta, pixelToMM, pixelToMM_warp, scanAngle, sector_half_num, warping_height, center_pixel, img_width):

    warping_half_angle = sector_half_num * scanAngle

    out_h = int(np.round(warping_height / pixelToMM_warp))
    out_w = int(np.round(2 * warping_half_angle * nominal_radius / pixelToMM_warp))

    h_mm = np.arange(out_h, dtype=np.float32) * pixelToMM_warp

    phi = (
        np.arange(out_w, dtype=np.float32)
        * pixelToMM_warp / nominal_radius
        - warping_half_angle
    )

    PHI, H = np.meshgrid(phi, h_mm)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    X = (actual_radius - H * sin_theta) * np.sin(PHI)

    Y = (
        H * cos_theta * cos_theta
        + actual_radius * sin_theta * (1.0 - np.cos(PHI))
        + H * sin_theta * sin_theta * np.cos(PHI)
    )

    map_x = (X / pixelToMM + (img_width-1) / 2 + center_pixel).astype(np.float32)
    map_y = (Y / pixelToMM).astype(np.float32)

    return map_x, map_y


def cone_unwarp(img, nominal_radius, actual_radius, theta, pixelToMM, pixelToMM_warp, scanAngle,
                sector_half_num, warping_height, center_pixel):

    img_height, img_width = img.shape[:2]
    map_x, map_y = build_cone_remap(
        nominal_radius, actual_radius, theta, pixelToMM, pixelToMM_warp, scanAngle,
        sector_half_num, warping_height, center_pixel, img_width
    )

    unwrapped = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return unwrapped


def draw_cone_boundary(img, r, theta, pixelToMM, pixelToMM_warp, scanAngle,
                       sector_half_num, warping_height, center_pixel,
                       color=(0, 255, 0), thickness=4):

    vis = img.copy()
    warping_half_angle = (sector_half_num-0.5) * scanAngle
    warping_center_col = (img.shape[1] - 1) / 2 + center_pixel

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    n = 1000

    def project(phi, h):
        x = (r - h * sin_theta) * np.sin(phi)
        y = (
            h * cos_theta * cos_theta
            + r * sin_theta * (1 - np.cos(phi))
            + h * sin_theta * sin_theta * np.cos(phi)
        )
        u = x / pixelToMM + warping_center_col
        v = y / pixelToMM
        return np.column_stack([
            np.round(u).astype(np.int32),
            np.round(v).astype(np.int32)
        ])

    phi_range = np.linspace(
        -warping_half_angle,
        warping_half_angle,
        n
    )

    curve1 = project(phi_range, np.zeros_like(phi_range))
    curve2 = project(phi_range, np.full_like(phi_range, warping_height))

    h = np.linspace(0, warping_height, n)
    # curve3 = project(np.full_like(h, -warping_half_angle), h)
    # curve4 = project(np.full_like(h, warping_half_angle), h)

    cv2.polylines(vis, [curve1], False, color, thickness)
    cv2.polylines(vis, [curve2], False, color, thickness)
    # cv2.polylines(vis, [curve3], False, color, thickness)
    # cv2.polylines(vis, [curve4], False, color, thickness)

    split_angles = [
        (i+0.5) * scanAngle
        for i in range(-sector_half_num, sector_half_num)
    ]
    for phi_split in split_angles:
        curve_split = project(np.full_like(h, phi_split), h)
        cv2.polylines(vis, [curve_split], False, color, thickness)

    return vis


if __name__ == "__main__":
    nominal_radius = 0.5
    img_width = 2688
    windowScale = 0.015
    pixelToMM = 0.0002846
    
    center_pixel = -2.89
    phi_lim = 0.253616

    scanWidth = np.sqrt(nominal_radius) * windowScale
    scanAngle = 2 * np.arcsin(scanWidth / 2.0 / nominal_radius)
    img_num = 98

    pixels_per_sector = scanAngle * nominal_radius // pixelToMM
    print(f"pixels_per_sector = {pixels_per_sector}")
    pixelToMM_warp = scanAngle * nominal_radius / pixels_per_sector
    print(f"pixelToMM_warp = {pixelToMM_warp}")

    sector_half_num = int(phi_lim * 0.7 // scanAngle)
    print(f"sector_half_num = {sector_half_num}")
    warping_height = 1512 * pixelToMM

    theta = np.deg2rad(0)
    actual_radius = np.linspace(0.5, 0.5, img_num)
    for i in range(1, 1 + img_num):
        img_path = f"tool1/raw5/img ({i}).png"
        img = cv2.imread(img_path)
        if img is None:
            print(f"skip: {img_path}")
            continue

        vis = cone_unwarp(
            img, nominal_radius, actual_radius[i-1], theta, pixelToMM, pixelToMM_warp, scanAngle,
            sector_half_num, warping_height, center_pixel
        )
        cv2.imwrite(f"tool1/warp1/img_{i:03d}.png", vis)

