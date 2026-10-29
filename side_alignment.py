import numpy as np
import cv2

def stripe_filter_channel(channel, pattern_num=20, width1=1, width2=3):
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)

    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2

    fshift_masked = fshift.copy()

    for i in range((cols // pattern_num - 1) // 2):
        width_start = (i + 1) * pattern_num

        fshift_masked[
            crow - width1:crow + width1 + 1,
            ccol + width_start - width2:ccol + width_start + width2 + 1
        ] = 0

        fshift_masked[
            crow - width1:crow + width1 + 1,
            ccol - width_start - width2:ccol - width_start + width2 + 1
        ] = 0

    magnitude_spectrum_masked = 20 * np.log(np.abs(fshift_masked) + 1)
    magnitude_spectrum_masked = cv2.normalize(magnitude_spectrum_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
    return np.uint8(img_back), magnitude_spectrum_masked


def stripe_filter_bgr_image(bgr_img, pattern_num=20, width1=1, width2=3, show_result=False):
    b, g, r = cv2.split(bgr_img)

    b_filtered, b_spectrum = stripe_filter_channel(b, pattern_num, width1, width2)
    g_filtered, g_spectrum = stripe_filter_channel(g, pattern_num, width1, width2)
    r_filtered, r_spectrum = stripe_filter_channel(r, pattern_num, width1, width2)
    if show_result:
        cv2.imwrite("spectrum_vis.png", g_spectrum)

    filtered_bgr = cv2.merge([
        b_filtered,
        g_filtered,
        r_filtered
    ])

    return filtered_bgr

def stitch_horizontal_bgr(images, overlap):
    if len(images) == 0:
        return None

    images = [img.astype(np.float32) for img in images]

    h = images[0].shape[0]
    result = images[0][:, overlap:, :]

    for i in range(1, len(images)):

        img = images[i]
        left_overlap = img[:, :overlap, :]
        right_overlap = result[:, -overlap:, :]
        alpha = np.linspace(0, 1, overlap).reshape(1, overlap, 1)

        blended = (
            right_overlap * (1 - alpha) +
            left_overlap * alpha
        )

        result = result[:, :-overlap, :]

        remain = img[:, overlap:, :]

        result = np.concatenate(
            [result, blended, remain],
            axis=1
        )

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def stitch_and_filter(images,
                      overlap,
                      gray_thresh=240):
    stitched = stitch_horizontal_bgr(
        images,
        overlap
    )
    original_stitched = stitched.copy()

    gray = cv2.cvtColor(
        stitched,
        cv2.COLOR_BGR2GRAY
    )

    bg_mask = gray < gray_thresh
    fg_mask = ~bg_mask
    fg_pixels = stitched[fg_mask]

    if len(fg_pixels) == 0:
        return stitched

    filled_img = stitched.copy().astype(np.float32)
    if len(bg_mask) > 0:
        mean_bgr = fg_pixels.mean(axis=0)
        filled_img[bg_mask] = mean_bgr

    filtered_img = filled_img

    final_img = filtered_img.copy()
    if len(bg_mask) > 0:
        final_img[bg_mask] = original_stitched[bg_mask]

    return final_img

if __name__ == "__main__":
    sector_num = 15
    sector_width = 37
    frameCount = 98

    for sector in range(sector_num):
        img_list = []
        start_col = sector_width * sector
        for i in range(frameCount):
            fname = f"tool1/warp1/img_{i+1:03d}.png"
            img = cv2.imread(fname)
            img = img[:, start_col:start_col + sector_width * 2, :]
            if img is None:
                raise FileNotFoundError(fname)
            img_list.append(img)

        final_result = stitch_and_filter(img_list, sector_width, gray_thresh=10)
        if sector == 0:
            final_result = final_result[:, (sector_num - sector - 1)*sector_width:, :]
        else:
            final_result = final_result[:, (sector_num - sector - 1)*sector_width: -(sector)*sector_width, :]
        cv2.imwrite(f"tool1/stitch1/sector_{sector:02d}.png", final_result)
