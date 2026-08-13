def mask_hue(
    h,
    sigma_hue=2,
    hue_range=(0, 0.5),
    ):
    """
    Gets rid of blue hued pixels in the polarized image.
    Blue sits on the high end of the HSV hue channel,
    so just exclude it completely by only getting 0-0.5
    """

    h_smoothed = ski.exposure.rescale_intensity(
        ndi.gaussian_filter(h, sigma=sigma_hue),
        in_range=hue_range
    )

    h_mask = h_smoothed < ski.filters.threshold_otsu(h_smoothed)
    h_mask = ski.morphology.remove_small_objects(h_mask)

    return h_mask


def denoise(
    img,
    p_range=(1, 99),
    params={"patch_size": 5,
            "patch_distance": 12,
            "fast_mode": True,
            }
    ):
    """
    Denoise image using denoise_nl_means
    """

    params["sigma"] = ski.restoration.estimate_sigma(img)

    p1, p2 = np.percentile(img, p_range)

    img_dn = ski.restoration.denoise_nl_means(
        ski.exposure.equalize_adapthist(
            ski.exposure.rescale_intensity(
                img,
                in_range=(p1, p2),
                out_range=(p1, p2)
            )
        ),
        **params
    )

    return img_dn

def dist_trans(
    img,
    sigma=1,
    scale=0.5
):
    """
    Make distance transform of the image
    """

    img_r = ski.transform.rescale(
        img,
        scale,
        mode='constant',
        anti_aliasing=None
    )

    sigma *= scale

    pol_tb = ski.filters.sato(
        img_r,
        sigmas=range(1,5),
        black_ridges=False,
        mode='constant'
    )

    low = ski.filters.threshold_li(pol_tb)
    high = ski.filters.threshold.isodata(pol_tb)

    pol_thresh = ski.morphology.remove_small_objects(
        ski.filters.apply_hysteresis_threshold(
            pol_tb, low, high
        )
    )

    pol_edt = ndi.gaussian_filter(
        ndi.distance_transform_edt(pol_thresh),
        sigma=sigma
    )

    return pol_edt

def make_tissue_mask(
    img,
    scale=0.5,
    sigma=1,
    ent_disk_r=(5,3),
    ent_nbins=32,
    ent_prominence=0.1,
    ent_dist=15,
    ):

    img_gray = ski.transform.rescale(
        ndi.gaussian_filter(
            ski.color.rgb2gray(img),
            sigma=sigma),
        scale,
        mode='constant',
        anti_aliasing=None
    )

    ent = ski.filters.rank_entropy(
        img_gray,
        ski.morphology.disk(ent_disk_r[1])
    )

    ent_count, ent_bins = np.histogram(ent, ent_nbins)
    ent_bins = ent_bins[1:]

    min_idx = sig.argrelextrema(ent_count, np.less)
    min_idx = np.array(*min_idx)

    peak_idx, peak_props = sig.find_peaks(
        ent_count,
        prominence=ent_prominence,
        distance=ent_distance
    )

    low, high = [
        ski.measure.shannon_entropy(img_gray, base=i)
        for i in [32, 4]
    ]

    min_x, min_y = [], []

    for idx in min_idx:
        x = ent_bins[idx]
        y = ent_count[idx]

        min_x.append(x)
        min_y.append(y)

    peak_x, peak_y = [], []

    for idx in peak_idx:
        x = ent_bins[idx]
        y = ent_count[idx]

        peak_x.append(x)
        peak_y.append(y)

    for binned in min_x:
        temp_thresh = binned
        if temp_thresh > low and temp_thresh < high:
            thresh_local = temp_thresh

    tissue_mask = ent > thresh_local
    tissue_mask = ndi.binary_erosion(
        tissue_mask,
        structure=ndi.iterate_structure(
            ndi.generate_binary_structure(2,1)
        )
    )
    tissue_mask = ndi.binary_fill_holes(tissue_mask)

    return tissue_mask







