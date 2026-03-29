import numpy as np
import lightkurve as lk
import h5py

kic_list = [
    757076, 892667, 1026032, 1161345, 1430163, 1571511, 1725815, 2010607,
    2162635, 2305372, 2436365, 2571868, 2718678, 2854693, 2987660, 3118883, 3241581, 3445671,
    3544595, 3656476, 3733346, 3858884, 3964109, 4141376, 4247791, 4351319, 4484238, 4758595,
    4914423, 5094751, 5182131, 5356349, 5520878, 5701829, 5858947, 6032730, 6185476, 6278762,
    6442183, 6603624, 6763132, 6936115, 7096821, 7269881, 7434955, 7591767, 7742534, 7871531,
    8179643, 8324268, 8462852, 8621637, 8751933, 8890170, 9025370, 9159301,
    9532219, 9655114, 9775454, 9897820, 10012345, 10135584, 10288502, 10417986, 10525077,
    10666592, 10797526, 10905692, 11013201, 11145139, 11253226, 11395018, 11613448, 11717120,
    11820830, 11904151, 12009504, 12106934, 12258514, 12317678, 12644769, 12736056,
    12980372, 13073592, 13191135, 13299529, 13397662, 13480232, 13589264, 13672047]


def build_valid_kic_catalog(kic_list):

    with h5py.File('data/kic_catalog.hdf5', 'a') as h5:

        key_list = list(h5.keys())

        for kic in kic_list:
            if kic not in key_list:
                srch = lk.search_lightcurve(f"KIC {kic}", mission="Kepler")

                if len(srch) > 0:
                    data = download_kepler_lc_stitched(srch)

                    h5.create_dataset(kic, data=data)


def download_kepler_lc_stitched(srch, min_points=1000):

    lcc = srch.download_all()
    if lcc is None or len(lcc) == 0:
        raise RuntimeError("Failed to download a usable stitched Kepler light curve.")

    lc = (lcc.stitch()
            .remove_nans()
            .remove_outliers(sigma=5)
            .normalize())

    if len(lc.flux) >= min_points:
        return lc
    return None


def find_random_kepler_lc(kic_list, max_tries=10, min_points=1000):
    """
    Pick a random KIC, download ALL available Kepler light curves, stitch into one,
    and return (kic, stitched_lc).

    Parameters
    ----------
    kic_list : list[int]
        List of KIC IDs to sample from.
    max_tries : int
        How many random KICs to try before failing.
    min_points : int
        Minimum number of cadences required after cleaning.

    Returns
    -------
    kic : int
    lc  : lk.LightCurve
        Cleaned, stitched Kepler light curve.
    """
    srch = -1

    for _ in range(max_tries):
        kic = int(np.random.choice(kic_list))
        srch = lk.search_lightcurve(f"KIC {kic}", mission="Kepler")

        if len(srch) == 0:
            raise RuntimeError("Failed to download a usable stitched Kepler light curve.")

    return srch


def extract_stellar_variability(lc, window_length=401):
    lc = lc.remove_nans().remove_outliers(sigma=5).normalize()  # important
    flat, trend = lc.flatten(window_length=window_length, return_trend=True)

    # stellar variability as fractional signal around 0
    stellar_var = (trend.flux / np.nanmedian(trend.flux)).value - 1.0
    return stellar_var, trend, flat


def random_chunk(arr, k):
    n = len(arr)
    if k > n:
        raise ValueError("k must be <= length of array")

    start = np.random.randint(0, n - k + 1)
    return arr[start:start + k]


def generate_noise(kic_dict, minimum_points=1000):

    kic_list = list(kic_dict.keys())

    rand_index = np.random.randint(0, len(kic_list))

    noise_chunk = random_chunk(kic_dict[kic_list[rand_index]], minimum_points)
    return noise_chunk, kic_list[rand_index]


