import numpy as np
import matplotlib.pyplot as plt

from functions.noise_functions import generate_noise, random_chunk
from functions.h5_functions import *

h5_path = "/Users/aaryanthusoo/Desktop/UCL/Research/Work/data/even_better_kic_noise.h5"

KIC_dict = load_kic_noise_dict(h5_path)

class LightCurve:
    """LightCurve object."""

    def __init__(self, kic=None, total_time=100, dt=29.4244):
        """
        Initialization of a simple light curve object.
        :param total_time: Total time of light curve occurance in days.
        :param dt: Preset to Kepler mission Long cadence time of ~29.4 minutes
        """

        dt = dt / (60 * 24)

        self.t = np.arange(0, total_time, dt)

        num_points = len(self.t)

        # Start with baseline flux = 1
        flux = np.ones(num_points)

        if kic is None:
            self.noise, self.kic = generate_noise(KIC_dict, num_points)

        else:
            self.noise = random_chunk(KIC_dict[kic], num_points)

        self.flux = flux * (1 + self.noise)

    def __str__(self):
        noise = lc.get_noise()

        std_noise, ptp_noise = np.std(noise) * 1e6, np.ptp(noise)* 1e6

        useful = True

        print("Cadence (min):", dt)

        # Check STD quality
        print("\nNoise std:",  std_noise, "ppm")
        # Standard Deviation
        if std_noise < 50:
            print("This star is not useful. Not enough stellar activity")
            useful = False
        elif std_noise < 300:
            print("This star is quiet. Enough stellar activity for Sun-like dwarfs")

        elif std_noise < 1000:
            print("This star is moderate. Enough stellar activity for Active Dwarfs or Young stars")

        elif std_noise < 5000:
            print("This star is active. Enough for strong spot modulation")

        else:
            print("This star is not useful. Too much stellar activity.")
            useful = False

        # Check PTP quality
        print("\nNoise peak-to-peak:", ptp_noise, "ppm")
        if ptp_noise < 200:
            print("This star is not useful. Peak to peak activity is too low")
            useful = False

        elif ptp_noise < 1800:
            print("This star is useful. Peak to peak activity is quiet")

        elif ptp_noise < 6000:
            print("This star is useful. Peak to peak activity is moderate")

        elif ptp_noise < 30000:
            print("This star is useful. Peak to peak activity is active")

        else:
            print("This star is not useful. Peak to peak activity is too active.")
            useful = False

        # Final usefulness of star
        if useful:
            return "\nThis light curve is useful."
        else:
            return "\nThis light curve is not useful."

    def get_flux(self):
        return self.flux

    def get_noise(self):
        return self.noise

    def plot_lc(self, title="Transit Light Curve"):
        plt.figure(figsize=(12, 4))
        plt.plot(self.t, self.flux, color="#000000", markersize=2, linestyle='-', linewidth=0.5, alpha=0.8)
        plt.scatter(self.t, self.flux, color="#000000", marker=".", s=2, alpha=0.8)
        plt.title(title)
        plt.xlabel("Time (days)")
        plt.ylabel("Normalized Flux")
        plt.show()

    def useful(self):
        std_useful, ptp_useful = True, True

        noise = self.get_noise()

        std_noise, ptp_noise = np.std(noise) * 1e6, np.ptp(noise) * 1e6

        # Standard Deviation
        if std_noise < 50 or std_noise > 5000:
            std_useful = False

        # Check PTP quality
        if ptp_noise < 200 or ptp_noise > 30000:
            ptp_useful = False

        return std_useful, ptp_useful, std_useful and ptp_useful


class TransitLightCurve(LightCurve):
    """Transit LightCurve object."""

    def __init__(self, kic=None, total_time=100, dt=29.4244):
        """Initialize LightCurve object."""

        super().__init__(kic, total_time, dt)

        # Realistic exoplanet transit parameters
        self.depth = np.random.uniform(2e-3, 1.5e-2)  # 0.2% to 1.5% (typical exoplanets)
        self.duration = np.random.uniform(0.2, 0.8)  # 5-20 hours in normalized time
        self.period = np.random.uniform(3, 20)  # 3-20 day orbital periods
        self.t0 = np.random.uniform(0, self.period)

        self.params = [self.t0, self.period, self.depth, self.duration]
        self.param_names = ['Initial Time', 'Period', 'Depth', 'Duration']

        self.flux = self.add_smooth_transit()

    def __str__(self):
        """Return string representation of LightCurve object."""
        str = "=================================================\n"
        str += "LightCurve Parameters:\n\n"
        str += (f"Initial Time: {self.t0} days\n" +
                f"Period:       {self.period} days\n"
                + f"Depth:        {self.depth} \n"
                + f"Duration:     {self.duration} days\n")
        str += "================================================="
        return str

    def add_smooth_transit(self):
        """
        Add a realistic exoplanet transit using a quadratic limb-darkening model.
        Creates smooth U-shaped dips like real Kepler data.
        """
        t, flux = self.t, self.flux
        t0, period, depth, duration = self.params

        # Phase relative to transit center
        phase = ((t - t0) % period)
        phase[phase > period / 2] -= period

        # Find points in transit window
        in_transit = np.abs(phase) < duration / 2

        if np.any(in_transit):
            # Normalized time within transit: -1 (start) to +1 (end)
            z = np.abs(phase[in_transit]) / (duration / 2)

            # Quadratic limb darkening approximation for smooth U-shape
            # This creates the characteristic smooth dip shape
            transit_flux = 1.0 - depth * np.maximum(0, 1 - z ** 2) ** 0.5
            flux[in_transit] = flux[in_transit] * transit_flux

        return flux

    def get_params(self):
        """Get parameters."""
        return self.params

    def get_flux(self):
        """Get flux."""
        return self.flux

    def plot_lc(self, title="Transit Light Curve"):
        plt.figure(figsize=(12, 4))
        plt.plot(self.t, self.flux, color="#000000", markersize=2, linestyle='-', linewidth=0.5, alpha=0.8)
        plt.scatter(self.t, self.flux, color="#000000", marker=".", s=4, alpha=0.8)
        plt.title(title)
        plt.xlabel("Time (days)")
        plt.ylabel("Normalized Flux")
        plt.show()

class EclipsingLightCurve(LightCurve):
    """Eclipsing LightCurve object."""

    def __init__(self, kic=None, total_time=100, dt=29.4244):
        """Initialize LightCurve object."""

        super().__init__(kic, total_time, dt)

        # Realistic binary star eclipse parameters
        self.depth1 = np.random.uniform(6e-3, 3.5e-2)  # 0.2% to 1.5% (typical exoplanets)
        self.duration1 = np.random.uniform(0.2, 0.8)  # 5-20 hours in normalized time

        self.depth2 = np.random.uniform(5e-3, 2.5e-2)
        self.duration2 = np.random.uniform(0.2, 0.8)

        self.period = np.random.uniform(3, 20)  # 3-20 day orbital periods
        self.t0_1 = np.random.uniform(0, self.period)
        self.t0_2 = self.t0_1 + (self.period / 2)

        self.flux = self.add_v_transit()

    def add_v_transit(self):

        t, flux = self.t, self.flux.copy()
        depth1, duration1, depth2, duration2 = self.depth1, self.duration1, self.depth2, self.duration2
        t0_1, t0_2 = self.t0_1, self.t0_2
        period = self.period

        # Phase relative to transit center
        for i in range(0, 2):
            if i==0:
                phase = ((t - t0_1) % period)
                phase[phase > period / 2] -= period

                # Find points in transit window
                in_transit = np.abs(phase) < duration1 / 2

                if np.any(in_transit):
                    # Normalized time within transit: -1 (start) to +1 (end)
                    z = np.abs(phase[in_transit]) / (duration1 / 2)

                    # This creates the v shape for clipsing binaries
                    transit_flux = 1.0 - depth1 * np.maximum(0, 1 - z) ** 0.5
                    flux[in_transit] = flux[in_transit] * transit_flux
            else:
                phase = ((t - t0_2) % period)
                phase[phase > period / 2] -= period
                # Find points in transit window

                in_transit = np.abs(phase) < duration1 / 2

                if np.any(in_transit):
                    # Normalized time within transit: -1 (start) to +1 (end)
                    z = np.abs(phase[in_transit]) / (duration2 / 2)

                    # This creates the v shape for clipsing binaries
                    transit_flux = 1.0 - depth2 * np.maximum(0, 1 - z) ** 0.5
                    flux[in_transit] = flux[in_transit] * transit_flux

        return flux


def test_train_lc(num_lc=200, lc_types=[0, 1], per_types=[0.5, 0.5], check=None):
    """

    0 - Normal Star Light Curve
    1 - Transit Light Curve
    2 - Eclipsing Light Curve

    :param num_lc:
    :param num_points:
    :param lc_types:
    :param per_type:
    :param check:
    :return:
    """

    if len(lc_types) != len(per_types):
        raise ValueError('Number of light curves types (lc_types) must match the ratio (per_types)')

    if np.abs((np.sum(per_types) - 1)) > 0.05:
        raise ValueError('Ratio of Light Curves must equal to 1')

    Titles = ["Light Curve", "Transit Curve", "Eclipsing Curve"]

    X_list, y_list = [], []
    for i in range(0, len(lc_types)):
        num_lc_type = round(num_lc * per_types[i])
        for j in range(0, num_lc_type):

            if lc_types[i] == 0:
                X_list.append(LightCurve().flux)
                y_list.append(0)
            elif lc_types[i] == 1:
                X_list.append(TransitLightCurve().flux)
                y_list.append(1)
            elif lc_types[i] == 2:
                X_list.append(EclipsingLightCurve().flux)
                y_list.append(2)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    if check is not None:
        t = np.arange(0, 4894)

        if check != 1:
            for i in lc_types:
                ii = np.where(y == i)[0]
                indexes = np.random.randint(0, len(ii), check)

                fig, axes = plt.subplots(nrows=check, ncols=1, sharex=True, sharey=True, figsize=(12, 4))
                fig.suptitle(Titles[i])
                for j, ax in enumerate(axes):
                    ax.plot(t, X[ii[indexes][j]])
                    if j==(check)//2:
                        ax.set_ylabel("Normalized Flux")

                axes[4].set_xlabel("Time (days)")
        else:
            for i in lc_types:
                ii = np.where(y == i)[0]
                indexes = np.random.randint(0, len(ii), check)

                plt.figure(figsize=(12, 4))
                plt.plot(t, X[ii[indexes][0]])
                plt.title(Titles[i])
                plt.xlabel("Time (days)")
                plt.ylabel("Normalized Flux")

        plt.show()


    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    num_lc = 7000
    lc_types = [0, 1, 2]
    per_types = [0.333, 0.333, 0.334]

    # First split: train/test
    X_train_full, X_test, y_train_full, y_test = test_train_lc(
        num_lc=num_lc,
        lc_types=lc_types,
        per_types=per_types,
        check=None
    )


