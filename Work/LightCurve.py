import numpy as np
import matplotlib.pyplot as plt

class LightCurve:
    """LightCurve object."""

    def __init__(self, time_length=30, n_points=2000):
        """Initialize LightCurve object."""
        self.t = np.linspace(0, time_length, n_points)

        # Start with baseline flux = 1
        flux = np.ones(n_points)
        noise_level = np.random.uniform(5e-4, 6e-4)

        white_noise = np.random.normal(0, noise_level, n_points)
        flux += white_noise

        self.flux = flux

    def plot_lc(self, title="Transit Light Curve"):
        plt.figure(figsize=(12, 4))
        plt.plot(self.t, self.flux, color="#000000", markersize=2, linestyle='-', linewidth=0.5, alpha=0.8)
        plt.scatter(self.t, self.flux, color="#000000", marker=".", s=2, alpha=0.8)
        plt.title(title)
        plt.xlabel("Time (days)")
        plt.ylabel("Normalized Flux")
        plt.show()

class TransitLightCurve(LightCurve):
    """Transit LightCurve object."""

    def __init__(self, time_length=30, n_points=2000):
        """Initialize LightCurve object."""

        super().__init__(time_length, n_points)

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

    def __init__(self, time_length=30, n_points=2000):
        """Initialize LightCurve object."""
        super().__init__(time_length, n_points)

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

        t, flux = self.t, self.flux
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

    def eclipses(self):
        self.flux = self.add_v_transit()
        self.flux -= 1

        self.flux = self.add_v_transit()
        return flux1 + flux2


def test_train_lc(num_lc=200, num_points=2000, lc_types=[0, 1], per_types=[0.5, 0.5], check=None):
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
                X_list.append(LightCurve(30, num_points).flux)
                y_list.append(0)
            elif lc_types[i] == 1:
                X_list.append(TransitLightCurve(30, num_points).flux)
                y_list.append(1)
            elif lc_types[i] == 2:
                X_list.append(EclipsingLightCurve(30, num_points).flux)
                y_list.append(2)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    if check is not None:
        t = np.arange(0, num_points) / 30

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
    X_train, X_test, y_train, y_test = test_train_lc(lc_types=[0, 1, 2], per_types=[0.3, 0.3, 0.4], check=1)