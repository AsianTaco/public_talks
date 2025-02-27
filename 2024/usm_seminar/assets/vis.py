import numpy as np
from scipy.fftpack import fftn, ifftn
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "text.usetex": True
})


def compute_power_spectrum(delta, Lbox, kmin=1e-3, kmax=1.0, Nk=32, normalize=True, return_k=False):
    """Compute a 3d power spectrum from density contrast
    Args:
        delta (np.array): if complex, it is expected to be the fourier representation directly, otherwise
             an FFT will be run first
        Lbox (tuple of float or float):
        kmin (float, optional): Defaults to 1e-3 h/Mpc
        kmax (float, optional): Defaults to 1.0 h/Mpc
        Nk (int, optional): Defaults to 32.
        logk (bool, optional): Defaults to False
              *WARNING* logk=True needs checking
        normalize (bool, optional): Defaults to True. Apply proper normalization of the spectrum.
        even (bool, optional): If delta is complex, one needs to know if the real representation has odd or even last dimension.
    """
    na = np.newaxis

    N1, N2, N3 = delta.shape
    N = N1, N2, N3

    Ntot = N1 * N2 * N3
    if type(Lbox) is float:
        Lbox = Lbox, Lbox, Lbox

    V = Lbox[0] * Lbox[1] * Lbox[2]

    ik = list(
        [np.fft.fftfreq(n, d=l / n) * 2 * np.pi for n, l in zip(N, Lbox)])

    ik[-1] = ik[-1][:(N[-1] // 2 + 1)]

    k_n = np.sqrt(ik[0][:, na, na] ** 2 + ik[1][na, :, na] ** 2 +
                  ik[2][na, na, :] ** 2)

    delta_hat = np.fft.rfftn(delta) * (V / Ntot)

    i_k_n = k_n
    i_k_min = kmin
    i_k_max = kmax

    # TODO: Handle nyquist correctly
    Hw, _ = np.histogram(i_k_n, range=(i_k_min, i_k_max), bins=Nk)
    H, b = np.histogram(i_k_n,
                        weights=delta_hat.real ** 2 + delta_hat.imag ** 2,
                        range=(i_k_min, i_k_max),
                        bins=Nk)

    H = H / Hw
    H = np.nan_to_num(H, nan=0)

    if normalize:
        H /= V

    if return_k:
        bc = 0.5 * (b[1:] + b[:-1])
        return bc, H
    else:
        return H


def generate_wave_numbers(n, physical_size):
    # Generate the frequency bins using np.fft.fftfreq
    freq = np.fft.fftfreq(n, d=physical_size) * n
    # Scale the frequencies by the physical size of the box
    k = 2 * np.pi * freq
    return k


def generate_gaussian_random_field(size, physical_size, k_values, p_values, seed=42):
    # Interpolate the provided power spectrum values
    power_spectrum_interp = interp1d(k_values, p_values, bounds_error=False, fill_value=0.0)

    # Create a 3D grid of wavenumbers
    kx = generate_wave_numbers(size[0], physical_size)
    ky = generate_wave_numbers(size[1], physical_size)
    kz = generate_wave_numbers(size[2], physical_size)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')

    # Compute the wavenumber magnitude
    k = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    # Avoid division by zero at the zero frequency
    k[0, 0, 0] = 1e-10

    # Compute the amplitude of the Fourier modes from the interpolated power spectrum
    amplitude = np.sqrt(power_spectrum_interp(k))

    # Generate random phases uniformly distributed on the unit circle in the complex plane
    np.random.seed(seed)
    random_phases = np.exp(2j * np.pi * np.random.rand(*size))

    # Impose the power spectrum
    fourier_field = amplitude * random_phases

    # Perform the inverse FFT to get the Gaussian random field
    real_field = np.real(ifftn(fourier_field))

    return real_field


# Example usage
grid_size = 128
grid = 3 * (grid_size,)
box_size = 1000
box = 3 * (box_size,)
kmin = 2 * np.pi / box_size
kmax = np.pi / (box_size / grid_size)

pk_sim = np.load("/home/sding/PhD/talks/2024/usm_seminar/quijote_low_res_pk_mean.npz")
k_values = pk_sim["k"]
pk_mean = pk_sim["pk"]

field = generate_gaussian_random_field(grid, box_size, k_values, pk_mean)

fig, field_ax = plt.subplots(figsize=(5, 5))
field_ax.imshow(field[42, :, :] * 8900, cmap='inferno')

field_ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

field_ax.tick_params(axis='both', length=0., which='both')
field_ax.xaxis.set_major_formatter(ticker.NullFormatter())
field_ax.xaxis.set_minor_formatter(ticker.NullFormatter())
field_ax.yaxis.set_major_formatter(ticker.NullFormatter())
field_ax.yaxis.set_minor_formatter(ticker.NullFormatter())

# fig.savefig('/home/sding/PhD/talks/2024/usm_seminar/assets/random_field.svg', format='svg', bbox_inches='tight',
#             pad_inches=0.02)

#####

pks_random = []
for i in range(500):
    field = generate_gaussian_random_field(grid, box_size, k_values, pk_mean, seed=i)
    pk_random = compute_power_spectrum(field, box, kmin=kmin, kmax=kmax, Nk=50, return_k=False)
    pks_random.append(pk_random)

random_pk_mean = np.array(pks_random).mean(axis=0)
scaling_factor = (pk_mean[1:-1] / random_pk_mean[1:-1]).mean()
print(scaling_factor)

power_fig, power_ax = plt.subplots(figsize=(5, 5))
power_ax.loglog(k_values, pk_mean, lw=2, label='Quijote simulation')
power_ax.loglog(k_values, scaling_factor * random_pk_mean, lw=2, ls='--', label='Random field')
power_ax.legend(prop={'size': 16})

power_ax.set_xlim(2e-2, 3e-1)
power_ax.set_ylim(3e2, 6e4)

power_ax.set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]", fontsize=16)
power_ax.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]", fontsize=16)

power_fig.savefig('/home/sding/PhD/talks/2024/usm_seminar/assets/pk_fields.svg', format='svg', bbox_inches='tight',
            pad_inches=0.02)
