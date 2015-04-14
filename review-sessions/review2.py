# EE 102B: Review Session, week 2
# Extended example: FIR DT approximation to RC lowpass filter
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
R = 3.0  # ohms
C = 1.0  # farads
def Hc(omega):
    return 1.0 / (1 + 1j*omega*R*C)

# Compute magnitude and phase of desired frequency response
omega = np.linspace(-2*np.pi, 2*np.pi, 200)
mag = np.absolute(Hc(omega))
phase = np.angle(Hc(omega))

# Plotting Code --YOU CAN IGNORE THIS--
plt.figure(1)
plt.clf()
plt.subplot(211)
plt.xlabel(r'$\omega$')
plt.plot(omega, mag)
plt.title(r"$|H_c(j\omega)|$")
plt.subplot(212)
plt.plot(omega, phase)
plt.title(r"$\angle H_c(j\omega)$")
plt.xlabel(r'$\omega$')
plt.show()

#%% 
# Compute the non-causal FIR filter approximation
# We get to choose T and M
T = R * C / 3
M = 10

# We're going to compute h_a[n] for each n from -M/2 to +M/2
# by numerically integrating over (capital) omega from -pi to pi.
ns = range(-M/2, M/2+1)
omega = np.linspace(-np.pi, np.pi, 200)
d_omega = omega[1] - omega[0]

ha = np.empty(M+1, dtype=complex)
for n in ns:
    integrand = Hc(omega/T) * np.exp(1j*n*omega)  # Note we divde omega by T for Hc
    ha[n + M/2] = 1.0/ (2 * np.pi) * np.trapz(integrand, dx=d_omega)  # in Matlab this is trapz(omega, integrand)


# Plotting Code --YOU CAN IGNORE THIS--
plt.figure(2)
plt.clf()
plt.stem(ns, np.real(ha), markerfmt='bo', linefmt='b--')
plt.stem(ns, np.imag(ha), markerfmt='ro', linefmt='r--')
plt.legend(['real','imaginary'])
plt.title(r"$h_a[n]$")
plt.show()

#%%
# Compute frequency response of approximate filter
def DTFT(ns, signal, omega):
    """ Compute the DTFT of a signal for a given set of frequencies.

    Args:
        ns: indices corresponding to values in signal
        signal: a discrete time signal x[n]
        omega: a vector of frequencies at which to evaluate the DTFT

    Returns:
        X(e^j\Omega) evaluated at \Omega = every value in |omega|
    """
    dtft = np.zeros(omega.shape, dtype=complex)
    for n, x in zip(ns, signal):
        dtft += x * np.exp(-1j * omega * n)

    return dtft

# Let's compute H_a for a few periods. Note: it should be periodic
# in Omega with period 2*pi.
omega = np.linspace(-2*np.pi, 2*np.pi, 400)
Ha = DTFT(ns, ha, omega)

plt.figure(3)
plt.clf()
plt.plot(omega, np.absolute(Ha))
plt.plot(omega, np.absolute(Hc(omega/T)))
plt.legend(['approximate','desired'])
plt.xlabel(r'$\Omega$')

# Compute error over one period
omega = np.linspace(-np.pi, np.pi, 200)
d_omega = omega[1] - omega[0]
Ha = DTFT(ns, ha, omega)
Hc = Hc(omega/T)
error = np.trapz(np.abs(Ha - Hc),dx=d_omega)
plt.text(np.pi, 0.8, "error = %0.3f" % error)
plt.show()