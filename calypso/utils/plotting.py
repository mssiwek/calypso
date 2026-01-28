import numpy as np
def make_time_grid(norb, nbins):
	"""
	Create a time grid based on the number of orbits and number of bins.

	Parameters
	----------
	norb : int
		Number of orbits.
	nbins : int
		Number of bins.

	Returns
	-------
	np.ndarray
		Time grid array.
	"""
	T_orbit = 1.0  # Assuming normalized orbital period
	total_time = norb * T_orbit
	time_grid = np.linspace(0, total_time, nbins)
	return time_grid