import jax.numpy as np
import jax.scipy as jsp

import zodiax as zdx
import optax

import dLux as dl

npix = 64
diam = 2

aperture = dl.layers.CircularAperture(radius=diam/2, transformation=dl.CoordTransform(), occulting=True, normalise=True)

optics = dl.AngularOpticalSystem(npix, diam, [("aperture",aperture)], 64, 50e-3, 1)

wavelengths = np.asarray([1.2e-6])
weights = np.asarray([1])

source = dl.PointSource(wavelengths=wavelengths)

#telescope = zdx.set_array(dl.Telescope(optics, ("binary",source)), "aperture.radius")

telescope = dl.Telescope(optics, ("binary", source))

data = telescope.model()

path = "aperture.radius"

@zdx.filter_value_and_grad(path)
def loss_fn(model, data):
    out = model.model()
    return -np.sum(jsp.stats.poisson.logpmf(data, out))

loss, grads = loss_fn(telescope, data)

print(grads.get(path))

print(grads)

#optim, opt_state = zdx.get_optimiser(zdx.set_array(telescope,"aperture.radius"), path, optax.adam(1))
optim, opt_state = zdx.get_optimiser(telescope, [path], [optax.adam(1e-5)])


losses, models = [], []
for i in range(10):
    # Calculate the loss gradients, and update the model
    loss, grads = loss_fn(telescope, data)
    updates, opt_state = optim.update(grads, opt_state)
    telescope = zdx.apply_updates(telescope, updates)

    # save results
    models.append(telescope)
    losses.append(loss)

    #  jtu.tree_map(lambda a, b: a * b, telescope, grads)
