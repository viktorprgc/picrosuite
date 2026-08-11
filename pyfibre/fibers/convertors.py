import logging
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from skimage import draw
from skimage.morphology import remove_small_holes

from pyfibre.tools.convertors import binary_to_regions
from pyfibre.fibers.fibre_utilities import get_node_coord_array

logger = logging.getLogger(__name__)


def draw_network(network, label_image, index=1):
    nodes_coord = get_node_coord_array(network)
    label_image[nodes_coord[:, 0], nodes_coord[:, 1]] = index

    for edge in list(network.edges):
        start = list(network.nodes[edge[1]]["xy"])
        end = list(network.nodes[edge[0]]["xy"])
        line = draw.line(*(start + end))
        label_image[line] = index

    return label_image


def networks_to_binary(networks, shape, area_threshold=200, iterations=9, sigma=None):
    """Return a global binary representing areas of an image
    containing networks"""

    binary = np.zeros(shape, dtype=int)

    # Create skeleton image based on connected components in network
    for index, network in enumerate(networks):
        draw_network(network, binary, index=1)

    # Dilate skeleton image
    if iterations > 0:
        binary = binary_dilation(binary, iterations=iterations)

    # Smooth dilated image
    if sigma is not None:
        smoothed = gaussian_filter(binary.astype(float), sigma=sigma)
        # Convert float image back to binary
        binary = np.where(smoothed, 1, 0)

    # Remove smooth holes with area less than threshold
    binary = remove_small_holes(binary.astype(bool), area_threshold=area_threshold)

    return binary.astype(int)


def fibres_to_label_image(
    fibres,
    shape,
    area_threshold: int = 200,
    iterations: int = 9,
    sigma: float | None = 0.5,
) -> np.typing.NDArray:
    """Convert a list of Fibre objects into a label image where
    *each fibre is a distinct region* — no merging across fibres.

    Unlike ``networks_to_regions`` that draws all networks at index=1
    (causing overlapping dilated regions to collapse into fewer labels),
    this routine draws every fibre at a unique label, then dilates each
    fibre's skeleton **independently** so regions cannot merge.  An
    iterative Voronoi-like expansion resolves overlaps at branch-points
    without bias.

    Parameters
    ----------
    fibres : list of Fibre
        Individual fibre objects (e.g. from ``FibreAssigner.assign_fibres``).
    shape : tuple
        (rows, cols) of the output label image.
    area_threshold : int
        Minimum hole area filled inside each fibre region after dilation.
    iterations : int
        Number of 1-pixel dilation passes per fibre (total expansion radius).
    sigma : float or None
        Gaussian sigma applied to the *combined* label image to smooth
        region boundaries.  Pass ``None`` to skip.

    Returns
    -------
    label_image : np.ndarray (int, shape=shape)
        Label image where pixel value *k* (1-based) belongs to ``fibres[k-1]``.
        Background is 0.
    """
    # ------------------------------------------------------------------
    # Step 1 – draw each fibre skeleton at a **unique** label index
    # ------------------------------------------------------------------
    skeleton = np.zeros(shape, dtype=int)
    for idx, fibre in enumerate(fibres):
        draw_network(fibre.graph, skeleton, index=idx + 1)

    # ------------------------------------------------------------------
    # Step 2 – iterative per-label dilation (Voronoi-like expansion)
    # ------------------------------------------------------------------
    # Each outer iteration dilates *every* label by 1 pixel into
    # currently-unclaimed foreground.  Because all dilations are
    # computed from the *previous* label image (via a copy), the
    # expansion is fair and order-independent within each pass.
    label_image = skeleton.copy()
    for _ in range(iterations):
        previous = label_image.copy()
        for lbl in range(1, len(fibres) + 1):
            binary = previous == lbl
            if not binary.any():
                continue
            dilated = binary_dilation(binary)
            # Claim only pixels that no other label has claimed yet
            claim = dilated & (label_image == 0)
            label_image[claim] = lbl

    # ------------------------------------------------------------------
    # Step 3 – fill small holes inside each label
    # ------------------------------------------------------------------
    for lbl in range(1, len(fibres) + 1):
        binary = label_image == lbl
        if not binary.any():
            continue
        filled = remove_small_holes(binary, area_threshold=area_threshold)
        # Only assign newly-filled (hole) pixels — never steal from
        # another label or shrink the region
        new_fill = filled & ~binary
        label_image[new_fill & (label_image == 0)] = lbl

    # ------------------------------------------------------------------
    # Step 4 – optional boundary smoothing
    # ------------------------------------------------------------------
    if sigma is not None and sigma > 0:
        # Smooth each label independently to avoid cross-label bleeding
        smoothed = np.zeros(shape, dtype=float)
        for lbl in range(1, len(fibres) + 1):
            binary = (label_image == lbl).astype(float)
            if binary.sum() == 0:
                continue
            smoothed += lbl * gaussian_filter(binary, sigma=sigma)
        label_image = np.round(smoothed).astype(int)

    return label_image


def networks_to_regions(
    networks, image=None, shape=None, area_threshold=200, iterations=9, sigma=None
):
    """Transform fibre networks into a set of scikit-image segments"""

    # If no intensity image is provided, make sure binary
    # shape is provided
    if image is None:
        assert shape
    else:
        shape = image.shape

    binary = networks_to_binary(
        networks,
        shape,
        area_threshold=area_threshold,
        iterations=iterations,
        sigma=sigma,
    )

    regions = binary_to_regions(binary, intensity_image=image)

    return regions
