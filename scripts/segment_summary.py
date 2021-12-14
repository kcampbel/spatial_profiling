#!/Users/katiecampbell/miniconda2/envs/p36/bin/python

# Import libraries
import skimage.io as skio
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import Delaunay
from plotly import graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
import sys
import chart_studio
from functools import reduce
from itertools import combinations

# Inputs
marker_file = "mantis_update/mesmer_segmentation/PT0325-010313_ROI1/PT0325-010313_ROI1.tiff"
segmentation_file = "mantis_update/mesmer_segmentation/PT0325-010313_ROI1/wc_mask.tif"
nuclear_file = "mantis_update/mesmer_segmentation/PT0325-010313_ROI1/nuc_mask.tif"
out_prefix = "test_out"

# Get marker names from TIFF layers
marker_list = list()
with tifffile.TiffFile(marker_file) as tif:
    for index, page in enumerate(tif.pages):
        marker_list.append(page.tags['PageName'].value.split(' (', 1)[0])

# Read in TIFF files
marker_imstack = skio.imread_collection(marker_file, plugin = "tifffile")
segmentation_imstack = skio.imread(segmentation_file, plugin = "tifffile")
nuclear_imstack = skio.imread(nuclear_file, plugin = "tifffile")

# Create dictionary of all cell segments identified
# key 0 indicates that the corresponding pixel is not part of the cell segment
segmentation_dict = {}
for i in range(0, len(segmentation_imstack[0])):
    for j in range(0, len(segmentation_imstack[0][i])):
        if segmentation_imstack[0][i][j][0] in segmentation_dict.keys():
            segmentation_dict[segmentation_imstack[0][i][j][0]].append([i, j])
        else:
            segmentation_dict[segmentation_imstack[0][i][j][0]] = [[i, j]]

# Functions
# Get alphashape (outline pixels) of list of pixels
def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

# Get centroid position from list of positions (later used independently on x and y)
def centroid_pos(pts):
     _len = len(pts)
     _centroid = sum(pts) / _len
     return(_centroid)

# Retrieve marker intensities for list of pixels
def pixel_intensities(x, y):
    markers = list()
    for index, marker in enumerate(marker_list):
        markers.append(marker_imstack[index][x][y])
    return(pd.Series(markers))

# Get number of nonzero values
def npixels(series):
    return(series.astype(bool).sum(axis=0))

# Get combination of 2 for list input
def combn2(x):
    return(list(combinations(list(x), 2)))

# For each segment identified, do the following:
# Obtain all marker intensitites for each pixel
# Label pixels as nuclear (in nuclear TIFF), nonnuclear (not contained in nucleus), segment (all pixels in segment), and "outer" (segment alpha_shape +/- one pixel inevery direction
segment_data = []
for idx in segmentation_dict.keys():
    if idx == 0:
        pass
    else:
        # Get pixel coordinates in segment
        segment_pixels = np.array(segmentation_dict[int(idx)])
        # Use alpha_shape to obtain outline
        outline_edges = alpha_shape(segment_pixels, 0.95)
        outline_idx = {x for l in outline_edges for x in l}
        outline_pixels = segment_pixels[list(outline_idx)]
        # Extend outline by one pixel in every direction
        outline_expand = list()
        for pixel in outline_pixels:
            x_neighbor = [pixel[0], pixel[0] + 1, pixel[0], pixel[0] - 1]
            y_neighbor = [pixel[1] + 1, pixel[1], pixel[1] - 1, pixel[1]]
            neighbors = np.stack((x_neighbor, y_neighbor), axis=1)
            outline_expand.extend([list(i) for i in neighbors])
        outer_pixels = np.concatenate((outline_pixels, np.array(outline_expand)), axis = 0)
        # Get nuclear pixel locations
        nuclear = list()
        for pixel in segment_pixels:
            if nuclear_imstack[0][pixel[0]][pixel[1]] == 0:
                nuclear.append(None)
            else:
                nuclear.append(1)
        # Convert to pandas data frames for merging
        segment_pd = pd.DataFrame(segment_pixels, columns = ['x','y'])
        segment_pd['segment'] = 1
        segment_pd['nuclear'] = nuclear
        outline_pd = pd.DataFrame(outline_pixels, columns = ['x','y'])
        outline_pd['outline'] = 1
        outer_pd = pd.DataFrame(outer_pixels, columns = ['x','y']).drop_duplicates()
        outer_pd['outer'] = 1
        # Merge all data frames
        segment_pixels = reduce(lambda left, right: 
                               pd.merge(left, right, on=['x', 'y'], how='outer'),
                               [segment_pd, outline_pd, outer_pd])
        segment_pixels['segment_id'] = int(idx)
        # 
        marker_intensities = segment_pixels.apply(lambda row : pixel_intensities(int(row['x']), int(row['y'])), axis = 1)
        marker_intensities.columns = marker_list
        segment_detailed = pd.concat([segment_pixels, marker_intensities], axis = 1, sort = False) 
        segment_data.append(segment_detailed)

# Create huge data frame of all pixel information
agg_data = pd.concat(segment_data)

# Get marker intensities by region
segment_long = agg_data.query('segment == 1').melt(segment_pixels.columns, marker_intensities.columns, var_name='marker', value_name='intensity')
segment_summ = segment_long.groupby(['segment_id','marker']).agg({'intensity': ['sum','mean','median',npixels]}).reset_index()
segment_summ['segment_region'] = 'segment'
#
nuclear_long = agg_data.query('nuclear == 1').melt(segment_pixels.columns, marker_intensities.columns, var_name='marker', value_name='intensity')
nuclear_summ = nuclear_long.groupby(['segment_id','marker']).agg({'intensity': ['sum','mean','median',npixels]}).reset_index()
nuclear_summ['segment_region'] = 'nuclear'
#
nonnuclear_long = agg_data.query('nuclear != 1').melt(segment_pixels.columns, marker_intensities.columns, var_name='marker', value_name='intensity')
nonnuclear_summ = nonnuclear_long.groupby(['segment_id','marker']).agg({'intensity': ['sum','mean','median',npixels]}).reset_index()
nonnuclear_summ['segment_region'] = 'nonnuclear'
#
membrane_long = agg_data.query('outer == 1').melt(segment_pixels.columns, marker_intensities.columns, var_name='marker', value_name='intensity')
membrane_summ = membrane_long.groupby(['segment_id','marker']).agg({'intensity': ['sum','mean','median',npixels]}).reset_index()
membrane_summ['segment_region'] = 'membrane'

# Obtain segment features
segment_features = agg_data.groupby('segment_id').agg({"segment":"sum", "nuclear":"sum", "outer":"sum"})
segment_features['nuclear_pct'] = segment_features.nuclear*100/segment_features.segment
#
centroid_features = agg_data[agg_data.outline == 1].groupby('segment_id').agg({"x":centroid_pos, "y":centroid_pos})
# Create final_segment_features (with nuclear, outer, and segment sizes; nuclear_pct; centroid)
final_segment_features = pd.merge(segment_features, centroid_features, on=['segment_id'], how='outer')

# Find boundaries
# Find shared outer pixels
shared_outer_pixels = agg_data.query('outer == 1').groupby(['x','y']).agg({'outer':'sum', 'segment_id':'unique'}).query('outer>1').reset_index()
# Get number of pixels per segment boundary
segment_boundary_size = shared_outer_pixels.explode('segment_id').groupby('segment_id').size()
segment_boundary_size = pd.DataFrame(segment_boundary_size, columns=['shared_boundary'])
# Append shared_boundary pixels to final_segment_features
final_segment_features = pd.merge(final_segment_features, segment_boundary_size, on=['segment_id'], how='outer')
final_segment_features['boundary_pct'] = final_segment_features.shared_boundary*100/final_segment_features.outer
shared_outer_pixels['segment_pair'] = shared_outer_pixels['segment_id'].apply(combn2)
shared_outer_pixels = shared_outer_pixels.explode('segment_pair')
shared_outer_pixels[['segment_1','segment_2']] = pd.DataFrame(shared_outer_pixels['segment_pair'].tolist(), index=shared_outer_pixels.index)

# Get boundary features
# Calculate number of pixels per boundary
pixels_per_boundary = shared_outer_pixels.groupby(['segment_1', 'segment_2']).size()
pixels_per_boundary = pd.DataFrame(pixels_per_boundary, columns=['n_pixels']).reset_index()
pixels_per_boundary["tm_id"] = pixels_per_boundary["segment_1"].astype(str) + "_" + pixels_per_boundary["segment_2"].astype(str)
# Calculate number of neighbors per segment
n_neighbors = pixels_per_boundary.melt(['tm_id','n_pixels'], ['segment_1','segment_2'], var_name = "segment", value_name = "segment_id").groupby('segment_id').size()
n_neighbors = pd.DataFrame(n_neighbors, columns=['n_neighbors']).reset_index()
# Append n_neighbors to final_segment_features
final_segment_features = pd.merge(final_segment_features, n_neighbors, on=['segment_id'], how='outer')

# Quantify boundary intensities
pixel_ints = agg_data.drop(['segment','nuclear','outline','outer','segment_id'], axis=1).drop_duplicates()
boundaries_long = pd.merge(shared_outer_pixels, pixel_ints, on=['x','y'], how='left').melt(shared_outer_pixels.columns, marker_intensities.columns, var_name='marker', value_name='intensity')
boundaries_summ = boundaries_long.groupby(['segment_1','segment_2','marker']).agg({'intensity': ['sum','mean','median',npixels]}).reset_index()

# Replace all NAs with 0
final_segment_features = final_segment_features.fillna(0)

# Write to file
# Outputs to file
nuclear_summ.to_csv(out_prefix+".nuclear_intensities.tsv", sep = '\t', index = False)
nonnuclear_summ.to_csv(out_prefix+".nonnuclear_intensities.tsv", sep = '\t', index = False)
membrane_summ.to_csv(out_prefix+".membrane_intensities.tsv", sep = '\t', index = False)
segment_summ.to_csv(out_prefix+".segment_intensities.tsv", sep = '\t', index = False)
final_segment_features.to_csv(out_prefix+".segment_features.tsv", sep = '\t', index = False)
#
pixels_per_boundary.drop('tm_id', axis=1).to_csv(out_prefix+".boundary_features.tsv", sep = '\t', index = False)
boundaries_summ.to_csv(out_prefix+".boundary_intensities.tsv", sep = '\t', index = False)

