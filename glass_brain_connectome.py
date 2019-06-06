#!/usr/bin/env Python
# @derekpisner @dpys
# 04/21/2019
# Glass brain connectome with FURY
import random
import itertools
import nibabel as nib
import numpy as np
import networkx as nx
import os
from nibabel.affines import apply_affine
from fury import actor, window, colormap, ui
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamline import select_by_rois
from dipy.tracking.utils import streamline_near_roi
from nilearn.plotting import find_parcellation_cut_coords
from nilearn.image import resample_to_img
from pynets.thresholding import normalize

streamlines_mni = '/Users/derekpisner/Downloads/test_subs/NKI_TRT/sub-0025427/ses-1/dwi/preproc/atlas_aal_8/streamlines_mni_csd_5000000_8mm_curv[60_30_10]_step[0.2_0.3_0.4_0.5]_warped.trk'
ch2bet_target = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
ch2bet =  '/Users/derekpisner/Applications/PyNets/pynets/templates/ch2better.nii.gz'
atlas = '/Users/derekpisner/Downloads/test_subs/NKI_TRT/sub-0025427/ses-1/dwi/preproc/registration/atlas_aal_8_t1w_mni.nii.gz'
conn_matrix_path = '/Users/derekpisner/Downloads/test_subs/NKI_TRT/sub-0025427/ses-1/dwi/preproc/atlas_aal_8/127_est_csd_unthresh_mat.npy'
interactive = False

# Instantiate scene
r = window.Renderer()

# Set camera
r.set_camera(position=(-176.42, 118.52, 128.20),
                 focal_point=(113.30, 128.31, 76.56),
                 view_up=(0.18, 0.00, 0.98))

# Load atlas rois
atlas_img = nib.load(atlas)
atlas_img_data = atlas_img.get_data()

# Collapse list of connected streamlines for visualization
streamlines = nib.streamlines.load(streamlines_mni).streamlines
parcels = []
i = 0
for roi in np.unique(atlas_img_data)[1:]:
    parcels.append(atlas_img_data==roi)
    i = i + 1

# Add streamlines as cloud of 'white-matter'
streamlines_actor = actor.line(streamlines, colormap.create_colormap(np.ones([len(streamlines)]), name='Greys_r', auto=True), lod_points=10000, depth_cue=True, linewidth=0.2, fake_tube=True, opacity=1.0)
r.add(streamlines_actor)

# Creat palette of roi colors and add them to the scene as faint contours
roi_colors = np.random.rand(int(np.max(atlas_img_data)), 3)
parcel_contours = []
i = 0
for roi in np.unique(atlas_img_data)[1:]:
    include_roi_coords = np.array(np.where(atlas_img_data==roi)).T
    x_include_roi_coords = apply_affine(np.eye(4), include_roi_coords)
    bool_list = []
    for sl in streamlines:
        bool_list.append(streamline_near_roi(sl, x_include_roi_coords, tol=1.0, mode='either_end'))
    if sum(bool_list) > 0:
        print('ROI: ' + str(i))
        parcel_contours.append(actor.contour_from_roi(atlas_img_data==roi, color=roi_colors[i], opacity=0.2))
    else:
        pass
    i = i + 1

for vol_actor in parcel_contours:
    r.add(vol_actor)

# Get voxel coordinates of parcels and add them as 3d spherical centroid nodes
[coords, label_names] = find_parcellation_cut_coords(atlas_img, background_label=0, return_label_names=True)
def mmToVox(nib_nifti, mmcoords):
    return nib.affines.apply_affine(np.linalg.inv(nib_nifti.affine), mmcoords)

coords_vox = []
for i in coords:
    coords_vox.append(mmToVox(atlas_img, i))
coords_vox = list(set(list(tuple(x) for x in coords_vox)))

# Build an edge list of 3d lines
conn_matrix = np.load(conn_matrix_path)
conn_matrix = normalize(conn_matrix)
G = nx.from_numpy_array(conn_matrix)
for i in G.nodes():
    nx.set_node_attributes(G, {i: coords_vox[i]}, label_names[i])

G.remove_nodes_from(list(nx.isolates(G)))
G_filt = nx.Graph()
fedges = filter(lambda x: G.degree()[x[0]] > 0 and G.degree()[x[1]] > 0, G.edges())
G_filt.add_edges_from(fedges)

coord_nodes = []
for i in range(len(G.edges())):
    edge = list(G.edges())[i]
    [x, y] = edge
    x_coord = list(G.nodes[x].values())[0]
    x_label = list(G.nodes[x].keys())[0]
    l_x = actor.label(text=str(x_label), pos=x_coord, scale=(1, 1, 1), color=(50,50,50))
    r.add(l_x)
    y_coord = list(G.nodes[y].values())[0]
    y_label = list(G.nodes[y].keys())[0]
    l_y = actor.label(text=str(y_label), pos=y_coord, scale=(1, 1, 1), color=(50,50,50))
    r.add(l_y)
    coord_nodes.append(x_coord)
    coord_nodes.append(y_coord)
    c = actor.line([(x_coord, y_coord)], window.colors.coral, linewidth=10*(float(G.get_edge_data(x, y)['weight']))**(1/2))
    r.add(c)

point_actor = actor.point(list(set(coord_nodes)), window.colors.grey, point_radius=0.75)
r.add(point_actor)

# Load glass brain template and resample to MNI152_2mm brain
template_img = nib.load(ch2bet)
template_target_img = nib.load(ch2bet_target)
res_brain_img = resample_to_img(template_img, template_target_img)
template_img_data = res_brain_img.get_data().astype('bool')
template_actor = actor.contour_from_roi(template_img_data, color=(50, 50, 50), opacity=0.05)
r.add(template_actor)

# Show scene
if interactive is True:
    window.show(r)
else:
    fig_path = os.path.dirname(streamlines_mni) + '/3d_connectome_fig.png'
    window.record(r, out_path=fig_path, size=(600, 600))
