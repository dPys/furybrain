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
import vtk
from nibabel.affines import apply_affine
from fury import actor, window, colormap, ui
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamline import select_by_rois
from dipy.tracking.utils import streamline_near_roi
from dipy.data import get_sphere
from nilearn.plotting import find_parcellation_cut_coords
from nilearn.image import resample_to_img
import pandas as pd
from collections import namedtuple
from math import sqrt
from pynets.stats.netstats import modularity_louvain_und_sign
from fury.utils import set_input
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_agg import RendererAgg
from vtk.util import numpy_support

def normalize(W, copy=True):
    """form pynets"""
    if copy:
        W = W.copy()
    W /= np.max(np.abs(W))
    return W

def euqli_dist(p, q, squared=False):
    # Calculates the euclidean distance, the "ordinary" distance between two
    # points
    #
    # The standard Euclidean distance can be squared in order to place
    # progressively greater weight on objects that are farther apart. This
    # frequently used in optimization problems in which distances only have
    # to be compared.
    if squared:
        return ((p[0] - q[0]) ** 2) + ((p[1] - q[1]) ** 2)
    else:
        return sqrt(((p[0] - q[0]) ** 2) + ((p[1] - q[1]) ** 2))

def closest(cur_pos, positions):
    low_dist = float('inf')
    closest_pos = None
    for pos in positions:
        dist = euqli_dist(cur_pos,pos)
        if dist < low_dist:
            low_dist = dist
            closest_pos = pos
    return closest_pos


def win_callback(obj, event):
    if win_callback.win_size != obj.GetSize():
        size_old = win_callback.win_size
        win_callback.win_size = obj.GetSize()
        size_change = [win_callback.win_size[0] - size_old[0], 0]
        win_callback.panel.re_align(size_change)

def get_neighborhood(coords, r, vox_dims, dims):
    """

    :param coords:
    :param r:
    :param vox_dims:
    :param dims:
    :return:
    """
    # Adapted from Neurosynth
    # Return all points within r mm of coords. Generates a cube and then discards all points outside sphere. Only
    # returns values that fall within the dimensions of the image."""
    r = float(r)
    xx, yy, zz = [slice(-r / vox_dims[i], r / vox_dims[i] + 0.01, 1) for i in range(len(coords))]
    cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
    sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** .5 <= r]
    sphere = np.round(sphere.T + coords)
    return sphere[(np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1), :].astype(int)

def point_left_click_callback(obj, ev):
    """Get the value of the clicked voxel and show it in the panel."""
    event_pos = point_left_click_callback.show_m.iren.GetEventPosition()
    showm = point_left_click_callback.show_m
    scene = point_left_click_callback.show_m.scene
    picker = point_left_click_callback.picker
    nb_sphere_vertices = point_left_click_callback.nb_sphere_vertices

    panel = ui.Panel2D(size=(550, 300),
                       position=(20, 900),
                       color=(1, 1, 1),
                       opacity=0.1,
                       align="right")

    picker.Pick(event_pos[0],
                event_pos[1],
                0,
                scene)

    point_id = picker.GetPointId()
    if point_id < 0:
        print("invalid picking")
        return

    error=4
    inds = [(int(round(coord[0], 0)), int(round(coord[1], 0)), int(round(coord[2], 0))) for coord in get_neighborhood(picker.GetPickPosition(), error, zooms, dims)]

    point_candidates = []
    for point in inds:
        if point in coord_nodes_int:
            point_candidates.append(point)

    if len(point_candidates) > 1:
        coord_pick = closest(picker.GetPickPosition(), point_candidates)
    else:
        try:
            coord_pick = point_candidates[0]
        except:
            pass

    try:
        node_ix = coord_node_mappings[coord_node(x=coord_pick[0], y=coord_pick[1], z=coord_pick[2])]
        print(node_ix)
        met_list = get_mets_by_pointid(node_ix, G)
    except:
        pass

    try:
        print(stringify_met_list(met_list))
    except:
        pass

    try:
        comm_members = get_mets_by_pointid_comm(node_ix, G)
        sphere = get_sphere('symmetric362')
        comm_coords = []
        for comm_mem in comm_members:
            comm_coords.append(node_coord_mappings[comm_mem])
        point_actor = actor.sphere(list(set(comm_coords)), window.colors.royal_blue,
                                   radii=1.25, vertices=sphere.vertices, faces=sphere.faces)

        scene.add(point_actor)
        showm.render()
        def timer_callback(_obj, _event):
            scene.RemoveActor(point_actor)
            showm.render()

        showm.add_timer_callback(False, 100, timer_callback)
    except:
        pass

    return

def stringify_met_list(met_list):
    met_lister = list(met_list.items())
    met_str = 'Label: ' + str(met_lister[0][0]) + '\nCoordinate: ' + str(met_lister[0][1]) + '\n'
    for i in met_lister[2:-1]:
        met_str = met_str + '\n' + ' '.join(i[0].split('_')[1:]) + ': ' + str(round(i[1][0], 3))

    return met_str


def visible_callback(checkbox):
    show_streamlines = True if "streamlines" in checkbox.checked else False
    show_brain = True if "brain" in checkbox.checked else False
    show_parcel_contours = True if "surfaces" in checkbox.checked else False

    visible_callback.streamlines_actor.SetVisibility(show_streamlines)
    for parcel in visible_callback.parcel_contours:
        parcel.SetVisibility(show_parcel_contours)
    visible_callback.brain_actor.SetVisibility(show_brain)

def figure_to_data(figure):
  """
  @brief Convert a Matplotlib figure to a numpy 2D array with RGBA uint8 channels and return it.
  @param figure A matplotlib figure.
  @return A numpy 2D array of RGBA values.
  """
  # Draw the renderer
  import matplotlib
  figure.tight_layout()
  figure.canvas.draw()

  # Get the RGBA buffer from the figure
  w, h = figure.canvas.get_width_height()
  buf = np.fromstring(figure.canvas.tostring_argb(), dtype=np.uint8)
  buf.shape = (h, w, 4)

  # canvas.tostring_argb gives pixmap in ARGB mode. Roll the alpha channel to have it in RGBA mode
  buf = np.roll(buf, 3, axis=2)

  return buf

def figure_to_image(figure):
  """
  @brief Convert a Matplotlib figure to a vtkImageData with RGBA unsigned char channels
  @param figure A matplotlib figure.
  @return a vtkImageData with the Matplotlib figure content
  """
  buf = figure_to_data(figure)

  # Flip rows to be suitable for vtkImageData.
  buf = buf[::-1,:,:].copy()

  return numpy_to_image(buf)

def numpy_to_image(numpy_array):
  """
  @brief Convert a numpy 2D or 3D array to a vtkImageData object
  @param numpy_array 2D or 3D numpy array containing image data
  @return vtkImageData with the numpy_array content
  """

  shape = numpy_array.shape
  if len(shape) < 2:
    raise Exception('numpy array must have dimensionality of at least 2')

  h, w = shape[0], shape[1]
  c = 1
  if len(shape) == 3:
    c = shape[2]

  # Reshape 2D image to 1D array suitable for conversion to a
  # vtkArray with numpy_support.numpy_to_vtk()
  linear_array = np.reshape(numpy_array, (w*h, c))

  vtk_array = numpy_support.numpy_to_vtk(linear_array)

  image = vtk.vtkImageData()
  image.SetDimensions(w, h, 1)
  image.AllocateScalars(vtk_array.GetDataType(), 4)
  image.GetPointData().GetScalars().DeepCopy(vtk_array)

  return image

def mmToVox(nib_nifti, mmcoords):
    return nib.affines.apply_affine(np.linalg.inv(nib_nifti.affine), mmcoords)

def plot_conn_mat(conn_matrix, label_names):
    import matplotlib as mpl
    import seaborn as sns
    from matplotlib import colors
    from matplotlib import pyplot as plt
    from nilearn.plotting import plot_matrix
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'

    colors.Normalize(vmin=0, vmax=1)
    clust_pal = mpl.colors.ListedColormap(sns.color_palette("RdBu_r", 40))

    rois_num = conn_matrix.shape[0]
    if rois_num < 100:
        try:
            fig = plot_matrix(conn_matrix, figure=(10, 10), labels=label_names, vmax=1, vmin=0, reorder=True,
                        auto_fit=True, grid=False, colorbar=False, cmap=clust_pal)
        except RuntimeWarning:
            print('Connectivity matrix too sparse for plotting...')
    else:
        try:
            fig = plot_matrix(conn_matrix, figure=(10, 10), vmax=1, vmin=0, auto_fit=True, grid=False,
                        colorbar=False, cmap=clust_pal)
        except RuntimeWarning:
            print('Connectivity matrix too sparse for plotting...')

    cur_axes = plt.gca()
    for spine in cur_axes.axes.spines.values():
        spine.set_visible(False)
    fig.figure.tight_layout(pad=0)
    fig.figure.patch.set_facecolor('black')
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])
    cur_axes.axes.get_xaxis().set_ticklabels([])
    cur_axes.axes.get_yaxis().set_ticklabels([])
    cur_axes.axes.get_xaxis().set_major_locator(plt.NullLocator())
    cur_axes.axes.get_yaxis().set_major_locator(plt.NullLocator())
    cur_axes.set_frame_on(False)
    cur_axes.margins(0,0, tight=True)
    cur_axes.set_axis_off()

    return fig

def get_mets_by_pointid(pointid, G):
    met_list = G.node[pointid]
    return met_list

def get_mets_by_pointid_comm(pointid, G):
    comm_ix = G.node[pointid]['community_id']
    comm_ix = [ix for ix in G.nodes() if G.node[ix]['community_id']]
    return comm_ix

if __name__ == '__main__':
    DATA_DIR = '/Users/derekpisner/Applications/examples_bak'
    streamlines_mni = os.path.join(DATA_DIR, 'streamlines_mni_csd_5000000_8mm_curv[60_30_10]_step[0.2_0.3_0.4_0.5]_warped.trk')
    ch2bet_target = os.path.join(DATA_DIR, 'MNI152_T1_2mm_brain_mask.nii.gz')
    ch2bet = os.path.join(DATA_DIR, 'ch2better.nii.gz')
    atlas = os.path.join(DATA_DIR, 'atlas_aal_8_t1w_mni.nii.gz')
    graph_properties = os.path.join(DATA_DIR, '127_net_metrics_csd_1.0_8mm_neat.csv')
    conn_matrix_path = os.path.join(DATA_DIR, '127_est_csd_unthresh_mat.npy')
    interactive = True

    # Instantiate scene
    scene = window.Scene()
    current_size = (1000, 1000)
    show_manager = window.ShowManager(scene=scene, size=current_size,
                                      title="Network Visualization")
    show_manager.initialize()

    panel = ui.Panel2D(size=(200, 200),
                       position=(750, 20),
                       color=(1, 1, 1),
                       opacity=0.1,
                       align="right")
    show_checkbox = ui.Checkbox(labels=["brain", "surfaces", "streamlines"])
    show_checkbox.on_change = visible_callback
    panel.add_element(show_checkbox, coords=(0.1, 0.333))
    scene.add(panel)

    win_callback.win_size = current_size
    win_callback.panel = panel

    # Load atlas rois
    atlas_img = nib.load(atlas)
    dims = atlas_img.shape
    zooms = atlas_img.get_header().get_zooms()
    atlas_img_data = atlas_img.get_data()

    # Collapse list of connected streamlines for visualization
    streamlines = nib.streamlines.load(streamlines_mni).streamlines
    parcels = [atlas_img_data==roi for roi in np.unique(atlas_img_data)[1:]]

    # Add streamlines as cloud of 'white-matter'
    streamlines_actor = actor.line(streamlines, colormap.create_colormap(np.ones([len(streamlines)]), name='Greys_r', auto=True), lod_points=10000, depth_cue=True, linewidth=0.2, fake_tube=True, opacity=1.0)
    scene.add(streamlines_actor)
    visible_callback.streamlines_actor = streamlines_actor

    # Creat palette of roi colors and add them to the scene as faint contours
    roi_colors = np.random.rand(int(np.max(atlas_img_data)), 3)
    parcel_contours = []

    # i = 0
    # for roi in np.unique(atlas_img_data)[1:]:
    #     include_roi_coords = np.array(np.where(atlas_img_data==roi)).T
    #     x_include_roi_coords = apply_affine(np.eye(4), include_roi_coords)
    #     bool_list = []
    #     for sl in streamlines:
    #         bool_list.append(streamline_near_roi(sl, x_include_roi_coords, tol=1.0, mode='either_end'))
    #     if sum(bool_list) > 0:
    #         print('ROI: ' + str(i))
    #         parcel_contours.append(actor.contour_from_roi(atlas_img_data==roi, color=roi_colors[i], opacity=0.2))
    #     else:
    #         pass
    #     i = i + 1

    visible_callback.parcel_contours = parcel_contours
    # for vol_actor in parcel_contours:
    #     # vol_actor.AddObserver('LeftButtonPressEvent',
    #     #                       point_left_click_callback,
    #     #                       1.0)
    #     scene.add(vol_actor)

    # Load glass brain template and resample to MNI152_2mm brain
    template_img = nib.load(ch2bet)
    template_target_img = nib.load(ch2bet_target)
    res_brain_img = resample_to_img(template_img, template_target_img)
    template_img_data = res_brain_img.get_data().astype('bool')
    template_actor = actor.contour_from_roi(template_img_data, color=(50, 50, 50), opacity=0.05)
    scene.add(template_actor)
    visible_callback.brain_actor = template_actor

    # Get voxel coordinates of parcels and add them as 3d spherical centroid nodes
    [coords, label_names] = find_parcellation_cut_coords(atlas_img, background_label=0, return_label_names=True)

    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(atlas_img, i))
    coords_vox = list(set(list(tuple(x) for x in coords_vox)))

    # Build an edge list of 3d lines
    df = pd.read_csv(graph_properties)
    node_cols = [s for s in list(df.columns) if isinstance(s, int) or any(c.isdigit() for c in s)]

    conn_matrix = np.load(conn_matrix_path)
    conn_matrix = normalize(conn_matrix)
    G = nx.from_numpy_array(conn_matrix)

    # Add adj. mat
    fig = plot_conn_mat(conn_matrix, label_names)
    fig.figure.set_size_inches(0.17,0.17)
    fig.figure.set_dpi(500)
    data = figure_to_image(fig.figure)

    plt_actor = set_input(vtk.vtkImageActor(), data)
    scene.add(plt_actor)

    # Get communities
    gamma = 1
    [node_comm_aff_mat, q] = modularity_louvain_und_sign(conn_matrix, gamma=float(gamma * 0.0000001))

    coord_node_mappings = {}
    node_coord_mappings = {}
    for i in G.nodes():
        node_props = [k for k in node_cols if str(i) == k.split('_')[0]]
        coord = tuple([round(j, 0) for j in coords_vox[i]])
        coord_node = namedtuple("coord", ["x", "y", "z"])
        coord_node_mappings[coord_node(x=coord[0], y=coord[1], z=coord[2])] = i
        node_coord_mappings[i] = coord
        nx.set_node_attributes(G, {i: coords_vox[i]}, label_names[i])
        for node_prop in node_props:
            G.node[i][node_prop] = df[node_prop]
        G.node[i]['community_id'] = node_comm_aff_mat[i]

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
        scene.add(l_x)
        y_coord = list(G.nodes[y].values())[0]
        y_label = list(G.nodes[y].keys())[0]
        l_y = actor.label(text=str(y_label), pos=y_coord, scale=(1, 1, 1), color=(50, 50, 50))
        scene.add(l_y)
        coord_nodes.append(x_coord)
        coord_nodes.append(y_coord)
        c = actor.line([(x_coord, y_coord)], window.colors.coral, linewidth=10*(float(G.get_edge_data(x, y)['weight']))**(1/2))
        scene.add(c)

    coord_nodes_int = [(int(round(coord[0], 0)), int(round(coord[1], 0)), int(round(coord[2], 0))) for coord in coord_nodes]

    sphere = get_sphere('symmetric362')
    point_actor = actor.sphere(list(set(coord_nodes)), window.colors.grey,
                               radii=0.75, vertices=sphere.vertices, faces=sphere.faces)
    point_left_click_callback.show_m = show_manager
    point_left_click_callback.graph = G
    point_left_click_callback.label_names = label_names
    point_left_click_callback.picker = vtk.vtkPointPicker()
    point_left_click_callback.picker.SetTolerance(0.025)
    point_left_click_callback.nb_sphere_vertices = len(sphere.vertices)

    point_actor.AddObserver('LeftButtonPressEvent',
                            point_left_click_callback,
                            1.0)
    scene.add(point_actor)

    # Show scene
    if interactive is True:
        # window.show(scene)
        show_manager.add_window_callback(win_callback)
        show_manager.start()
    else:
        fig_path = os.path.dirname(streamlines_mni) + '/3d_connectome_fig.png'
        window.record(scene, out_path=fig_path, size=(600, 600))
