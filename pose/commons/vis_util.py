import os

import numpy as np
from cv2 import cv2
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
from mpl_toolkits.mplot3d import Axes3D

import prior_sk_data as psk_data

# limb_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]
limb_parents = psk_data.limb_parents

lcolor = 'red'
rcolor = 'blue'
mcolor = 'green'

color_array = [{-1: lcolor, 0: mcolor, 1: rcolor}[flag] for flag in psk_data.lr_flags]

plt_angles = [
    [-45, 10],
    [-135, 10],
    [-90, 10],
]

black = (0, 0, 0)
white = (1, 1, 1)


def draw_limbs_3d_plt(joints_3d, ax, limb_parents=limb_parents):
    for i in range(joints_3d.shape[0]):
        x_pair = [joints_3d[i, 0], joints_3d[limb_parents[i], 0]]
        y_pair = [joints_3d[i, 1], joints_3d[limb_parents[i], 1]]
        z_pair = [joints_3d[i, 2], joints_3d[limb_parents[i], 2]]
        color = color_array[i]
        ax.plot(x_pair, y_pair, zs=z_pair, c=color, linewidth=2, antialiased=True)
    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], c='black', s=10)


def get_sk_frame_figure(pred_3d, hmap_img, hmap_title):
    fig = plt.figure(frameon=False, figsize=(10, 10))
    for i, ang in enumerate(plt_angles):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.set_axis_off()
        ax.clear()
        ax.view_init(azim=ang[0], elev=ang[1])

        ax.set_xlim(-800, 800)
        ax.set_ylim(-800, 800)
        ax.set_zlim(-800, 800)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        draw_limbs_3d_plt(pred_3d * 100, ax)
    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(hmap_img)
    ax.set_title(hmap_title)
    return fig


def save_sk_frame_plt(video_project, pred_3d, img_id):
    hmap_img = cv2.imread(video_project.get_hmap_path(img_id))[:, :, [2, 1, 0]]
    hmap_title = 'Frame: %04d' % img_id
    fig = get_sk_frame_figure(pred_3d, hmap_img, hmap_title)
    fig.savefig(video_project.get_sk_frame_path(img_id))
    plt.close(fig)
    print('%s: Saved sk_frame for image %d' % (video_project.project_name, img_id))


def save_sk_frames_plt(video_project, start=0, end=None, translation=False):
    pred_3d = loadmat(video_project.get_pred_path())['pred_3d_fit']

    if translation:
        pelvis_position = video_project.get_preds()['pred_delta']
        pelvis_position[1] = pelvis_position[0]
        for i in xrange(1, pelvis_position.shape[0]):
            pelvis_position[i] += pelvis_position[i - 1]

    if end is None:
        end = len(pred_3d) - 1
    for i in xrange(start, end + 1):
        pred = pred_3d[i]
        if translation:
            pred[:, [0, 2]] += (pelvis_position[i] * .10)
        save_sk_frame_plt(video_project, pred, i)


def get_skeleton_plot(joints_3d, limb_parents=limb_parents, title=""):
    fig = plt.figure(frameon=False, figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(azim=90, elev=10)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_zlim(-8, 8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    draw_limbs_3d_plt(joints_3d, ax, limb_parents)
    plt.title(title)
    return fig


def plot_skeleton(joints_3d, limb_parents=limb_parents, title=""):
    fig = get_skeleton_plot(joints_3d, limb_parents, title)
    img = fig2data(fig)
    plt.close(fig)
    return img


def plot_skeleton_sequence(joints_seq, limb_parents=limb_parents, title=""):
    imgs = []
    for i, joints in enumerate(joints_seq):
        frame_title = "%s: Frame %4d" % (title, i)
        imgs.append(plot_skeleton(joints, limb_parents, title=frame_title))
    return np.array(imgs)


def plot_skeleton_in_axes(ax, joints_3d, limb_parents=limb_parents, title=""):
    ax.view_init(azim=90, elev=10)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_zlim(-8, 8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    draw_limbs_3d_plt(joints_3d, ax, limb_parents)


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


class Ax3DPose(object):
    def __init__(self, ax=None, skel=None, ground_size=8):
        # Start and endpoints of our representation

        if skel is None or skel.shape != (17, 3):
            skel = np.zeros((17, 3))

        self.ax = ax or plt.gca(projection='3d')
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        self.ax.set_xlim(-ground_size, ground_size)
        self.ax.set_ylim(-ground_size, ground_size)
        self.ax.set_zlim(-ground_size, ground_size)

        self.ax.set_aspect('equal')

        self.ax.view_init(azim=60, elev=30)

        C = [{-1: lcolor, 0: mcolor, 1: rcolor}[flag] for flag in psk_data.lr_flags]
        P = limb_parents

        # Make connection matrix
        self.joint_lines = []
        for i in range(skel.shape[0]):
            x = [skel[i, 0], skel[P[i], 0]]
            y = [skel[i, 1], skel[P[i], 1]]
            z = [skel[i, 2], skel[P[i], 2]]
            self.joint_lines.append(self.ax.plot(x, y, z, lw=2, c=C[i]))
        self.joint_points = self.ax.scatter(skel[:, 0], skel[:, 1], skel[:, 2], c='black', s=10)

    def update(self, skel):

        assert skel.shape == (17, 3), "channels should have 17x3 entries, it has %d instead" % skel.shape

        P = limb_parents

        for i in range(skel.shape[0]):
            x = [skel[i, 0], skel[P[i], 0]]
            y = [skel[i, 1], skel[P[i], 1]]
            z = [skel[i, 2], skel[P[i], 2]]
            self.joint_lines[i][0].set_xdata(x)
            self.joint_lines[i][0].set_ydata(y)
            self.joint_lines[i][0].set_3d_properties(z)

        self.joint_points._offsets3d = (skel[:, 0], skel[:, 1], skel[:, 2])

    def get_img(self):
        pass

def create_skeleton_grid(sk_batch, nrows, ncols):
    assert sk_batch.shape[0] <= nrows * ncols, 'Number skeletons exceeds the grid capacity'
    fig = plt.figure(figsize=(ncols * 2, nrows * 2))
    spec = GridSpec(nrows, ncols, figure=fig, hspace=0.0, wspace=0.0)
    axes = []
    for i in range(nrows):
        for j in range(ncols):
            axes.append(Ax3DPose(fig.add_subplot(spec[i, j], projection='3d')))
    for k in range(len(sk_batch)):
        axes[k].update(sk_batch[k])
    return fig

def create_video(skeleton_batch, frames_dir, vid_path, fps=25):
    """
    :param skeleton_batch:
    :param frames_dir:
    :param vid_path:
    :param fps:
    :return: HTML content to play this video on Jupyter Notebook
    >>> from IPython.display import HTML
    >>> html_content = create_video(sk_batch, frames_dir, vid_path, 25)
    >>> HTML(html_content)
    >>> '*** Embedded Video in Jupyter Cell Output ***'
    """
    if not vid_path.endswith('.mp4'):
        vid_path += '.mp4'
    ob = Ax3DPose()
    fname_format = os.path.join(frames_dir, 'frame-%03d.png')
    vid_dir = os.path.dirname(vid_path)
    os.system('mkdir -p {}'.format(frames_dir))
    os.system('mkdir -p {}'.format(vid_dir))
    for i, skeleton in enumerate(skeleton_batch):
        ob.update(skeleton)
        fname = fname_format % i
        plt.savefig(fname)
    make_video_cmd = "ffmpeg -y -r {fps} -i {filename_format} -vcodec libx264 -crf 0  -pix_fmt yuv720p {video_filename}".format(
        fps=fps,
        filename_format=fname_format,
        video_filename=vid_path
    )
    print(make_video_cmd)
    os.system(make_video_cmd)

    return (
        """
        <video width="320" height="240" controls>
          <source src="{}" type="video/mp4">
        </video>
        """.format(vid_path)
    )
