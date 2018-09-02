import os
import matplotlib

matplotlib.use('Agg')

import imageio
from matplotlib import pyplot as plt

import utils
from model_componets import *
from model import loss_x_recon, loss_z_recon, loss_cyclic, loss_gen_adv

from commons.video_project import VideoProject

from data_loader import get_data_loader

data_loader = get_data_loader('val')

from model_utils import PoseModel

results_dir = 'results_test'


model = PoseModel()

def recon_path(dir_name='reconstructed'):
    return os.path.join(results_dir, dir_name)


def gif_path():
    return os.path.join(results_dir, 'walks')


def create_recon_dirs(dir_name):
    r_path = recon_path(dir_name)
    os.system('mkdir -p %s/' % r_path)
    os.system('mkdir -p %s/good' % r_path)
    os.system('mkdir -p %s/bad' % r_path)
    os.system('mkdir -p %s/median' % r_path)
    os.system('mkdir -p %s/random' % r_path)
    return r_path


def save_gif(imgs, gif_name):
    imageio.mimsave('{}/{}'.format(gif_path(), gif_name), imgs)


def get_random_z_batch(batch_size):
    x_batch = data_loader.get_test_data_batch(batch_size, norm=True)
    z_ = model.encode(x_batch)
    z_batch = np.zeros(z_.shape)
    for i in range(z_batch.shape[0]):
        s = np.random.randint(0, z_.shape[0])
        e = np.random.randint(0, z_.shape[0])
        while s == e:
            e = np.random.randint(0, z_.shape[0])
        k = np.random.rand()
        z_batch[i] = z_[s] + k * (z_[e] - z_[s])
    return z_batch


num_steps = 50


def get_elbow_index(A):
    slope = lambda i: (A[i + 1] - A[i - 1]) / 2.0

    n = A.shape[0]

    slope_th = 1.0 / n

    l, h = 1, n - 2

    while l < h:
        m = (l + h) // 2
        if slope(m) < slope_th:
            l = m + 1
        else:
            h = m
    return l


def get_tp_value(A, p=0.90):
    return A[int(p * (A.shape[0]) - 1)]


def save_cherry_picked_samples(x_recons_bundle, perm, num_samples, r_path):
    print('Saving %d Good Samples...' % num_samples)

    for i in range(num_samples):
        x_recons = [x_recons[perm[i]] for x_recons in x_recons_bundle]
        fig = utils.compare_multiple_local_skeletons(x_recons)
        plt.savefig('%s/good/compare_fig_%03d' % (r_path, i))
        plt.close(fig)

    # Bad Samples
    print('Saving %d Bad Samples...' % num_samples)
    for i in range(num_samples):
        x_recons = [x_recons[perm[-i]] for x_recons in x_recons_bundle]
        fig = utils.compare_multiple_local_skeletons(x_recons)
        plt.savefig('%s/bad/compare_fig_%03d' % (r_path, i))
        plt.close(fig)

    print('Saving %d Median Samples...' % (2 * num_samples))
    for i in range(-num_samples, num_samples):
        m = len(perm) // 2
        x_recons = [x_recons[perm[m + i]] for x_recons in x_recons_bundle]
        fig = utils.compare_multiple_local_skeletons(x_recons)
        plt.savefig('%s/median/compare_fig_%03d' % (r_path, i))
        plt.close(fig)

    # Random Samples
    print('Saving %d Random Samples...' % num_samples)
    for i in range(num_samples):
        x_recons = [x_recons[i] for x_recons in x_recons_bundle]
        fig = utils.compare_multiple_local_skeletons(x_recons)
        plt.savefig('%s/random/compare_fig_%03d' % (r_path, i))
        plt.close(fig)


def save_sorted_loss_plot(sorted_losses, path):
    n = 1.0 * len(sorted_losses)
    i_elbow = get_elbow_index(sorted_losses)

    elbow_tile = (i_elbow / n * 100)
    elbow_tile_m = ((i_elbow - 1) / n * 100)
    elbow_tile_p = ((i_elbow + 1) / n * 100)

    plt.rcParams["figure.figsize"] = 15, 15
    x_scale = list(100 * np.arange(1.0, n + 1.0) / n)

    fig = plt.figure()
    fig.suptitle('Val Losses: %s' % utils.get_exp_name(), fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    ax.plot(x_scale, sorted_losses)

    ax.plot([elbow_tile], [sorted_losses[i_elbow]], 'o')
    ax.plot([elbow_tile_m], [sorted_losses[i_elbow - 1]], 'o')
    ax.plot([elbow_tile_p], [sorted_losses[i_elbow + 1]], 'o')

    font_size = 14

    ax.set_xlabel('%-tile')
    ax.set_ylabel('Loss')

    ax.text(5, .950, '# Examples: %d' % int(n), fontsize=font_size)
    ax.text(5, .925, 'Elbow Index : %d' % i_elbow, fontsize=font_size)
    ax.text(5, .900, 'Elbow %%tile: %.4f' % elbow_tile, fontsize=font_size)

    ax.text(5, .850, 'Min Loss: %.4f' % sorted_losses[0], fontsize=font_size)
    ax.text(5, .825, 'Max Loss: %.4f' % sorted_losses[-1], fontsize=font_size)
    ax.text(5, .800, 'Loss at elbow index: %.4f' % sorted_losses[i_elbow], fontsize=font_size)

    for i, p in enumerate(np.arange(0.8, 1, 0.02)):
        text = 'LP %5.2f %.4f' % (100 * p, get_tp_value(sorted_losses, p))
        ax.text(5, 0.750 - i * 0.02, text, fontsize=font_size)

    ax.axis([0, 105, 0, 1])

    plt.savefig(path)
    plt.close()


def compare_z_reconstructions(z_batch, num_samples=10, dir_name='z_reconstructed'):
    z_recon, z_loss = model.reconstruct_z(z_batch, get_loss=True)
    perm = sorted(range(z_batch.shape[0]), key=lambda i: z_loss[i])
    print('least loss: ', perm[:num_samples])
    print('most loss: ', perm[-num_samples:])

    r_path = create_recon_dirs(dir_name)

    x_batch = model.decode(z_batch)
    x_recon = model.decode(z_recon)

    save_cherry_picked_samples((x_batch, x_recon), perm, num_samples, r_path)

    print('Plotting Sorted Losses...')
    sorted_losses = z_loss[perm]
    save_sorted_loss_plot(sorted_losses, '%s/sorted_losses.png' % r_path)


def compare_x_reconstructions(x_batch, num_samples=10, dir_name='x_reconstructed', num_recon=4):
    x_recons_bundle, x_loss_bundle = [x_batch], [np.zeros((x_batch.shape[0],))]
    for i_level in range(num_recon):
        print 'Reconstructing Level: %d' % (i_level + 1)
        x_recon, x_loss = model.reconstruct_x(x_recons_bundle[-1], get_loss=True)
        x_recons_bundle.append(x_recon)
        x_loss_bundle.append(x_loss_bundle[-1] + x_loss)

    lv1_x_loss = x_loss_bundle[1]

    perm = sorted(range(x_batch.shape[0]), key=lambda i: lv1_x_loss[i])
    print('least loss: ', perm[:num_samples])
    print('most loss: ', perm[-num_samples:])

    r_path = create_recon_dirs(dir_name)

    save_cherry_picked_samples(x_recons_bundle, perm, num_samples, r_path)

    print('Plotting Sorted Losses...')
    sorted_losses = lv1_x_loss[perm]
    save_sorted_loss_plot(sorted_losses, '%s/sorted_losses.png' % r_path)


def evaluate_losses(n_iter=100, batch_size=128):
    loss_types = [loss_x_recon, loss_z_recon, loss_cyclic, loss_gen_adv]
    sum_losses = np.zeros((len(loss_types),))
    num_batches = data_loader.get_num_batches('test', batch_size)
    for i in range(n_iter):
        batch_id = np.random.randint(0, num_batches)
        x_batch = data_loader.get_test_data_batch(batch_size, batch_index=batch_id, norm=True)
        z_batch = get_random_z_batch(x_batch.shape[0])
        losses = model.get_losses_batch(x_batch, z_batch, loss_types)
        sum_losses += losses
        if i % 5 == 0:
            print('Evaluation done for all: %d' % (i + 1))
    return sum_losses / n_iter


tasks = {
    'walks',
    # 'recons',
    'x_recons',
    # 'gif_recons'
    'z_recons',
    'eval',
    'zdist',
}

if len(tasks.intersection({'recons', 'x_recons', 'z_recons', 'gif_recons'})) > 0:
    batch_size = data_loader.num_test_examples

    ## Visualizing Reconstructions
    if len(tasks.intersection({'recons', 'x_recons'})) > 0:
        print('\nComparing X Reconstructions...')
        x_batch = data_loader.get_test_data_batch(batch_size, norm=True)
        compare_x_reconstructions(x_batch, num_samples=25)

    if len(tasks.intersection({'recons', 'z_recons'})) > 0:
        print('\nComparing Z Reconstructions...')
        z_batch = get_random_z_batch(batch_size)
        compare_z_reconstructions(z_batch, num_samples=25)

    if len(tasks.intersection({'recons', 'gif_recons'})) > 0:
        print('\nComparing Gif Reconstructions...')
        dance_id = 100
        vp = VideoProject('dance_%d.mp4' % dance_id)

        x_batch = utils.unit_norm(vp.get_preds()['pred_17_local'], axis=2)[1000:2000]
        x_recon = model.reconstruct_x(x_batch)

        print('Converting %d Frame...' % len(x_batch))

        x_batch_skeletons = [utils.plot_local_skeleton(x) for x in x_batch]
        x_recon_skeletons = [utils.plot_local_skeleton(x) for x in x_recon]

        save_gif(x_batch_skeletons, 'sk_orign_dance_%d.gif' % dance_id)
        save_gif(x_recon_skeletons, 'sk_recon_dance_%d.gif' % dance_id)

if 'walks' in tasks:
    ## Creating Traversals Visualizations

    print('\nVisualizing Walks')

    z_start = np.ones((num_zdim,)) * -1
    z_end = np.ones((num_zdim,))
    os.system('mkdir -p %s' % gif_path())
    imgs = model.generate_sk_seq_from_z(z_start, z_end, num_steps)
    save_gif(imgs, 'diagonal_1.gif')

    for i in range(5):
        z_start = np.random.uniform(-1, 1, (num_zdim))
        z_end = np.random.uniform(-1, 1, (num_zdim))
        imgs = model.generate_sk_seq_from_z(z_start, z_end, num_steps)
        save_gif(imgs, 'uniform_z_walk_%d.gif' % i)

        z_start = np.clip(np.random.normal(0, 0.27, (num_zdim)), -1, 1)
        z_end = np.clip(np.random.normal(0, 0.27, (num_zdim)), -1, 1)
        imgs = model.generate_sk_seq_from_z(z_start, z_end, num_steps)
        save_gif(imgs, 'normal_z_walk_%d.gif' % i)

        z_start = np.random.randint(0, 2, (num_zdim)) * 2 - 1
        z_end = -z_start
        imgs = model.generate_sk_seq_from_z(z_start, z_end, num_steps)
        save_gif(imgs, 'diagonal_random_%d.gif' % i)

        x_batch = data_loader.get_test_data_batch(1024, 0, norm=True)

        x_start = x_batch[np.random.randint(0, x_batch.shape[0])]
        x_end = x_batch[np.random.randint(0, x_batch.shape[0])]
        imgs = model.generate_sk_seq_from_x(x_start, x_end, num_steps)
        save_gif(imgs, 'random_x_walk_%d.gif' % i)

if 'eval' in tasks:
    ## Evaluating Losses
    loss_x, loss_z, loss_c, loss_g = evaluate_losses(n_iter=20)

    exp_name = utils.get_exp_name()

    print '\n Losses:'
    print exp_name, 'X_recon_loss: %.05f iteration: %d' % (loss_x, model.iter_number)
    print exp_name, 'Z_recon_loss: %.05f iteration: %d' % (loss_z, model.iter_number)
    print exp_name, 'C_recon_loss: %.05f iteration: %d' % (loss_c, model.iter_number)
    print exp_name, 'A_gener_loss: %.05f iteration: %d' % (loss_g, model.iter_number)
    print ''

if 'zdist' in tasks:
    print('\nPlotting distribution for Z')
    z_dist_batch_size = 1024
    x_batch = data_loader.get_test_data_batch(z_dist_batch_size, 0, shuffle=True, norm=True)
    z_ = model.encode(x_batch)

    rows, cols = (4, 4) if num_zdim <= 16 else ((6, 6) if num_zdim <= 36 else (8, 8))

    fig = plt.figure(frameon=False, figsize=(14, 14))
    for i in range(num_zdim):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.hist(z_[:, i], bins=50, ec='black')
        ax.set_title('Dim: %d' % (i + 1))
        ax.set_xlim(-1, 1)
    plt.savefig('{}/z_dist.png'.format(results_dir))
    plt.close()
