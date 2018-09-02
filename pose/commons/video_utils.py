import pathlib2 as pathlib
import numpy as np
import os, re, cv2
from moviepy.video.io.VideoFileClip import VideoFileClip


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def natural_sort(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def trim_and_save(video_path, new_video_path, start=0, end=None):
    clip = VideoFileClip(video_path).subclip(start, end)
    print 'clip fps:', clip.fps
    clip.write_videofile(new_video_path, fps=clip.fps)


def generate_frames(video_path, fps=None, start=None, end=None, output_dir=None):
    vidcap = cv2.VideoCapture(video_path)
    num_video_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    required_fps = fps or video_fps

    required_frame_delay = video_fps / required_fps

    num_frames = round(num_video_frames // required_frame_delay)
    start_frame = 0 if start is None else round(video_fps * start)
    end_frame = num_frames if end is None else round(video_fps * end)

    selected_frames = {round(required_frame_delay * i) for i in range(int(start_frame), int(end_frame) + 1)}

    print('frame delay: %f, #frames: %d' % (required_frame_delay, len(selected_frames)))
    fcount = 0
    icount = 0
    success = True

    if output_dir:
        frame_dir_path = pathlib.Path(output_dir)
    else:
        parent_dir = pathlib.Path(video_path).absolute().parent
        video_name = os.path.basename(video_path)
        frames_dir_name = '_'.join(video_name.split('.'))
        frame_dir_path = parent_dir.joinpath(frames_dir_name)

    frame_dir_path.mkdir(parents=True, exist_ok=True)

    while success:
        success, image = vidcap.read()
        if success and fcount in selected_frames:
            frame_path = str(frame_dir_path.joinpath("frame%04d.png" % icount))
            cv2.imwrite(frame_path, image)
            icount += 1
        fcount += 1

    return icount


def generate_vid(dir_path, vid_format='XVID', outvid=None, fps=25, size=None, is_color=True):
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize

    fourcc = VideoWriter_fourcc(*vid_format)
    # out = cv2.VideoWriter(video_name, fourcc, 20.0, (640,480))

    vid = None

    images = [str(p) for p in pathlib.Path(dir_path).iterdir()]
    natural_sort(images)

    for image in images:
        if not os.path.exists(image):
            raise Exception('Image path does not exist: %s' % image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


def mask_clip(video_path, output_path, start=None, end=None, vid_format='XVID', x1=0, x2=1, y1=0, y2=1):
    vidcap = cv2.VideoCapture(video_path)
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    num_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    start_frame = 0 if start is None else round(video_fps * start)
    end_frame = num_frames if end is None else round(video_fps * end)

    selected_frames = {i for i in range(int(start_frame), int(end_frame) + 1)}

    fcount = 0
    success = True

    images = []
    while success:
        success, image = vidcap.read()
        if success and fcount in selected_frames:
            images.append(image)
        else:
            success = False
        fcount += 1

    print 'debug point'

    from cv2 import VideoWriter, VideoWriter_fourcc
    fourcc = VideoWriter_fourcc(*vid_format)
    print 'fourcc', fourcc
    size = images[0].shape[1], images[0].shape[0]
    print size
    vid = VideoWriter(output_path, fourcc, float(video_fps), size, True)

    if images:
        img = images[0]
        y1, y2, x1, x2 = y1 * img.shape[0], y2 * img.shape[0], x1 * img.shape[1], x2 * img.shape[1]
        y1, y2, x1, x2 = [int(i) for i in (y1, y2, x1, x2)]
        print y1, y2, x1, x2
    for img in images:
        masked_img = np.ones(img.shape, dtype=np.uint8) * 128
        masked_img[y1:y2, x1:x2] = img[y1:y2, x1:x2]
        vid.write(masked_img)
    vid.release()
    return vid



if __name__ == '__main__':
    dir_path = generate_frames('faded.mp4', fps=None, start=6, end=50)

    print('Frames stored at ', dir_path)

    dir_path = 'skeleton'
    vid = generate_vid(dir_path, outvid='skeleton.avi', fps=10)

    trim_and_save('data/raw_videos/cheez_badi.mp4', 'cheez_badi_1.mp4', start=39, end=44)
