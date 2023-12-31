import motmetrics as mm
import numpy as np
import os

# # APPROACH 1: print metrics using MOTMETRICS
# requires reformatting to remove trailing spaces
# metrics = metrics_motchallenge_files(data_dir='../sompt22/train')

# PROVIDED BY MOTMETRICS GITHUB
# calculate motmetrics given ground truth and tracker object files
import numpy as np
import motmetrics as mm

def calculate_motmetrics(gtSource, tSource, sample_rate, bottom_left=False):
    # load ground truth
    gt = np.loadtxt(gtSource, delimiter=',')

    # load tracking output
    t = np.loadtxt(tSource, delimiter=',')

    if bottom_left:
        # Filter ground truth and tracking output to consider only the bottom-left corner
        # Image dimension: Width = 1920, Height = 1080
        gt = gt[(gt[:, 1] >= 0) & (gt[:, 2] >= 0) & (gt[:, 1] <= 960) & (gt[:, 2] <= 540)]
        t = t[(t[:, 1] >= 0) & (t[:, 2] >= 0) & (t[:, 1] <= 960) & (t[:, 2] <= 540)]

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number may be different for gt and t files
    for frame in range(1, int(t[:, 0].max()) + 1):
        # match frame id to downsampled id 
        frame_id = frame * sample_rate

        # print(frame_id, frame)
        # frame += 1  # detection and frame numbers begin at 1

        # select id, x, y, width, height for the current frame
        # required format for distance calculation is X, Y, Width, Height
        # We already have this format
        gt_dets = gt[gt[:, 0] == frame_id, 1:6]  # select all detections in gt
        t_dets = t[t[:, 0] == frame, 1:6]  # select all detections in t

        C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5)  # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:, 0].astype('int').tolist(), t_dets[:, 0].astype('int').tolist(), C)

        if frame == int(t[:, 0].max()):
            break

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                       'recall', 'precision', 'num_objects', \
                                       'mostly_tracked', 'partially_tracked', \
                                       'mostly_lost', 'num_false_positives', \
                                       'num_misses', 'num_switches', \
                                       'num_fragmentations', 'mota', 'motp' \
                                       ], \
                         name='acc')

    strsummary = mm.io.render_summary(
        summary,
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
                 'precision': 'Prcn', 'num_objects': 'GT', \
                 'mostly_tracked': 'MT', 'partially_tracked': 'PT', \
                 'mostly_lost': 'ML', 'num_false_positives': 'FP', \
                 'num_misses': 'FN', 'num_switches': 'IDsw', \
                 'num_fragmentations': 'FM', 'mota': 'MOTA', 'motp': 'MOTP', \
                 }
    )
    print(strsummary)

# PROVIDED BY MOTRACKERS
def compute_motchallenge(dname):
    df_gt = mm.io.loadtxt(os.path.join(dname, 'gt/gt.txt'))
    df_test = mm.io.loadtxt(os.path.join(dname, 'det/new_det.txt'))
    return mm.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)

def metrics_motchallenge_files(data_dir):
    """
    Metric evaluation for MOTChallenge.
    """
    dnames = ['SOMPT22-07']

    # accumulators for two datasets TUD-Campus and TUD-Stadtmitte.
    accs = [compute_motchallenge(os.path.join(data_dir, d)) for d in dnames]

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=dnames, generate_overall=True)

    print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names, formatters=mh.formatters))