import numpy as np

def NormalizeReconstruction(images, tracks, depths=None, fixed_scale=False, extent=10., p0=0.1, p1=0.9):
    coords = np.array([image.center() for image in images])
    coords_sorted = np.sort(coords, axis=0)
    P0 = int(p0 * (coords.shape[0] - 1)) if coords.shape[0] > 3 else 0
    P1 = int(p1 * (coords.shape[0] - 1)) if coords.shape[0] > 3 else coords.shape[0] - 1
    bbox_min = coords_sorted[P0]
    bbox_max = coords_sorted[P1]
    mean_coord = np.mean(coords_sorted[P0:P1+1], axis=0)

    if depths is not None:
        # depth-based normalization
        depth_gt_list = []
        depth_pred_list = []
        for track in tracks.values():
            for image_id, feature_id in track.observations:
                image = images[image_id]
                depth_gt = image.depths[feature_id]
                if depth_gt > 0:
                    C = image.center()
                    P = track.xyz
                    depth_pred = np.linalg.norm(P - C)
                    depth_gt_list.append(depth_gt)
                    depth_pred_list.append(depth_pred)
        if len(depth_gt_list) > 0:
            log_scales = np.log(np.array(depth_gt_list)) - np.log(np.array(depth_pred_list))
            scale = np.exp(np.median(log_scales))
        else:
            scale = 1.0
    else:
        # default normalization
        scale = 1.
        if not fixed_scale:
            old_extent = np.linalg.norm(bbox_max - bbox_min)
            if old_extent >= 1e-6:
                scale = extent / old_extent
        
    
    coords = (coords - mean_coord) * scale
    for idx, image in enumerate(images):
        image.world2cam[:3, 3] = -image.world2cam[:3, :3] @ coords[idx]
    for track in tracks.values():
        track.xyz = (track.xyz - mean_coord) * scale
