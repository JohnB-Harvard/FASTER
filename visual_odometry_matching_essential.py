from PIL import Image, ImageDraw
import numpy as np
import h5py
import glob
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# Based on FAST corner finding. This implementation could be further optimized to speedup execution
def fast_corner_finder(image_a, px, py):
    circle = [(0,3), (1,3), (2,2), (3,1), (3,0), (3,-1), (2,-2), (1,-3), (0,-3), (-1,-3), (-2,-2), (-3,-1), (-3,0), (-3,1), (-2,2), (-1,3)]
    half_size = 3

    # This part of the method assumes a threshold of n=12 and we use n=9
    # count_pos = 0
    # count_neg = 0
    # for i in range(0,len(circle), int(len(circle)/4)):
    #     if image[py,px] > image[py+circle[i][0]][px+circle[i][1]]:
    #         count_neg += 1
    #     elif image[py,px] < image[py+circle[i][0]][px+circle[i][1]]:
    #         count_pos += 1
    # if(count_pos < 3 and count_neg < 3):
    #     return False 

    if py + half_size > len(image_a)-1 or px + half_size > len(image_a[0])-1 or py < half_size or px < half_size:
        return False
    if (image_a[py,px] == 0):
        return False
    if image_a[py-3,px] == image_a[py+3,px] and image_a[py,px-3] == image_a[py,px+3]:
        return False
    start_run = 0  
    start_run_ended = False
    max_run = 0
    cur_run = 0
    greater = image_a[py,px] < image_a[py+circle[0][0], px+circle[0][1]]
    greater_found = False
    less_found = False            
    for i in range(len(circle)):
        if image_a[py+circle[i][0], px+circle[i][1]] == -1:
            less_found = True
        if image_a[py+circle[i][0], px+circle[i][1]] == 1:
            greater_found = True
        if greater:
            if image_a[py,px] < image_a[py+circle[i][0], px+circle[i][1]]:
                cur_run += 1
                continue
            # end of current run
            if cur_run > max_run:
                max_run = cur_run
            if not start_run_ended:
                start_run = cur_run
                start_run_ended = True 
            if image_a[py,px] > image_a[py+circle[i][0], px+circle[i][1]]:
                cur_run = 1
                greater = False
            if image_a[py,px] == image_a[py+circle[i][0], px+circle[i][1]]:
                cur_run = 0

        else:
            if image_a[py,px] > image_a[py+circle[i][0], px+circle[i][1]]:
                cur_run += 1
                continue
            # end of current run
            if cur_run > max_run:
                max_run = cur_run
            if not start_run_ended:
                start_run = cur_run
                start_run_ended = True 
            if image_a[py,px] < image_a[py+circle[i][0], px+circle[i][1]]:
                cur_run = 1
                greater = True
            if image_a[py,px] == image_a[py+circle[i][0], px+circle[i][1]]:
                cur_run = 0
    if(cur_run + start_run > max_run):
        max_run = cur_run + start_run

    if (not (greater_found and less_found)):
        return False

    # Seems to work well for 1mX1mY. 
    return max_run > int(len(circle)/2) and max_run != len(circle)

# Returns an estimated vel as (vy,vx)
def calc_flow_matching(im, im_next, feature_list, window_size):
    search_size = int(window_size/12)
    vel_list = []
    for feature in feature_list:
        if feature[0] == None:
            vel_list.append((None,None))
            continue
        old_point = None
        new_point = None
        '''
        # this is searching from the feature out in roughly the same order as the feature finding code
        # for r in range(search_size):
        #     for x in range(2*r+1):
        #         if new_point is not None:
        #             break
        #         px = feature[1]+x-r
        #         py = feature[0]-r
        #         is_corner = fast_corner_finder(im, px, py, window_size)
        #         if is_corner:
        #             new_point = (py,px)
        #             break
        #     for y in range(2*r-1):
        #         if new_point is not None:
        #             break
        #         for px in [feature[1]-x, feature[1]-x]:
        #             py = feature[0]-r+y+1
        #             is_corner = fast_corner_finder(im, px, py, window_size)
        #             if is_corner:
        #                 new_point = (py,px)
        #                 break
        #     for x in range(2*r+1):
        #         if new_point is not None:
        #             break
        #         px = feature[1]+x-r
        #         py = feature[0]+r
        #         is_corner = fast_corner_finder(im, px, py, window_size)
        #         if is_corner:
        #             new_point = (py,px)
        #             break
        #     if new_point is not None:
        #         break
        '''
        # This part is just to cover. I should think of a way to search locally first. Note: Local search doesn't seem to work as well
        for py in range(feature[0]-search_size, feature[0]+search_size+1):
            for px in range(feature[1]-search_size, feature[1]+search_size+1):
                is_corner = fast_corner_finder(im, px, py, window_size)
                if is_corner:
                    old_point = (py,px)
                    break
            if old_point is not None:
                break
        for py in range(feature[0]-search_size, feature[0]+search_size+1):
            for px in range(feature[1]-search_size, feature[1]+search_size+1):
                is_corner = fast_corner_finder(im_next, px, py, window_size)
                if is_corner:
                    new_point = (py,px)
                    break
            if new_point is not None:
                break
        if new_point is None or old_point is None:
            vel_list.append((None,None))
        else:
            vel_list.append((new_point[0]-old_point[0], new_point[1]-old_point[1]))
    return vel_list

# Returns an estimated vel as (vy,vx)
def update_features(im, im_next, feature_list, search_size, valid):
    num_valid = valid
    new_features = [(None, None) for _ in feature_list]
    for i, feature in enumerate(feature_list):
        if feature[0] == None:
            continue
        old_point = None
        # Search for a new feature near the previous one. Note that the window size should be set so I don't confuse features here
        # If you are too close to an edge stop tracking
        if feature[1] < search_size or feature[0] < search_size or feature[1] > len(im[0])-search_size or feature[0] > len(im)-search_size:
            num_valid -= 1
            continue
        for py in range(feature[0]-search_size, feature[0]+search_size+1):
            for px in range(feature[1]-search_size, feature[1]+search_size+1):
                is_corner = fast_corner_finder(im, px, py)
                if is_corner:
                    old_point = (py, px)
                    break
            if old_point is not None:
                break
        if old_point == None:
            num_valid -= 1
            continue
        for py in range(feature[0]-search_size, feature[0]+search_size+1):
            for px in range(feature[1]-search_size, feature[1]+search_size+1):
                is_corner = fast_corner_finder(im_next, px, py)
                if is_corner:
                    new_features[i] = (feature[0] + py-old_point[0], feature[1] + px-old_point[1])
                    break
            if new_features[i][0] is not None:
                break
        if new_features[i][0] is None:
            num_valid -= 1
    return new_features, num_valid



def calc_flow(im, im_next, feature_list, window_size):
    half_size = int(window_size/2)
    velocity_list = []
    for feature in feature_list:
        I_x = np.zeros((window_size,window_size))
        I_y = np.zeros((window_size,window_size))
        I_t = np.zeros((window_size,window_size))
        for i in range(-half_size,half_size+1):
            for j in range(-half_size,half_size+1):
                I_x[i+half_size,j+half_size] = im[feature[0]+i+1, feature[1]+j] - im[feature[0]+i, feature[1]+j]
                I_y[i+half_size,j+half_size] = im[feature[0]+i, feature[1]+j+1] - im[feature[0]+i, feature[1]+j]
                I_t[i+half_size,j+half_size] = -1*(im_next[feature[0]+i, feature[1]+j] - im[feature[0]+i, feature[1]+j])
        # print(I_x)
        # print(I_y)
        # print(I_t)
        # compute velocity vector:

        # form S matrix now:
        flatten_x = I_x.flatten()
        transpose_changes_x = np.array([flatten_x]).transpose()

        flatten_y = I_y.flatten()
        transpose_changes_y = np.array([flatten_y]).transpose()

        flatten_t = I_t.flatten()
        transpose_changes_t = np.array([flatten_t]).transpose()

        s_matrix = np.concatenate(np.array([transpose_changes_x, transpose_changes_y]), axis=1)
        s_matrix_transpose = s_matrix.transpose()
        t_matrix = transpose_changes_t
        st_s = np.matmul(s_matrix_transpose, s_matrix)

        w, _ = np.linalg.eig(st_s)
        if(np.abs(w[0]) < .001 or np.abs(w[1]) < .001):
            velocity_list.append([0,0])
            break
        # cond = abs(w[0]) / abs(w[1])
        # print("Condition number:")
        # print(cond)

        st_s_inv = np.linalg.inv(st_s)
        temp_matrix = np.matmul(st_s_inv, s_matrix_transpose)
        velocity_list.append(np.matmul(temp_matrix, t_matrix).flatten())
    return velocity_list

# return the estimated egomotion in the form x, y, z, xrot, yrot, zrot.
def egomotion_from_features(feature_list, old_features, prev_pos):
    print(feature_list[0], old_features[0])
    assert(len(feature_list) == len(old_features))
    # based on https://courses.cs.duke.edu/spring19/compsci527/slides/L17_LonguetHiggins.pdf

    # Create the C matrix
    C = []
    for i in range(len(feature_list)):
        if feature_list[i][0] != None:
            x1 = feature_list[i][1]
            y1 = feature_list[i][0]
            x2 = old_features[i][1]
            y2 = old_features[i][0]
            C.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    C = np.array(C)
    print(C.shape)
    # Least squares solution to CtCx=0 for |x| = 1 is eigenvector corresponding to minimum eigenvalue of CtC by https://foto.aalto.fi/seura/julkaisut/pjf/pjf_e/2005/Inkila_2005_PJF.pdf
    eig,eig_vec = np.linalg.eig(np.matmul(C.transpose(), C))
    eig = np.abs(eig)
    min_eig = min(eig)
    index = -1
    for i in range(len(eig)):
        if eig[i] == min_eig:
            index = i
            break
    assert(index != -1)
    essential_vectorized = eig_vec[:,index]
    essential = essential_vectorized.reshape(3,3)

    # now using https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf
    U,D,Vt = np.linalg.svd(essential)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    print(f"SVD of essential matrix got U=\n{u}, d=\n{D}, vt = \n{Vt}")
    R1 = U @ W @ Vt
    print(f"R1 is {R1} with det {np.linalg.det(R1)}")
    R2 = U @ W.T @ Vt
    print(f"R2 is {R2} with det {np.linalg.det(R2)}")
    pose_1 = np.concatenate((R1, U[:,2].reshape(3,1)), axis=1)
    print(f"Pose 1 is \n{pose_1}")
    print(U[0,2]/U[2,2], U[1,2]/U[2,2], U[2,2]/U[2,2])

def egomotion_cv2(feature_list, old_features, prev_pos):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in range(len(feature_list)):
        if feature_list[i][0] is not None:
            x1.append(old_features[i][1])
            y1.append(old_features[i][0])
            x2.append(feature_list[i][1])
            y2.append(feature_list[i][0])
    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    # From blender, K = 
    # (817.7778,    0.0000, 256.0000)
    # (  0.0000, 1226.6666, 256.0000)
    # (  0.0000,    0.0000,   1.0000)

    K = np.array([[817.7778, 0, 256],
                  [0, 1226.6666, 256],
                  [0, 0, 1]])
    # print(f"using K={K}")
    img1_kpts = np.hstack([ x1.reshape(-1,1), y1.reshape(-1,1) ]).astype(np.int32)
    img2_kpts = np.hstack([ x2.reshape(-1,1), y2.reshape(-1,1) ]).astype(np.int32)
    print(f"Num points is: {len(img1_kpts)}")
    cam2_E_cam1, inlier_mask = cv2.findEssentialMat(img1_kpts, img2_kpts, K, method=cv2.RANSAC, threshold=0.05)
    print('Num inliers: ', inlier_mask.sum())

    _,R,t,_ = cv2.recoverPose(cam2_E_cam1, img1_kpts, img2_kpts, K, mask=inlier_mask)
    r = Rotation.from_matrix(R)
    angles = r.as_euler("zyx", degrees=True)
    print(f"Computed rotation is \n{angles}\n and translation is {t}")

    # R1, R2, t = cv2.decomposeEssentialMat(cam2_E_cam1)
    # ### first transform the matrix to euler angles
    # r1 =  Rotation.from_matrix(R1)
    # r2 = Rotation.from_matrix(R2)
    # angles1 = r1.as_euler("zyx",degrees=True)
    # angles2 = r2.as_euler("zyx",degrees=True)
    # print(f"Possible rotations are R1=\n{R1}\n which is {angles1} \nR2=\n{R2}\n which is {angles2}\n and Translation {t}")

# OpenCv Rodrigues to interpret the rotation  matrix
    # open cv uses RVec and tvec where rvec is the rodrigues vector and t is the translation for camera pose

folders = glob.glob('*5mDepthRender')
window_size = 51
half_size = 25
search_size = 10
history_size = 100
max_features = 36
num_divs = 9

color_list = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(50)]

for search_size in [75]:
    dirs = ['15dZRot_5mDepthRender']
    # dirs = ['1mX_5mDepthRender', 'Thyme_1mX1mY_5mDepthRender', '1mZ_5mDepthRender', '15dZRot_5mDepthRender']
    # for folder in folders:
    for folder in dirs:
        print(f"Starting with {folder}")
        f = h5py.File(f'{folder}\events.h5')

        # events is a list of tuples of form (x, y, polarity, time), sort by time
        events = list(f.get("events"))
        events.sort(key=lambda e: e[3])
        print(events[0])
        print(events[-1])

        events_a = [np.zeros((512,512), dtype=int) for _ in range(events[-1][3]+1)]
        for event in events:
            assert(event[2] == 1 or event[2] == -1)
            events_a[event[3]][event[0]][event[1]] = event[2]
        print("Created event images")

        events_im = Image.fromarray(events_a[0]*127+128)
        im2 = Image.fromarray(events_a[0]*127+128)
        draw = ImageDraw.Draw(events_im)
        draw2 = ImageDraw.Draw(im2)

        events_images = []
        events_draws = []
        for i in range(len(events_a)):
            assert(events_a[i] is not None)
            events_im = Image.fromarray(events_a[i]*127+128).convert('P')
            events_images.append(events_im)
            events_draws.append(ImageDraw.Draw(events_im))

        old_features = [] # oldest history
        old_pos = np.zeros(6)
        prev_features = [] # update for each timestep
        feature_list = []
        color_list = []
        valid = 0
        initial_len = 0
        for ts in range(len(events_a)-1):
            if valid <= 15:
                # need to refresh feature list
                # feature_list = []
                # for py in range(len(events_a[0])):
                #     for px in range(len(events_a[0][0])):
                #         is_corner = fast_corner_finder(events_a[ts], px, py)
                #         if(is_corner):
                #             found = False
                #             for u in range(-half_size,half_size+1):
                #                 for v in range(-half_size,half_size+1):
                #                     if (py+u,px+v) in feature_list:
                #                         found = True
                #                         break
                #                 if found:
                #                     break
                #             if not found:
                #                 feature_list.append((py,px))
                # Trying new refresh system to get spread out features with max_features. Steps through the image in divisions
                feature_list = []
                all_failed = False
                while(len(feature_list) < max_features):
                    all_failed = True
                    for iy in range(num_divs):
                        for ix in range(num_divs):
                            region_found = False
                            for py in range(iy*int(len(events_a[0])/num_divs), (iy+1)*int(len(events_a[0])/num_divs)):
                                for px in range(ix*int(len(events_a[0][0])/num_divs), (ix+1)*int(len(events_a[0][0])/num_divs)):
                                    if(fast_corner_finder(events_a[ts], px, py)):
                                        found = False
                                        for u in range(-half_size,half_size+1):
                                            for v in range(-half_size,half_size+1):
                                                if (py+u,px+v) in feature_list:
                                                    found = True
                                                    break
                                            if found:
                                                break
                                        if not found:
                                            region_found = True
                                            all_failed = False
                                            feature_list.append((py,px))
                                    if region_found:
                                        break
                                if region_found:
                                    break
                    if all_failed:
                        break
                initial_len = len(feature_list)
                valid = len(feature_list)
                print(f"found {initial_len} features")
                old_features = []
                old_features.extend(feature_list)
                prev_features = []
                prev_features.extend(feature_list)
                color_list = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(len(feature_list))]
            prev_features = []
            prev_features.extend(feature_list)
            new_features, valid = update_features(events_a[ts], events_a[ts+1], feature_list, search_size, valid)
            for i, feature in enumerate(new_features):
                if feature[0] is not (None):
                    for events_draw in events_draws[ts:]:
                        events_draw.line((feature[1], feature[0], prev_features[i][1], prev_features[i][0]), fill=color_list[i], width=2)
            feature_list = []
            feature_list.extend(new_features)
            # if ts == len(events_a)-2:
            #     # print(feature_list)
            #     egomotion_cv2(feature_list, old_features, old_pos)
            #     color_list = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(len(feature_list))]
            #     for i, feature in enumerate(feature_list):
            #         if feature[0] is None:
            #             continue
            #         for events_draw in events_draws[ts:]:
            #             events_draw.line((feature[1], feature[0], old_features[i][1], old_features[i][0]), fill=color_list[i], width=2)
            #             events_draw.ellipse((feature[1]-2, feature[0]-2, feature[1]+2, feature[0]+2), outline=color_list[i], fill=color_list[i])
            #     break

        # events_images[0].save(f'{folder}\{folder}_annotated_flowMatch_cv2_size51_searchSize5.gif',
        #             save_all=True, append_images=events_images[1:], optimize=False, duration=20, loop=0)
        events_images[-1].save(f'{folder}\\LocalSearch_results_{search_size}.png', format="PNG")