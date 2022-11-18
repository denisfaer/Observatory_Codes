""" import required libraries """

from sklearn.neighbors import KDTree
from dataclasses import dataclass
import operator
import os
import csv
import math
import numpy as np
import cv2
import time
StartTime = time.time()


""" setup custom dataclasses """

@dataclass
class ConfAdaptiveThreshold:
    kernel_size: int
    C: float


@dataclass
class ConfBinarize:
    kernel_size: int


@dataclass
class ConfContour:
    min_area: float
    max_area: float
    min_ratio: float
    max_ratio: float


@dataclass
class Conf:
    adaptive_threshold: ConfAdaptiveThreshold
    binarize: ConfBinarize
    contour: ConfContour


default_conf = Conf(
    ConfAdaptiveThreshold(15, 6),
    ConfBinarize(2),
    ConfContour(50, 200, 0.45, 0.75)) # area_min, area_max, perim-to-area ratio min, perim-to-area ratio max


@dataclass
class BlobFrame:
    time: float
    cx: float
    cy: float
    area: float
    contour: any
    disp: float


class Blob:
    is_active: bool

    def __init__(self, id, time, cx, cy, area, contour):
        self.id = id
        self.data = [BlobFrame(time, cx, cy, area, contour, 0.0)]
        self.is_active = True

    def add_frame_data(self, time, cx, cy, area, contour):
        self.data.append(BlobFrame(time, cx, cy, area, contour, self.distance))

    @property
    def time(self):
        return self.data[-1].time

    @property
    def distance(self):
        if len(self.data) < 2:
            return 0.0
        cx2, cy2 = self.data[-2].cx, self.data[-2].cy
        cx1, cy1 = self.data[-1].cx, self.data[-1].cy
        return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

    @property
    def point(self):
        return [self.data[-1].cx, self.data[-1].cy]

    @property
    def area(self):
        return self.data[-1].area


@dataclass
class VideoTrackerPlate:
    ox: int
    oy: int
    radius: int


class VideoTracker:
    conf: Conf
    plate: VideoTrackerPlate
    blobs: list[Blob]
    next_blob_id: int

    def __init__(self, conf: Conf, plate: VideoTrackerPlate):
        self.conf = conf
        self.plate = plate
        self.blobs = []
        self.next_blob_id = 1
        self.roi_im = None

    def treat_frame(self, time, frame):
        """
        Parameters:
            - time: float
            - frame: image numpy array
        Returns:
            (original_im_crop, colored_im_crop)
        """
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        r_im = frame[0:h, 0:w, 0]
        rend_im = np.zeros_like(r_im)
        self.roi_im = cv2.circle(rend_im,
                                 (self.plate.ox, self.plate.oy),
                                 self.plate.radius,
                                 color = 1,
                                 thickness = cv2.FILLED)
        # Cut the ROI
        plate_specific_im = r_im * self.roi_im

        # Binarize
        binary_adaptive_im = cv2.adaptiveThreshold(plate_specific_im,
                                                   1,
                                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY,
                                                   self.conf.adaptive_threshold.kernel_size,
                                                   self.conf.adaptive_threshold.C)

        # Remove Salt&Pepper
        kernel = np.ones((self.conf.binarize.kernel_size, self.conf.binarize.kernel_size), np.uint8)
        binarized_im = binary_adaptive_im
        binarized_im = cv2.morphologyEx(binarized_im, cv2.MORPH_CLOSE, kernel)
        binarized_im = cv2.morphologyEx(binarized_im, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours_im = binarized_im.copy()
        contours, _ = cv2.findContours(contours_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        raw_blobs = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            ratio = cv2.arcLength(contour, True) / area 
            if area < self.conf.contour.min_area:
                cv2.drawContours(
                    binarized_im, [contour], -1, 128, thickness = cv2.FILLED)
                continue
            if area > self.conf.contour.max_area:
                continue
            if ratio < self.conf.contour.min_ratio:
                continue
            if ratio > self.conf.contour.max_ratio:
                continue
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if (cx > labels[cnt][0] and cx < labels[cnt][0] + labels[cnt][2]) and (cy > labels[cnt][1] and cy < labels[cnt][1] + labels[cnt][3]):
                continue
            raw_blobs.append((cx, cy, area, contour))

        # Connect raw blobs to the current blobs
        values = [(index, blob.point) for index, blob in enumerate(self.blobs) if blob.is_active]
        if len(values) == 0:
            for raw_blob in raw_blobs:
                cx, cy, area, contour = raw_blob
                id = self.next_blob_id
                self.next_blob_id += 1
                self.blobs.append(Blob(id, timef, cx, cy, area, contour))
        else:
            blob_indices, points = zip(*values)
            tree = KDTree(points, leaf_size = 2)
            used_blob_indices = set()
            for raw_blob in raw_blobs:
                cx, cy, area, contour = raw_blob
                # Be careful, a contour cannot be connected to more than one blob, so
                # you must control the contours that you already used!
                best_blob_index = None
                distances, query_indices = tree.query([(cx, cy)], k = min(8, len(points)))
                for distance, index in zip(distances.flatten(), query_indices.flatten()):
                    if distance > 100:
                        break
                    blob_index = blob_indices[index]
                    if blob_index in used_blob_indices:
                        continue
                    best_blob_index = blob_index
                    break

                if best_blob_index is not None:
                    self.blobs[best_blob_index].add_frame_data(timef, cx, cy, area, contour)
                    used_blob_indices.add(best_blob_index)
                else:
                    id = self.next_blob_id
                    self.next_blob_id += 1
                    self.blobs.append(Blob(id, timef, cx, cy, area, contour))

        # Mark
        for blob in self.blobs:
            # if a blob is not active for more than 2 minutes, it becomes inactive
            if timef - blob.time > 120_000:
                blob.is_active = False

        # Construct frame output for debugging
        original_im = frame.copy()

        colored_im = np.stack([binarized_im * 255, binarized_im * 255, binarized_im * 255], axis = -1)
        for blob in self.blobs:
            if blob.is_active:
                color = (255, 0, 0)
            elif timef - blob.time < 240_000:
                color = (0, 255, 128)
            else:
                continue
            cv2.putText(original_im, f"{blob.id}", blob.point, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(colored_im, f"{blob.id}", blob.point, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        original_im_crop = original_im[(self.plate.oy - self.plate.radius):(self.plate.oy + self.plate.radius),
                                  (self.plate.ox - self.plate.radius):(self.plate.ox + self.plate.radius)]
        colored_im_crop = colored_im[(self.plate.oy - self.plate.radius):(self.plate.oy + self.plate.radius),
                                (self.plate.ox - self.plate.radius):(self.plate.ox + self.plate.radius)]

        return original_im_crop, colored_im_crop

def bresenline(x0, y0, xd, yd):
    ray_list = []
    
    if xd == x0:
        y = y0
        step = np.sign(yd - y0)
        while y != yd:
            ray_list.append([x0, y])
            y += step
    else:
        slope = (yd - y0) / (xd - x0)
        if abs(xd - x0) > abs(yd - y0):
            x = x0
            step = np.sign(xd - x0)
            while x != xd:
                y = round(slope * (x - x0) + y0)
                ray_list.append([x, y])
                x += step
        else:
            y = y0
            step = np.sign(yd - y0)
            while y != yd:
                x = round((y - y0) / slope + x0)
                ray_list.append([x, y])
                y += step

    return ray_list


""" initial setup """

Config = [21, 3, 3] # plate identification binarizing thresholds
Radius_censor = 30 # how much smaller in pixels the ROI should be
Blur_val = 7 # blurring threshold for plate identification
speedup = 5 # frame rate acceleration for output videos
desired_time = 60_000 # process these many millisec
N_worms = 6 # expected number of plates per video

trouble_image = True # process step-by step images
shorten = True # limit analysis to millisec saved in timep
write_individual_videos = True # produce a zoomed-in video for each plate
harsh = False # do not process videos with not exactly 6 plates identified
specific_blob_output = True # output average displacements for specified blobs

# setup main log
main_log = []
main_log.append('PlateConfig: [' + str(Config[0]) + ', ' + str(Config[1]) + ', ' + str(Config[2]) + ']\n')
main_log.append('RadiusCens: ' + str(Radius_censor) + '\n')
main_log.append('BlurVal: ' + str(Blur_val) + '\n')
main_log.append('FPSaccel: ' + str(speedup) + '\n')
if shorten:
    main_log.append('AnalysisTime: ' + str(desired_time) + ' msec\n')
else:
    main_log.append('AnalysisTime: Full\n')
main_log.append('\n')


""" execute video analysis """

# establish folder naming scheme
cages = [[1, "Cage 1"]]
cameras = [[1, "Camera 1"]]

main_directory = os.getcwd()

if specific_blob_output:
    blob_file = open(os.path.join(main_directory, 'specific_blobs.txt'))

run_disp_ave = []

if specific_blob_output:
    run_disp_spe = []

for cage in cages:
    if len(cages) > 1:
        cage_directory = os.path.join(main_directory, cage[1])
    else:
        cage_directory = main_directory
    cage_disp_ave = []
    
    if specific_blob_output:
        cage_disp_spe = []
    
    for camera in cameras:
        print('Cage ' + str(cage[0]) + ' Camera ' + str(camera[0]))
        main_log.append('Cage ' + str(cage[0]) + ' Camera ' + str(camera[0]) + '\n')
        if len(cameras) > 1:
            camera_directory = os.path.join(cage_directory, camera[1])
        else:
            camera_directory = cage_directory
        
        StartTimeCam = time.time()
        
        # setup analysis log
        log = []
        log.append('PlateConfig: [' + str(Config[0]) + ', ' + str(Config[1]) + ', ' + str(Config[2]) + ']\n')
        log.append('RadiusCens: ' + str(Radius_censor) + '\n')
        log.append('BlurVal: ' + str(Blur_val) + '\n')
        log.append('FPSaccel: ' + str(speedup) + '\n')
        if shorten:
            log.append('AnalysisTime: ' + str(desired_time) + ' msec\n')
        else:
            log.append('AnalysisTime: Full\n')
        
        avi_files = [f for f in os.listdir(camera_directory) if f.endswith('.avi')]
        video = avi_files[0]
        
        # get first frame for area analysis
        file = os.path.join(camera_directory, video)

        cap = cv2.VideoCapture(file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        path = os.path.join(camera_directory, 'Output')
        if not os.path.exists(path):
            os.mkdir(path)


        """ find Plates from first frame """

        ret, frametemp = cap.read()
        framer = cv2.cvtColor(frametemp, cv2.COLOR_BGR2GRAY)

        # Binarize
        bin_adaptive_im = cv2.adaptiveThreshold(framer,
                                                1,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY,
                                                Config[0],
                                                Config[1])

        # Remove Salt&Pepper
        kernel = np.ones((Config[2], Config[2]), np.uint8)
        bin_im = bin_adaptive_im
        bin_im = cv2.morphologyEx(bin_im, cv2.MORPH_CLOSE, kernel)
        bin_im = cv2.morphologyEx(bin_im, cv2.MORPH_OPEN, kernel)

        # make plate image
        plate_im = np.stack([bin_im * 255, bin_im * 255, bin_im * 255], axis = -1)
        plate_im = cv2.medianBlur(plate_im, Blur_val)

        plate_im = cv2.cvtColor(plate_im, cv2.COLOR_BGR2GRAY)
        ray_im = plate_im.copy()

        circles = cv2.HoughCircles(plate_im,
                                   method = cv2.HOUGH_GRADIENT,
                                   dp = 1,
                                   minDist = 1500,
                                   param1 = 100,
                                   param2 = 30,
                                   minRadius = 730,
                                   maxRadius = 770)

        circles = np.uint16(np.around(circles))

        formPlate = circles[0, :]  # formatted Plate coordinate list

        N_plates = len(formPlate)
        if N_plates < N_worms or N_plates > N_worms:
            print('WARNING: ' + str(N_plates) + ' plates identified')
            log.append('WARNING: ' + str(N_plates) + ' plates identified\n')
            main_log.append('WARNING: ' + str(N_plates) + ' plates identified\n')
        
        if (N_plates < N_worms or N_plates > N_worms) and harsh:
            cap.release()
            break
        
        # new plate code
        plate_im = cv2.cvtColor(plate_im, cv2.COLOR_GRAY2RGB)
        for i in circles[0, :]:
            cv2.circle(plate_im, (i[0], i[1]), i[2], (0, 0, 255), 5)
            cv2.circle(plate_im, (i[0], i[1]), 2, (255, 0, 0), 25)
        
        if trouble_image:
            cv2.imwrite(os.path.join(path, 'trb_simple_plates.jpg'), plate_im)
            ray_im_out = cv2.cvtColor(ray_im, cv2.COLOR_GRAY2RGB)
        
        smartPlate = []
        for plate in formPlate:
            found = []
            for angle in range(0, 360, 10):
                angle_rad = math.radians(angle)
                x_origin, y_origin = plate[0], plate[1]
                x_crcle, y_circle = round(x_origin + plate[2] * math.cos(angle_rad)), round(y_origin + plate[2] * math.sin(angle_rad))
                ray = bresenline(x_origin, y_origin, x_crcle, y_circle)
                for dot in ray:
                    if ray_im[dot[1], dot[0]] == 0:
                        found.append([dot[0], dot[1]])
                        break

            cont = np.array(found).reshape((-1,1,2)).astype(np.int32)
            (x, y), radius = cv2.minEnclosingCircle(cont)
            smartPlate.append([int(x), int(y), int(radius)])
            
            if trouble_image:
                center = (int(x), int(y))
                R = int(radius)
                cv2.circle(ray_im_out, center, R, (0, 0, 255), 5)
                cv2.circle(ray_im_out, center, R - Radius_censor, (0, 255, 0), 5)
                for ray in found:
                    cv2.line(ray_im_out, (plate[0], plate[1]), (ray[0], ray[1]), (255, 0, 0), 2)

        if trouble_image:
            cv2.imwrite(os.path.join(path, 'trb_ray_plates.jpg'), ray_im_out)
        
        """ order plates """

        sortPlate = [(-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1)] # holds plate (X, Y, R) sorted left-to-right top-down in rows

        for plate in smartPlate:
            if plate[1] < height/2:
                if plate[0] < width/3:
                    sortPlate[0] = plate
                else:
                    if plate[0] > width*2/3:
                        sortPlate[2] = plate
                    else:
                        sortPlate[1] = plate
            else:
                if plate[0] < width/3:
                    sortPlate[3] = plate
                else:
                    if plate[0] > width*2/3:
                        sortPlate[5] = plate
                    else:
                        sortPlate[4] = plate


        """ identify plate labels """
        
        if trouble_image:
            full_im = np.stack([bin_im * 255, bin_im * 255, bin_im * 255], axis = -1)
        
        cnt = 1
        labels = []

        for plate in sortPlate:
            contours_im = bin_im.copy()
            contours_im = contours_im[(plate[1] - plate[2]):(plate[1] + plate[2]), (plate[0] - plate[2]):(plate[0] + plate[2])]
            contours, _ = cv2.findContours(contours_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            suspects = []
            
            for contour in contours:
                if cv2.contourArea(contour) > 9_000:
                    color = (0, 0, 255)
                elif cv2.contourArea(contour) > 4_000:
                    color = (255, 0, 0)
                    suspects.append(contour)
                elif cv2.contourArea(contour) > 150:
                    color = (255, 255, 0)    
                elif cv2.contourArea(contour) > 50:
                    color = (0, 255, 0)
                else:
                    continue
                if trouble_image:
                    x, y, w, h = cv2.boundingRect(contour)
                    y += plate[1] - plate[2]
                    x += plate[0] - plate[2]
                    cv2.rectangle(full_im, (x, y), (x + w, y + h), color, 2)
            
            suspect_id = []
            suspect_count = 0
            for suspect in suspects:
                x, y, w, h = cv2.boundingRect(suspect)
                y += plate[1] - plate[2]
                x += plate[0] - plate[2]
                suspect_id.append([x, y, w, h, 2 * (w + h)])
                suspect_count += 1
            
            if suspect_count > 0:
                indexed_suspects = enumerate([k[4] for k in suspect_id])
                sorted_suspects = sorted(indexed_suspects, key = operator.itemgetter(1))
                label = suspect_id[sorted_suspects[0][0]]
                labels.append(label)
                
                if trouble_image:
                    if cnt <= 3:
                        offset = -10
                    else:
                        offset = label[3] + 50
                    cv2.putText(full_im, 'Label ' + str(cnt), (label[0], label[1] + offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
                    cv2.putText(plate_im, 'Label ' + str(cnt) , (label[0], label[1] + offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
                    cv2.rectangle(plate_im, (label[0], label[1]), (label[0] + label[2], label[1] + label[3]), (255, 0, 0), 2)
            else:
                print('Warning: No labels found on Plate ' + str(cnt))
                log.append('Warning: No labels found on Plate ' + str(cnt) + '\n')
                main_log.append('Warning: No labels found on Plate ' + str(cnt) + '\n')
                labels.append([1, 1, 1, 1, 4])

                cnt += 1

        if trouble_image:
            cv2.imwrite(os.path.join(path, 'trb_full_im.jpg'), full_im)
            cv2.imwrite(os.path.join(path, 'trb_plates.jpg'), plate_im)
        


        """ process video """

        if write_individual_videos:
            processed_output_ind = []
            binarized_output_ind = []
            cnt = 0
            for plate in sortPlate:
                cnt += 1
                processed_output_ind.append(cv2.VideoWriter(os.path.join(path, str(cnt) + '_processed.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps * speedup, (2 * (plate[2] - Radius_censor), 2 * (plate[2] - Radius_censor))))
                binarized_output_ind.append(cv2.VideoWriter(os.path.join(path, str(cnt) + '_binarized.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps * speedup, (2 * (plate[2] - Radius_censor), 2 * (plate[2] - Radius_censor))))
            
        processed_output = cv2.VideoWriter(os.path.join(path, 'output_processed.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps * speedup, (width, height))
        binarized_output = cv2.VideoWriter(os.path.join(path, 'output_binarized.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps * speedup, (width, height))

        tracker = []
        for plate in sortPlate:
            tracker.append([VideoTracker(conf = default_conf, plate = VideoTrackerPlate(plate[0], plate[1], plate[2] - Radius_censor)), plate[0], plate[1], plate[2] - Radius_censor])

        count = 0

        if shorten:
            timep = desired_time
        else:
            timep = 999_999_999_000

        # core frame processing loop
        while cap.isOpened() and (cap.get(cv2.CAP_PROP_POS_MSEC) <= timep):
            ret, frame_read = cap.read()      
            timef = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            # stop looping if frame not read
            if not ret:
                break
            processed_im = []
            binarized_im = []
            cnt = 0             
            for track in tracker:
                processed_im_crop, binarized_im_crop = track[0].treat_frame(timef, frame_read)
                processed_im.append([processed_im_crop, track[1], track[2], track[3]])
                binarized_im.append([binarized_im_crop, track[1], track[2], track[3]])
                if write_individual_videos:
                    processed_output_ind[cnt].write(processed_im_crop)
                    binarized_output_ind[cnt].write(binarized_im_crop)
                cnt += 1
            
            processed_frame = frame_read
            binarized_frame = np.zeros((height, width, 3), np.uint8)
            binarized_frame[:] = (255, 255, 255)
            
            for image in processed_im:
                processed_frame[(image[2] - image[3]):(image[2] + image[3]), (image[1] - image[3]):(image[1] + image[3])] = image[0]
            
            for image in binarized_im:
                binarized_frame[(image[2] - image[3]):(image[2] + image[3]), (image[1] - image[3]):(image[1] + image[3])] = image[0]
            
            processed_output.write(processed_frame)
            binarized_output.write(binarized_frame)
            
            # infinite looping precaution
            count += 1
            if count > 1_000_000:
                break

        # release AVI files
        cap.release()
        processed_output.release()
        binarized_output.release()
        if write_individual_videos:
            for writer in processed_output_ind:
                writer.release()
            for writer in binarized_output_ind:
                writer.release()


        """ output data for highest mover in each plate """

        camera_disp_ave = []
        cnt = 0
        
        for track in tracker:
            # calculate total movement
            mover = []
            count_movers = 0
            for blob in track[0].blobs:
                  sumr = 0
                  vectr = 0
                  count_movers += 1
                  times = 0
                  x0 = blob.data[0].cx
                  y0 = blob.data[0].cy
                  for timer in blob.data:
                     sumr += timer.disp
                     vectr += math.sqrt((timer.cx - x0) ** 2 + (timer.cy - y0) ** 2)
                     times += 1
                  mover.append([blob.id, sumr, times, sumr / times, vectr])
                  
            if count_movers > 0:
                # output average displacement for all blobs
                data = []
                data.append(["BlobID", "TotalDisp", "Frames", "AveDisp"])
                for worm in mover:
                    data.append([worm[0], worm[1], worm[2], worm[3]])
                file = open(os.path.join(path, str(cnt + 1) + '_average_disp.csv'), 'w+', newline = '')
                with file:
                    write = csv.writer(file)
                    write.writerows(data)
                file.close()
                
                # sort movers and find highest
                indexed_mover = enumerate([k[4] for k in mover])
                sorted_mover = sorted(indexed_mover, key = operator.itemgetter(1), reverse = True)
                highest_mover = mover[sorted_mover[0][0]]
                call_id = highest_mover[0] - 1
                
                print('Plate ' + str(cnt + 1) + ' worms is: ' + str(highest_mover[0]))
                log.append('Plate ' + str(cnt + 1) + ' worms is: ' + str(highest_mover[0]) + '\n')
                main_log.append('Plate ' + str(cnt + 1) + ' worms is: ' + str(highest_mover[0]) + '\n')
                
                # write data for highest mover
                header = ["cX", "cY", "Disp", "Area", "ContLeng", "Ratio"]
                data = [(b.cx, b.cy, b.disp, b.area, cv2.arcLength(b.contour, True), cv2.arcLength(b.contour, True)/b.area) for b in track[0].blobs[call_id].data]
                file = open(os.path.join(path, str(cnt + 1) + '_out.csv'), 'w+', newline = '')
                with file:
                    write = csv.writer(file)
                    write.writerow(header)
                    write.writerows(data)
                
                file.close()
                
                disp = [(b.disp) for b in track[0].blobs[call_id].data]
                count = 0
                ave = 0
                for d in disp:
                    count += 1
                    ave += d
                camera_disp_ave.append([cnt + 1, ave/count])
            else:
                print('Warning: No blobs identified on ' + 'Plate ' + str(cnt + 1))
                log.append('Warning: No blobs identified on ' + 'Plate ' + str(cnt + 1) + '\n')
                main_log.append('Warning: No blobs identified on ' + 'Plate ' + str(cnt + 1) + '\n')

            cnt += 1
        
        if specific_blob_output:
            camera_disp_spe = []
            cnt = 1
            for track in tracker:
                line = blob_file.readline()
                
                count_movers = 0
                for blob in track[0].blobs:
                    count_movers += 1
            
                if count_movers == 0 or 'NaN' in line:
                    camera_disp_spe.append([cnt, -1])
                else:
                    call_id = int(line) - 1
                    
                    disp = [(b.disp) for b in track[0].blobs[call_id].data]
                    count = 0
                    ave = 0
                    for d in disp:
                        count += 1
                        ave += d
                    camera_disp_spe.append([cnt, ave/count])

                cnt += 1
           
            cage_disp_spe.append([camera[0], camera_disp_spe])
        
        cage_disp_ave.append([camera[0], camera_disp_ave])
        
        print(str(time.time() - StartTimeCam))
        print()
        
        main_log.append('VideoRuntime: ' + str(time.time() - StartTimeCam) + ' sec\n')
        main_log.append('\n')
        
        # write log file
        runtime = str(time.time() - StartTimeCam)
        log.append('SecRuntime: ' + runtime + '\n')
        file = open(os.path.join(path, 'log.txt'), 'w')
        file.writelines(log)
        file.close()
    
    run_disp_ave.append([cage[0], cage_disp_ave])
    
    if specific_blob_output:
        run_disp_spe.append([cage[0], cage_disp_spe])

if specific_blob_output:
    blob_file.close()

# output average movement file for all top movers
file = open(os.path.join(main_directory, 'average_disp_auto.csv'), 'w+', newline = '')
with file:
    writer = csv.writer(file)
    header = ["ID", "AveDisp"]
    writer.writerow(header)
    for cage in run_disp_ave:
        for camera in cage[1]:
            for worm in camera[1]:
                ID = 'w' + str(cage[0]) + '-' + str(camera[0]) + '-' + str(worm[0])
                rowtemp = [ID, worm[1]]
                writer.writerow(rowtemp)
file.close()

# output average movement file for pre-specified blobs
if specific_blob_output:
    file = open(os.path.join(main_directory, 'average_disp_specific.csv'), 'w+', newline = '')
    with file:
        writer = csv.writer(file)
        header = ["ID", "AveDisp"]
        writer.writerow(header)
        for cage in run_disp_spe:
            for camera in cage[1]:
                for worm in camera[1]:
                    ID = 'w' + str(cage[0]) + '-' + str(camera[0]) + '-' + str(worm[0])
                    rowtemp = [ID, worm[1]]
                    writer.writerow(rowtemp)
file.close()

runtime = str(time.time() - StartTime)
print('Total Runtime: ' + runtime)
main_log.append('TotalRuntime: ' + runtime + ' sec\n')

file = open(os.path.join(main_directory, 'main_log.txt'), 'w')
file.writelines(main_log)
file.close()