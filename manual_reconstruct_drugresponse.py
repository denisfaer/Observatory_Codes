""" import required libraries """

from dataclasses import dataclass
import operator
import os
import cv2
import math
import csv
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
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
    ConfAdaptiveThreshold(21, 3.5),
    ConfBinarize(3),
    ConfContour(100, 250, 0.4, 0.5)) # area_min, area_max, perim-to-area ratio min, perim-to-area ratio max


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


def manual_reconstruct(trajectory):
    data = []
    act = 0
    frames = 0
    last_end = 0

    while len(trajectory) > 0:
        blobID = trajectory[0]
        trajectory.pop(0)
        
        # find overlapping period
        end_time = 0
        start_time = 999_999_999_000
        
        if len(trajectory) > 0:
            # get timestamp of last frame current ID was used
            for move in mover:
                if move[0] == blobID:
                    end_time = move[4] / 1000
                    break

            # get timestamp of next ID appearing
            nextID = trajectory[0]
            for move in mover:
                if move[0] == nextID:
                    start_time = move[3] / 1000
                    break
            
            if trouble:
                print('Blob ' + str(blobID) + ' ends at ' + str(end_time) + ', Blob ' + str(nextID) + ' starts at ' + str(start_time) + '; Overlap: ' + str(end_time - start_time))
        
        data_temp = [[b.time / 1000, b.cx, b.cy, b.disp, b.area, cv2.arcLength(b.contour, True) / b.area , blobID] for b in track[0].blobs[blobID - 1].data]
        
        cnt = 0
        lasttime = data_temp[0][0]
        lastx = data_temp[0][1]
        lasty = data_temp[0][2]
        normalized_data = []
        
        for temp in data_temp:
            cnt += 1
            if cnt == 1:
                normalized_data.append(temp)
                continue
            temp[3] = math.sqrt((temp[1] - lastx) ** 2 + (temp[2] - lasty) ** 2) / (temp[0] - lasttime)
            normalized_data.append(temp)
            lasttime = temp[0]
            lastx = temp[1]
            lasty = temp[2]
        
        for temp in normalized_data:
            if temp[0] > last_end and temp[0] < start_time:
                data.append(temp)
                act += temp[3]
                frames += 1
        
        last_end = end_time
    
    return data, act, frames


def count_squares(data, hourlim = 999):
    count = 0
    squares = np.zeros((square_fine, square_fine))
    x0 = track[1] - track[3]
    y0 = track[2] - track[3]
    side = 2 * track[3] / square_fine
    
    for point in data:
        if (video_duration - point[0]) / 3600 <= hourlim:
            i = math.floor((point[1] - x0) / side)
            j = math.floor((point[2] - y0) / side)
            squares[i, j] = 1
    
    for i in range(square_fine):
        for j in range(square_fine):
            count += squares[i,j]
                
    return squares, count


""" execute data analysis """

# establish folder naming scheme
cages = [[1, "Cage 1"], [2, "Cage 2"], [3, "Cage 3"], [4, "Cage 4"]]
cameras = [[1, "Camera 1"], [2, "Camera 2"], [3, "Camera 3"], [4, "Camera 4"]]

main_directory = os.getcwd()
global_cnt = 0
trouble = False # print troubleshooting messages
legends = False # add color-blob ID legends to matplot panels
square_fine = 10 # square exploration grid side size

lastdata = True # output results only for last X hours
lasthours = 2 # last X hours to save

run_avedisp = []
if lastdata:
    run_lastdisp = []

globcnt = 0

# reconstruct series of IDs that tracked worms
for cage in cages:
    if len(cages) > 1:
        cage_directory = os.path.join(main_directory, cage[1])
    else:
        cage_directory = main_directory
    
    cage_avedisp = []
    if lastdata:
        cage_lastdisp = []
    
    for camera in cameras:
        if len(cameras) > 1:
            camera_directory = os.path.join(cage_directory, camera[1])
        else:
            camera_directory = cage_directory
        
        camera_avedisp = []
        if lastdata:
            camera_lastdisp = []
        
        # load pickled trackers for this camera
        path = os.path.join(camera_directory, 'Output')
        pickle_file = open(os.path.join(path, 'trackers.pkl'), 'rb')
        tracker = pickle.load(pickle_file)
        pickle_file.close()
        
        ID_file = open(os.path.join(path, 'manual.txt'), 'rb')
        
        cnt = 0
        globcnt += 1
        plt.figure(globcnt)
        
        for track in tracker:
            cnt += 1
            print('Plate '+ str(cage[0]) + '-' + str(camera[0]) + '-' + str(cnt))
            
            # skip this plate if no blobs were identified
            if len(track[0].blobs) == 0:
                print('WARNING: No blobs identified')
                continue
            
            plt.subplot(2, 3, cnt)
            
            # generate summarry movement statistics and plots for all blobs
            mover = []
            blob_plot_labels = []
            video_duration = 0
            
            for blob in track[0].blobs:
                sumr = 0
                vectr = 0
                times = 0
                time_start = blob.data[0].time
                x0 = blob.data[0].cx
                y0 = blob.data[0].cy
                lasttime = time_start
                for timer in blob.data:
                    if timer.time != lasttime:
                        sumr += timer.disp * 1000 / (timer.time - lasttime)
                    else:
                        sumr += timer.disp
                    vectr += math.sqrt((timer.cx - x0) ** 2 + (timer.cy - y0) ** 2)
                    times += 1
                    lasttime = timer.time
                time_end = timer.time
                if timer.time > video_duration:
                    video_duration = timer.time
                mover.append([blob.id, sumr, times, time_start, time_end, vectr / times])
                vec = vectr / times
                
                blob_vec = []
                blob_time = []
                for timer in blob.data:
                    blob_vec.append(vec)
                    blob_time.append(timer.time / 1000)
                
                if legends:
                    blob_plot_labels.append(blob.id)
                plt.plot(blob_time, blob_vec)
            
            video_duration = video_duration / 1000
            
            plt.legend(blob_plot_labels)
            plt.title('Plate ' + str(cage[0]) + '-' + str(camera[0]) + '-' + str(cnt))
            
            # sort movers and find highest
            indexed_mover = enumerate([k[5] for k in mover])
            sorted_mover = sorted(indexed_mover, key = operator.itemgetter(1), reverse = True)
            
            # read manually identified starting worm ID
            line = ID_file.readline()
            line = str(line)
            if "NaN" in line:
                continue
            
            forw = "b'"
            backw = "\\r\\n"
            endw = "'"
            temps1 = line.replace(forw, "")
            temps2 = temps1.replace(backw, "")
            temps = temps2.replace(endw, "")
            worm_trajectory = [int(s) for s in temps.split(',')] # holds a list of manually defined worm blobIDs
            if trouble:
                print(worm_trajectory)
            merged_data, totdisp, frames = manual_reconstruct(worm_trajectory) # reconstructs data from manually-defined blobIDs
            
            # write plate-specific files
            header = ["Time", "cX", "cY", "Disp", "Area", "ContRat", "BlobID"]
            file = open(os.path.join(path, str(cnt) + '_out.csv'), 'w+', newline = '')
            with file:
                write = csv.writer(file)
                write.writerow(header)
                write.writerows(merged_data)
            file.close()
            
            # generate output data only for last X hours
            if lastdata:
                last_totdisp = 0
                last_frames = 0
                last_square_arr, last_square_cnt = count_squares(merged_data, lasthours)
                for data in merged_data:
                    if (video_duration - data[0]) / 3600 <= lasthours:
                        last_totdisp += data[3]
                        last_frames += 1        
                
                camera_lastdisp.append([cnt , last_totdisp, last_frames, last_square_cnt]) # save average displacement for global file output
                
            square_arr, square_cnt = count_squares(merged_data)
            camera_avedisp.append([cnt , totdisp, frames, square_cnt]) # save average displacement for global file output
            
            # output average displacement for all blobs
            header = ["BlobID", "TotalDisp", "Frames", "AveDisp"]
            data = []
            for worm in mover:
                data.append([worm[0], worm[1], worm[2], worm[1] / worm[2]])
            file = open(os.path.join(path, str(cnt) + '_average_disp.csv'), 'w+', newline = '')
            with file:
                write = csv.writer(file)
                write.writerow(header)
                write.writerows(data)
            file.close()
            
            
        cage_avedisp.append([camera[0], camera_avedisp])
        if lastdata:
            cage_lastdisp.append([camera[0], camera_lastdisp])
        
        ID_file.close()

    
    run_avedisp.append([cage[0], cage_avedisp])
    if lastdata:
        run_lastdisp.append([cage[0], cage_lastdisp])


# write global movement averages
file = open(os.path.join(main_directory, 'average_disp.csv'), 'w+', newline = '')
with file:
    writer = csv.writer(file)
    header = ["wormID", "TotalDisp", "AveDisp", "Squares", "Frames"]
    writer.writerow(header)
    for cage in run_avedisp:
        for camera in cage[1]:
            for worm in camera[1]:
                ID = 'w' + str(cage[0]) + '-' + str(camera[0]) + '-' + str(worm[0])
                rowtemp = [ID, worm[1], worm[1] / worm[2], worm[3], worm[2]]
                writer.writerow(rowtemp)    
file.close()

# write movement averages for last X hours
if lastdata:
    file = open(os.path.join(main_directory, 'last_'+ str(lasthours) + 'h_disp.csv'), 'w+', newline = '')
    with file:
        writer = csv.writer(file)
        header = ["wormID", "TotalDisp", "AveDisp", "Squares", "Frames"]
        writer.writerow(header)
        for cage in run_lastdisp:
            for camera in cage[1]:
                for worm in camera[1]:
                    if worm[2] > 0:
                        ID = 'w' + str(cage[0]) + '-' + str(camera[0]) + '-' + str(worm[0])
                        rowtemp = [ID, worm[1], worm[1] / worm[2], worm[3], worm[2]]
                        writer.writerow(rowtemp)        
    file.close()

runtime = str(time.time() - StartTime)
print('Total Runtime (sec): ' + runtime)