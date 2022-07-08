import cv2
import sys
import glob

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':
    
    # Set up tracker
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[1] # MOSSE

    print('Setup tacker ...')
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    print('Tracker set Done')    
    # Read images
    imdir = 'DragonBaby/DragonBaby/img/'
    ext = ['jpg']

    files = []
    [files.extend(glob.glob(imdir + '*' + e)) for e in ext]
    images = [cv2.imread(file) for file in files]
    
    # Define an initial bounding box
    bbox = (160,83,56,65)
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(images[0], bbox)
    tracking_rec = []
    for i in range(1, len(images)):
        # Update tracker
        print('Start traking ... frame: {}'.format(i))
        ok, bbox = tracker.update(images[i])

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(images[i], p1, p2, (255,0,0), 2, 1)
            tracking_rec.append(bbox)
        else:
            # Tracking failure
            cv2.putText(images[i], "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        # Display tracker type on frame
        cv2.putText(images[i], tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
        
        # Display result
        cv2.imshow("Tracking", images[i])

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    print('Write file ...')
    with open('tracking_rect.txt', 'w') as f:
        for track in tracking_rec:
            f.write(str(track[0]) + ',' + str(track[1]) + ',' + str(track[2]) + ',' + str(track[3]))
            f.write('\n')
    print('Write Done')
        