import numpy as np
import serial
import threading
import cv2
import time
from scipy.ndimage import gaussian_filter, convolve
import rospy
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import PoseStamped
import tf.transformations as tf_trans
from std_msgs.msg import Bool

contact_data_norm = np.zeros((16, 8))
median = np.zeros((16, 8))
flag = False
reinit_flag = False
lock = threading.Lock()
MASK_BOTTOM_ROWS = False
MASK_BOTTOM_LEFT_CORNER = False  
ground_truth_angle = None  

# THRESHOLD = 9
THRESHOLD = 7
NOISE_SCALE = 70
WINDOW_WIDTH = contact_data_norm.shape[1] * 30
WINDOW_HEIGHT = contact_data_norm.shape[0] * 30

cv2.namedWindow("Finger 1 Contact", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Finger 1 Contact", WINDOW_WIDTH, WINDOW_HEIGHT)

def pose_callback(msg):
    global ground_truth_angle
    try:
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        
        # print(f"[DEBUG] Received pose - qx:{qx:.4f}, qy:{qy:.4f}, qz:{qz:.4f}, qw:{qw:.4f}")
        
        euler = tf_trans.euler_from_quaternion([qx, qy, qz, qw])
        yaw = euler[2] 
        
        ground_truth_angle = yaw * 180.0 / np.pi
        # print(f"[DEBUG] Ground truth angle calculated: {ground_truth_angle:.2f} degrees")
    except Exception as e:
        rospy.logwarn(f"Error processing pose message: {e}")

def readThread(serDev):
    global contact_data_norm, median, flag, reinit_flag

    current = []
    backup = None

    def initialize_median():
        nonlocal current, backup
        data_tac = []
        current = []
        num = 0
        print("[INFO] Sampling 30 frames for baseline median...")

        while num < 30 and not rospy.is_shutdown():
            if serDev.in_waiting > 0:
                try:
                    line = serDev.readline().decode('utf-8').strip()
                except:
                    continue
                if len(line) < 10:
                    if current is not None and len(current) == 16:
                        backup = np.array(current)
                        data_tac.append(backup)
                        num += 1
                    current = []
                else:
                    values = line.split()
                    if len(values) >= 8:
                        row = [int(val) for val in values[:8]]
                        current.append(row)

        if len(data_tac) == 30:
            with lock:
                med = np.median(np.array(data_tac), axis=0)
                median[:, :] = med
                print("[INFO] Median updated.")
            return True
        return False

    if initialize_median():
        flag = True

    while not rospy.is_shutdown():
        if reinit_flag:
            if initialize_median():
                flag = True
            with lock:
                reinit_flag = False
            continue

        if serDev.in_waiting > 0:
            try:
                line = serDev.readline().decode('utf-8').strip()
            except:
                continue

            if len(line) < 10:
                if current is not None and len(current) == 16:
                    backup = np.array(current)
                current = []
                if backup is not None:
                    with lock:
                        contact_data = backup - median - THRESHOLD
                    contact_data = np.clip(contact_data, 0, 100)
                    if np.max(contact_data) < THRESHOLD:
                        contact_data_norm = contact_data[:, :8] / NOISE_SCALE
                    else:
                        contact_data_norm = contact_data[:, :8] / np.max(contact_data)
                continue

            values = line.split()
            if len(values) >= 8:
                row = [int(val) for val in values[:8]]
                current.append(row)

def apply_gaussian_blur(contact_map, sigma=0.1):
    return gaussian_filter(contact_map, sigma=sigma)

def temporal_filter(new_frame, prev_frame, alpha=0.1):
    return alpha * new_frame + (1 - alpha) * prev_frame

def resize_contact_map(contact_map, scale=3):
    h, w = contact_map.shape
    return cv2.resize(contact_map, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

def filter_small_values(contact_map, threshold=0.1, min_neighbors=3):
    kernel = np.ones((3, 3))
    binary_map = (contact_map > threshold).astype(np.float32)
    neighbor_count = convolve(binary_map, kernel, mode='constant', cval=0)
    return np.where(neighbor_count >= min_neighbors, contact_map, 0)

def filter_islands(image, min_size=7):
    # image: uint8, shape (16,8), 0~255
    binary = (image > 30).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            mask[labels == i] = 1
    return image * mask

def filter_largest_island(image):
    # image: uint8, shape (16,8), 0~255
    binary = (image > 30).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
    if num_labels <= 1:
        return image  
    max_score = -1
    best_label = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        mask = (labels == i)
        max_val = image[mask].max() if np.any(mask) else 0
        score = area * max_val 
        if score > max_score:
            max_score = score
            best_label = i
    mask = (labels == best_label)
    out = np.zeros_like(image)
    out[mask] = image[mask]
    return out

def estimate_angle(image):
    # image: uint8, shape (16,8), 0~255
    _, binary = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(binary > 0)
    if len(xs) < 8:
        return None, None
    points = np.column_stack((xs, ys))  # (x, y)
    [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan2(vy, vx) * 180 / np.pi
    return float(angle), tuple(v.item() for v in (vx, vy, x0, y0))

PORT = '/dev/ttyUSB0'
BAUD = 2000000
serDev = serial.Serial(PORT, BAUD)
serDev.flush()

serialThread = threading.Thread(target=readThread, args=(serDev,))
serialThread.daemon = True
serialThread.start()

if __name__ == '__main__':
    rospy.init_node('contact_image_publisher', anonymous=True)
    list_pub = rospy.Publisher('/finger1/contact_image_list', Float32MultiArray, queue_size=10)
    sum_pub = rospy.Publisher('/finger1/contact_sum', Float32, queue_size=10)
    
    pose_sub = rospy.Subscriber('/aruco_delta', PoseStamped, pose_callback)
    
    def array_refresh_callback(msg):
        global reinit_flag
        if msg.data:
            print("[SIGNAL] Received array refresh signal: Triggering reinitialization.")
            with lock:
                reinit_flag = True
    
    refresh_sub = rospy.Subscriber('/array_refresh_signal', Bool, array_refresh_callback)

    prev_frame = np.zeros_like(contact_data_norm)

    print('[ROS] Contact data publisher started. Press F to reinitialize median.')
    print('[ROS] Array refresh signal subscribed. Send True to /array_refresh_signal to trigger reinitialization.')

    while not rospy.is_shutdown():
        for _ in range(300):
            if flag:
                with lock:
                    temp_filtered = temporal_filter(contact_data_norm, prev_frame)
                prev_frame = temp_filtered

                filtered = filter_small_values(temp_filtered, threshold=0.05, min_neighbors=4)
                norm_data = np.clip(filtered, 0, 1)
                image_16x8 = (norm_data * 255).astype(np.uint8)
                image_16x8 = filter_islands(image_16x8, min_size=5)
                if MASK_BOTTOM_ROWS:
                    image_16x8[-14:, :] = 0
                if MASK_BOTTOM_LEFT_CORNER:
                    image_16x8[7:, :3] = 0  
                    # make the last 5 rows to be 0
                    image_16x8[-6:, :] = 0
                image_for_angle = filter_largest_island(image_16x8)

                list_msg = Float32MultiArray(data=image_16x8.flatten().tolist())
                list_pub.publish(list_msg)

                contact_sum = np.sum(norm_data).astype(np.float32)
                sum_pub.publish(Float32(data=contact_sum))

                vis_data = resize_contact_map(image_16x8, scale=30)
                vis_colored = cv2.applyColorMap(vis_data, cv2.COLORMAP_VIRIDIS)
                
                if ground_truth_angle is not None:
                    gt_text = f"GT Angle: {ground_truth_angle:.1f} deg"
                    # cv2.putText(vis_colored, gt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                #     print(f"[DEBUG] Displaying ground truth angle: {ground_truth_angle:.2f}")
                # else7
                #     print("[DEBUG] ground_truth_angle is None")
                
                angle, line_params = estimate_angle(image_for_angle)
                if angle is not None and line_params is not None:
                    text = f"Est. Angle: {angle:.1f} deg"
                    # cv2.putText(vis_colored, text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    
                    vx, vy, x0, y0 = line_params
                    scale = 30
                    length = 8
                    pt1 = (int((x0 - vx * length) * scale), int((y0 - vy * length) * scale))
                    pt2 = (int((x0 + vx * length) * scale), int((y0 + vy * length) * scale))
                    # cv2.line(vis_colored, pt1, pt2, (255,255,255), 2)
                cv2.imshow("Finger 1 Contact", vis_colored)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('f') or key == ord('F'):
                    print("[KEY] F pressed: Triggering reinitialization.")
                    with lock:
                        reinit_flag = True
            time.sleep(0.01)
