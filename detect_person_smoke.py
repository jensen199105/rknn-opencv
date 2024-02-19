import time
import numpy as np
import cv2
import datetime
from PIL import Image, ImageDraw, ImageFont
from rknnlite.api import RKNNLite


RKNN_MODEL = 'weights/yolov5.rknn'
IMG_PATH = 'data/1.jpg'
VEDIO_PATH='data/demo.mp4'
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 960

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])
    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = sigmoid(input[..., 5:])
    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(rknn, frame):
    img, ratio, (dw, dh) = letterbox(frame, new_shape=(IMG_SIZE, IMG_SIZE))
    frame = img
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    outputs = rknn.inference(inputs=[img])
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]
    input0_data = outputs[0]
    input1_data = outputs[1]
    input2_data = outputs[2]
    input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
    input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
    input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))
    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)
    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
    if not nclasses and not nscores:
        return None, None, None
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def perform_object_detection(rknn, frame, region_coords):
    region_x1, region_y1, region_x2, region_y2 = region_coords
    # cv2.rectangle(frame, (region_x1, region_y1), (region_x2, region_y2), (255, 0, 0), 2)

    # Draw the specified region on the frame
    class_list = []
    x_y_w_h = []

    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)
    
    boxes, classes, scores = yolov5_post_process(rknn, frame)
    class_names = ["person", "smoke",'stool']
    class_colors = {'person': (0, 0, 255), 'smoke': (255, 0, 0),'stool':(0,255,0)}
    result_person = '人物入侵'
    result_smoke = '正在吸烟'

    count=0
    count_stool=0
    label_l=[]
    try:
        if len(boxes) == 0:
            return np.array(frame_pil),x_y_w_h
    except:
        return np.array(frame_pil),x_y_w_h
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        score = scores[index]
        label = classes[index]

        class_name = class_names[label]
        if class_name=='person':
            label_l.append(0)
        if class_name=='smoke':
            label_l.append(1)
        if class_name=='stool':
            label_l.append(2)
        color = class_colors.get(class_name, (0, 0, 0))

        if region_x1 <= x1 <= region_x2 and region_y1 <= y1 <= region_y2 and \
            region_x1 <= x2 <= region_x2 and region_y1 <= y2 <= region_y2:
            # result = '该区域有异物'  # Update result when a box is found
            # break  # Break the loop once a box is found
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            font = ImageFont.truetype('SimHei.ttf', 25)
            text = f"{class_name} {score: .2f}"
            draw.text((x1, y1 - 20), text, fill=color, font=font)
        else:
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            font = ImageFont.truetype('SimHei.ttf', 25)
            text = f"{class_name} {score: .2f}"
            draw.text((x1, y1 - 20), text, fill=color, font=font)

        class_list.append(label)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        x_y_w_h.append([x_center, y_center, width])
    for i in label_l:
        if i==0:
            # font = ImageFont.truetype('SimHei.ttf', 20)
            # draw.text((region_x1+100, 0), result_person, fill=(0, 0, 255), font=font)
            count+=1
        if i==1:
            #吸烟字体大小设置
            font = ImageFont.truetype('SimHei.ttf', 30)
            draw.text((region_x1+200, 0), result_smoke, fill=(0, 255, 0), font=font)
        if i==2:
            count_stool+=1
    if count_stool<2:
        nums=2-count_stool
        # 凳子字体大小设置
        font = ImageFont.truetype('SimHei.ttf', 30)
        draw.text((region_x1 + 350, 0), '缺少凳子:{}个'.format(nums), fill=(0, 255, 0), font=font)
    font = ImageFont.truetype('SimHei.ttf', 30)
    draw.text((region_x1 + 550, 0), '凳子数量:{}个'.format(count_stool), fill=(0, 255, 0), font=font)
    #人数字体大小设置
    font = ImageFont.truetype('SimHei.ttf', 30)
    draw.text((0, 0), '人数:'+str(count), fill=(0, 255, 0), font=font)
    frame_with_detection = np.array(frame_pil)
    return frame_with_detection,x_y_w_h

def test_vedio(rknn):
    capture = cv2.VideoCapture(0)
    region_coordinates = (50, 50, 500, 500)
    frame_rate,frame_count = 3,0
    while True:
        Capture_TIME = datetime.datetime.now()
        print('开始时间:', Capture_TIME.strftime("%Y-%m-%d %H:%M:%S.%f"))
        ref, frame = capture.read()
        if not ref:
            break 
        
        if (frame_count % frame_rate == 0):
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            frame_with_detection, x_y_w_h = perform_object_detection(rknn, frame, region_coordinates)
            frame_with_detection = cv2.resize(frame_with_detection, (frame.shape[1], frame.shape[0]))
        else:
            frame_with_detection = frame
            
        frame_count += 1
        cv2.imshow("Press q to end", frame_with_detection)
        cv2.waitKey(1)  # Add this line to wait for 1 millisecond to ensure the image window has enough time to display
        Capture_TIME = datetime.datetime.now()
        print('结束时间:', Capture_TIME.strftime("%Y-%m-%d %H:%M:%S.%f"))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Video Detection Done!")

def test_image(rknn):
    frame = cv2.imread(IMG_PATH)
    boxes, classes, scores = yolov5_post_process(rknn, frame)

def read_vedio(rknn):
    capture = cv2.VideoCapture("rtsp://admin:Tsjg@123456@192.168.1.253:554/Streaming/Channels/101")
    region_coordinates = (50, 50, 500, 500)

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter('saved_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_rate,frame_count = 3,0
    while True:
        start_time = datetime.datetime.now()
        ref, frame = capture.read()
        if not ref:
            continue 
            
        if (frame_count % frame_rate == 0):   
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            frame_with_detection, x_y_w_h = perform_object_detection(rknn, frame, region_coordinates)
            frame_with_detection = cv2.resize(frame_with_detection, (width, height))
        else:
            frame_with_detection = frame

        video_writer.write(frame_with_detection)
        end_time = datetime.datetime.now() 
        delta_time = end_time - start_time 
        print('耗时: {}, 帧率:{}'.format(delta_time.total_seconds(), 1000 / delta_time.total_seconds()))
    video_writer.release()    

if __name__ == '__main__':
    rknn = RKNNLite()

    print('--> Load RKNN model')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    
    #test_vedio(rknn)
    #test_image(rknn)
    read_vedio(rknn)
    
