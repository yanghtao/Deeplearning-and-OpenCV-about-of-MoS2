import os
import time
import json
import math
import cv2
import torch

import numpy as np

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="opencv predict MoS2 twist angel")

    # exclude background
    parser.add_argument("--image-path", default="./MoS2_669.png", help="predicting image path")
    # list of name of classes
    thickness_dict = ['1L', '2L', 'ML', 'TL']
    parser.add_argument("--thickness-dict", default=thickness_dict, help="Plot the predicted image RGB values")
    # 选择绘制标签在原图或者语义分割结果图，并输出最终结果
    parser.add_argument("--result-image", default=False, type=bool, help="draw label in origin image or segmentation image")
    # 选择输出结果图像类型，如果想输出三角图像，则选择True，否则输出带标记的结果图像，不显示三角
    parser.add_argument("--result-type", default=True, type=bool, help="optional result image type")
    # 选择是否获得预测结果每层的单独图像，并存储
    parser.add_argument("--get-layers", default=True, type=bool, help="get every layers of predict result image")
    # 选择是否绘制拟合三角形内角信息到结果图上
    parser.add_argument("--draw-interior", default=False, type=bool, help="Draw the internal angle of triangle to the result diagram")
    # 选择检测所有三角还是检测扭角信息
    parser.add_argument("--predict-all-triangle", default=False, type=bool, help="Whether to detect all triangles or torsional angle information")

    args = parser.parse_args()
    return args

# 创建一个函数，用于获取最左侧两个点,输入的为三角形的点集合
def get_left_point(Points):
    if (Points[1][0] < Points[0][0]):
        if (Points[1][0] < Points[2][0]):
            point_left1 = Points[1]
            point_left2 = Points[2]if(Points[2][0] < Points[0][0])else Points[0]
        else:
            point_left1 = Points[2]
            point_left2 = Points[1]
    else:
        if (Points[0][0] < Points[2][0]):
            point_left1 = Points[0]
            point_left2 = Points[2] if (Points[2][0] < Points[1][0]) else Points[1]
        else:
            point_left1 = Points[2]
            point_left2 = Points[0]
    return point_left1, point_left2


# 创建一个函数，绘制每个拟合三角形的内角角度，显示在结果图上
def draw_interior_angle(draw_img, Points):
    angle_a1 = math.atan2(-(Points[2][1] - Points[0][1]),
                          (Points[2][0] - Points[0][0])) * 180.0 / np.pi
    angle_b1 = math.atan2(-(Points[1][1] - Points[0][1]),
                          (Points[1][0] - Points[0][0])) * 180.0 / np.pi

    angle_a2 = math.atan2(-(Points[0][1] - Points[1][1]),
                          (Points[0][0] - Points[1][0])) * 180.0 / np.pi
    angle_b2 = math.atan2(-(Points[2][1] - Points[1][1]),
                          (Points[2][0] - Points[1][0])) * 180.0 / np.pi

    angle_a3 = math.atan2(-(Points[1][1] - Points[2][1]),
                          (Points[1][0] - Points[2][0])) * 180.0 / np.pi
    angle_b3 = math.atan2(-(Points[0][1] - Points[2][1]),
                          (Points[0][0] - Points[2][0])) * 180.0 / np.pi

    # 计算主轴的角度

    angle1 = (-angle_a1) if (angle_a1 <= 0) else (360 - angle_a1)
    angle2 = (-angle_a2) if (angle_a2 <= 0) else (360 - angle_a2)
    angle3 = (-angle_a3) if (angle_a3 <= 0) else (360 - angle_a3)

    # 计算圆弧的结束角度

    end_angle1 = (angle_a1 - angle_b1) if (angle_b1 < angle_a1) else (
            360 + (angle_a1 - angle_b1))
    end_angle2 = (angle_a2 - angle_b2) if (angle_b2 < angle_a2) else (
            360 + (angle_a2 - angle_b2))
    end_angle3 = (angle_a3 - angle_b3) if (angle_b3 < angle_a3) else (
            360 + (angle_a3 - angle_b3))
    cv2.ellipse(draw_img, (int(Points[0][0]), int(Points[0][1])), (9, 9), angle1,
                0, end_angle1, (255, 0, 0), 2,cv2.LINE_AA)
    cv2.ellipse(draw_img, (int(Points[1][0]), int(Points[1][1])), (9, 9), angle2,
                0, end_angle2, (255, 0, 0), 2,cv2.LINE_AA)
    cv2.ellipse(draw_img, (int(Points[2][0]), int(Points[2][1])), (9, 9), angle3,
                0, end_angle3, (255, 0, 0), 2,cv2.LINE_AA)

    # 在图片上绘制各个角的大小
    a1 = round(end_angle1, 2)
    a2 = round(end_angle2, 2)
    a3 = round(end_angle3, 2)
    cv2.putText(draw_img, str(a1),
                (int(Points[0][0] -5), int(Points[0][1]-5)),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1,cv2.LINE_AA)
    cv2.putText(draw_img, str(a2),
                (int(Points[1][0]) - 5, int(Points[1][1]) - 5),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1,cv2.LINE_AA)
    cv2.putText(draw_img, str(a3),
                (int(Points[2][0]) - 5, int(Points[2][1]) - 5),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1,cv2.LINE_AA)
    return draw_img


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


# 计算两点间线段长度
def get_lenth(point_x,point_y):
    value = np.sqrt((point_x[0]-point_y[0])**2 + (point_x[1]-point_y[1])**2)
    return value

# 获取拟合三角形的角度等信息
def get_triangle_infro(Points):
    # 获取各个边的边长
    a = get_lenth(Points[0], Points[1])
    b = get_lenth(Points[1], Points[2])
    d = get_lenth(Points[2], Points[0])
    # 边长获取完成，开始处理角度问题//三角形余弦定理为cosA=(b^2+c^2-a^2)/2bc;cosB=(a^2+c^2-b^2)/2ac;cosC=(b^2+a^2-c^2)/2ab
    A = math.acos((b * b + d * d - a * a) / (2 * d * b)) * 180.0 / np.pi
    B = math.acos((a * a + d * d - b * b) / (2 * a * d)) * 180.0 / np.pi
    C = math.acos((a * a - d * d + b * b) / (2 * a * b)) * 180.0 / np.pi
    return A, B, C


# 获取扭角角度值
def get_twist_angel(args):
    #   根据语义分割预测结果图片,test_result.png来检测扭角角度。
    imgpath_mask = 'test_result.png'

    img = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)  # 读取png图像带参数；
    img2 = cv2.imread(imgpath_mask, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (512, 512))
    img2 = cv2.resize(img2, (512, 512))
    out = cv2.addWeighted(img, 1, img2, 0.3, 1)  # mask图像与预测图像混合


    # 绘制到原图，得到结果
    if(args.result_image):
        result_img = img
    else:
        result_img = img2
    # 设置颜色提取范围
    # # 背景
    # lower_index5 = np.array([0, 0, 0])
    # upper_index5 = np.array([0, 0, 0])
    # color_5 = [lower_index5, upper_index5]
    # 第一层颜色
    lower_index3 = np.array([0, 100, 100])
    upper_index3 = np.array([0, 130, 130])
    color_1 = [lower_index3, upper_index3]
    # 第二层颜色
    lower_index = np.array([0, 0, 120])
    upper_index = np.array([0, 0, 130])
    color_2 = [lower_index, upper_index]
    # 第三层颜色
    lower_index2 = np.array([0, 120, 0])
    upper_index2 = np.array([0, 130, 0])
    color_4 = [lower_index2, upper_index2]
    # 第四层颜色
    lower_index = np.array([100, 0, 0])
    upper_index = np.array([130, 0, 0])
    color_3 = [lower_index, upper_index]
    # 提取颜色
    Inrange_color = [color_1, color_2, color_3, color_4]
    test_img = np.zeros((512, 512, 3), np.uint8)
    i = 0
    box_color = [(0, 150, 255), (0, 220, 200), (255, 150, 155), (180, 150, 250)]
    color_dict = [(0, 150, 255), (0, 220, 200), (255, 150, 155), (180, 150, 250)]
    conturn_img = np.zeros((512, 512, 3), np.uint8)
    for co in Inrange_color:

        mask = cv2.inRange(img2, co[0], co[1])

        # 实现形态学操作中的闭操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        mask_layer_3 = cv2.merge([binary, binary, binary])
        ROI = cv2.bitwise_and(img2, mask_layer_3)  # 图形与mask的and运算，如果
        # 遍历ROI，当背景为[0,0,0]时，转为[255,255,255]
        ROI_array = np.array(ROI)

        # 显示不同层的图像
        # 在显示不同层的图像时进行翻转，黑的变白色
        if args.get_layers:
            for h_i in range(img.shape[1] - 1):
                for j in range(img.shape[1] - 1):
                    if ((np.array(ROI_array[h_i, j])[0].all() == np.array([0, 0, 0])[0].all()) and (
                            np.array(ROI_array[h_i, j])[1].all() == np.array([0, 0, 0])[1].all()) and (
                            np.array(ROI_array[h_i, j])[2].all() == np.array([0, 0, 0])[2].all())):
                        ROI_array[h_i, j] = [255, 255, 255]
            cv2.imwrite('layers_{}.png'.format(i + 1), ROI_array)   # 保存每层识别的结果图

        bbox_list = []
        # 使用轮廓寻找函数来对轮廓进行提取
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制所有轮廓图到一张conturn.img上

        cv2.drawContours(conturn_img, contours, -1, color_dict[i], 2)
        # 变成白底

        test_img2 = np.zeros((512, 512, 3), np.uint8)
        for c in contours:
            # 测试找质心
            M = cv2.moments(c)
            cX = int(M["m10"] / (M["m00"] + float("1e-8")))
            cY = int(M["m01"] / (M["m00"] + float("1e-8")))
            # cv2.circle(img, (cX, cY), 7, (255, 0, 255), -1)

            # 建立一个空白画布

            # 测试结果很好

            # 检测轮廓的面积，如果轮廓面积小于该值，则不进行操作
            Min_area = 15
            if Min_area < cv2.contourArea(c):
                epsilon = 0.04 * cv2.arcLength(c, True)  # 多边形拟合的距离参数，下一个函数要用到。原理见代码后链接
                approx = cv2.approxPolyDP(c, epsilon, True)  # 轮廓近似。将圆润曲线折线化，以此得到该图像的角点坐标。
                corners = len(approx)  # 得到角点数量



                # 如果识别三角型：否则
                if args.result_type:
                    if corners == 3:  # 三个角点的就是三角形
                        area, point_triangle1 = cv2.minEnclosingTriangle(c) # 该函数返回拟合三角形的面积以及三个顶点位置坐标
                        point_triangle1 = np.squeeze(point_triangle1)

                        # 添加判定条件，角度在50-70范围内保存
                        min_angle = 55.0
                        max_angle = 65.0
                        test_list = [co[0], co[1]]
                        # 获取边长，角度等信息
                        A, B, C = get_triangle_infro(point_triangle1)
                        # if(np.array(test_list).all() == (np.array(color_2).all())):   # 绘制所有角度
                        # 从这个地方判断一层的三角,新增一个判定，输出预测三角图像还是输出只标注层数图像
                        # 这里加限制条件，如果识别所有拟合三角并绘制旋转角度，就选择True，否则就检测扭角信息
                        if args.predict_all_triangle:
                            if min_angle < A and min_angle < B and min_angle < C and A < max_angle and B < max_angle and C < max_angle:

                                cv2.line(result_img,
                                         (int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                         (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                         (200, 200, 255), 2, cv2.LINE_8)
                                cv2.line(result_img,
                                         (int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                         (int(point_triangle1[2][0]), int(point_triangle1[2][1])),
                                         (200, 200, 255), 2, cv2.LINE_8)
                                cv2.line(result_img,
                                         (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                         (int(point_triangle1[2][0]), int(point_triangle1[2][1])),
                                         (200, 200, 255), 2, cv2.LINE_8)
                                # 调用绘制拟合三角内角角度值在结果图像上
                                if args.draw_interior:
                                    result_img = draw_interior_angle(result_img, point_triangle1)

                                # # 文章用图，用于绘制所有三角形拟合
                                # if (((np.array(test_list))[0][0].all() == (np.array(color_1))[0][0].all()) and (
                                #         (np.array(test_list))[0][1].all() == (np.array(color_1))[0][1].all())
                                #         and ((np.array(test_list))[0][2].all() == (np.array(color_1))[0][2].all())):
                                #     cv2.drawContours(test_img, [np.array(
                                #         [(int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                #          (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                #          (int(point_triangle1[2][0]), int(point_triangle1[2][1]))])], 0,
                                #                      (0, 0, 255), -1)
                                # else:
                                #     cv2.drawContours(test_img, [np.array(
                                #         [(int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                #          (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                #          (int(point_triangle1[2][0]), int(point_triangle1[2][1]))])], 0,
                                #                      (255, 0, 255), -1)
                                #
                                # for h_i in range(img.shape[1] - 1):
                                #     for j in range(img.shape[1] - 1):
                                #         if ((np.array(test_img[h_i, j])[0].all() ==
                                #              np.array([0, 0, 0])[
                                #                  0].all()) and (
                                #                 np.array(test_img[h_i, j])[1].all() ==
                                #                 np.array([0, 0, 0])[
                                #                     1].all()) and (
                                #                 np.array(test_img[h_i, j])[2].all() ==
                                #                 np.array([0, 0, 0])[
                                #                     2].all())):
                                #             test_img[h_i, j] = [255, 255, 255]
                                # cv2.imwrite("all_tri.png", test_img)




                                # 获取拟合三角形最左侧的两个点
                                point_left1, point_left2 = get_left_point(point_triangle1)
                                # cv2.imshow("temp123", result_img)
                                # cv2.waitKey(0)
                                rotateAngle1 = math.atan2(-(point_left1[1] - point_left2[1]),
                                                          (point_left1[0] - point_left2[0])) * 180.0 / np.pi

                                rotateAngle1 = rotateAngle1 + 180
                                if rotateAngle1 > 180:
                                    rotateAngle1 = rotateAngle1 - 180

                                # 绘制旋转角度

                                rotateAngle1 = round(rotateAngle1, 2)

                                cv2.putText(result_img, str(rotateAngle1),
                                            (int(point_left1[0] + 20), int(point_left1[1] - 10)),
                                            cv2.FONT_HERSHEY_TRIPLEX, 0.8, (100, 255, 255), 1, cv2.LINE_AA)

                                x, y, w, h = cv2.boundingRect(c)
                                if w > 1 and w > 1:  # suppress small bounding boxes (optional)
                                    bbox_list.extend([x + 1, y + 1, w, h])  # MATLAB convention: start from 1 instead of 0

                                    cv2.rectangle(result_img, (x, y), (x + w - 1, y + h - 1), box_color[i], 2,
                                                    cv2.LINE_AA)
                        else:
                            if(((np.array(test_list))[0][0].all() == (np.array(color_1))[0][0].all()) and ((np.array(test_list))[0][1].all() == (np.array(color_1))[0][1].all())
                                    and ((np.array(test_list))[0][2].all() == (np.array(color_1))[0][2].all())):
                                if min_angle < A and min_angle < B and min_angle < C and A < max_angle and B < max_angle and C < max_angle:
                                    cv2.drawContours(test_img2, [np.array(
                                        [(int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                         (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                         (int(point_triangle1[2][0]), int(point_triangle1[2][1]))])], 0, (255, 255, 255),
                                                     -1)



                                    # 在这里保存1L的三角区域图像
                                    # cv2.imwrite("1L_tri.png",test_img)
                                    # 获取拟合三角形最左侧的两个点
                                    point_left1, point_left2 = get_left_point(point_triangle1)

                                    rotateAngle1 = math.atan2(-(point_left1[1] - point_left2[1]),
                                                              (point_left1[0] - point_left2[0])) * 180.0 / np.pi
                                    rotateAngle1 = rotateAngle1 if(rotateAngle1>0) else (rotateAngle1+180)

                                    rotateAngle1 = round(rotateAngle1, 2)



                                    # 这个时候我们已经获得了需要的1L的三角区域，绘制在test_img上
                                    # 接下来，我们将该图与结果图进行bitwise_and融合，然后提取二层

                                    mask_img = cv2.cvtColor(test_img2, cv2.COLOR_BGR2GRAY)
                                    mask_layer_3 = cv2.merge([mask_img, mask_img, mask_img])
                                    temp_img = cv2.bitwise_and(img2, mask_layer_3)
                                    # 对bitwise_and函数再进行一次上述操作

                                    # cv2.imshow("temp_img",temp_img)

                                    # 结果很好，每次都会显示一个1L的三角形区域，接下来，我们再做一次根据inRange函数的操作
                                    test1 = cv2.inRange(temp_img, color_2[0], color_2[1])

                                    kernel_test = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                                    binary = cv2.morphologyEx(test1, cv2.MORPH_CLOSE, kernel_test, iterations=1)

                                    # 测试完成，可以提取出位于1L内部的2L层，接下来进行三角形拟合操作
                                    contours1, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    for ci in contours1:
                                        # 测试找质心
                                        M = cv2.moments(ci)
                                        cX = int(M["m10"] / (M["m00"] + float("1e-8")))
                                        cY = int(M["m01"] / (M["m00"] + float("1e-8")))

                                        # 初始化我们的测试图像

                                        # 检测轮廓的面积
                                        if Min_area < cv2.contourArea(ci):
                                            epsilon = 0.04 * cv2.arcLength(ci, True)  # 多边形拟合的距离参数，下一个函数要用到。原理见代码后链接
                                            approx = cv2.approxPolyDP(ci, epsilon, True)  # 轮廓近似。将圆润曲线折线化，以此得到该图像的角点坐标。
                                            corners = len(approx)  # 得到角点数量
                                            if corners == 3:  # 三个角点的就是三角形
                                                area, point_triangle = cv2.minEnclosingTriangle(ci)
                                                point_triangle = np.squeeze(point_triangle)
                                                # 获取拟合三角形内角角度信息
                                                A, B, C = get_triangle_infro(point_triangle)
                                                if min_angle < A and min_angle < B and min_angle < C and A < max_angle and B < max_angle and C < max_angle:
                                                    cv2.drawContours(test_img, [np.array(
                                                        [(int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                                         (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                                         (int(point_triangle1[2][0]), int(point_triangle1[2][1]))])], 0,
                                                                     (0, 0, 255), -1)
                                                    cv2.drawContours(test_img, [np.array(
                                                        [(int(point_triangle[0][0]), int(point_triangle[0][1])),
                                                         (int(point_triangle[1][0]), int(point_triangle[1][1])),
                                                         (int(point_triangle[2][0]), int(point_triangle[2][1]))])], 0,
                                                                     (255, 0, 255), -1)

                                                    for h_i in range(img.shape[1] - 1):
                                                        for j in range(img.shape[1] - 1):
                                                            if ((np.array(test_img[h_i, j])[0].all() ==
                                                                 np.array([0, 0, 0])[
                                                                     0].all()) and (
                                                                    np.array(test_img[h_i, j])[1].all() ==
                                                                    np.array([0, 0, 0])[
                                                                        1].all()) and (
                                                                    np.array(test_img[h_i, j])[2].all() ==
                                                                    np.array([0, 0, 0])[
                                                                        2].all())):
                                                                test_img[h_i, j] = [255, 255, 255]

                                                    cv2.imwrite("2L_tri.png",test_img)
                                                    cv2.line(result_img,
                                                                (int(point_triangle[0][0]), int(point_triangle[0][1])),
                                                                (int(point_triangle[1][0]), int(point_triangle[1][1])),
                                                                (200, 200, 255), 2, cv2.LINE_8)
                                                    cv2.line(result_img,
                                                                 (int(point_triangle[0][0]), int(point_triangle[0][1])),
                                                                 (int(point_triangle[2][0]), int(point_triangle[2][1])),
                                                                 (200, 200, 255), 2, cv2.LINE_8)
                                                    cv2.line(result_img,
                                                                 (int(point_triangle[1][0]), int(point_triangle[1][1])),
                                                                 (int(point_triangle[2][0]), int(point_triangle[2][1])),
                                                                 (200, 200, 255), 2, cv2.LINE_8)

                                                        # 在这里绘制1L的三角形边界
                                                    cv2.line(result_img,
                                                                 (int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                                                 (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                                                 (255, 100, 200), 2, cv2.LINE_8)
                                                    cv2.line(result_img,
                                                                 (int(point_triangle1[0][0]), int(point_triangle1[0][1])),
                                                                 (int(point_triangle1[2][0]), int(point_triangle1[2][1])),
                                                                 (255, 100, 200), 2, cv2.LINE_8)
                                                    cv2.line(result_img,
                                                                 (int(point_triangle1[1][0]), int(point_triangle1[1][1])),
                                                                 (int(point_triangle1[2][0]), int(point_triangle1[2][1])),
                                                                 (255, 100, 200), 2, cv2.LINE_8)


                                                        # 调用绘制拟合三角内角角度值在结果图像上
                                                    if args.draw_interior:
                                                        result_img = draw_interior_angle(result_img, point_triangle1)

                                                        # 获取左侧的两个点
                                                    point_left1, point_left2 = get_left_point(point_triangle)
                                                    rotateAngle = math.atan2(-(point_left1[1] - point_left2[1]),
                                                                                (point_left1[0] - point_left2[
                                                                                    0])) * 180.0 / np.pi
                                                    rotateAngle = np.abs(rotateAngle)

                                                    # 绘制旋转角度
                                                    rotateAngle = np.abs(rotateAngle1 - rotateAngle)
                                                    rotateAngle = round(rotateAngle, 2)
                                                    cv2.putText(result_img, str(rotateAngle),
                                                                (int(point_left1[0] + 20), int(point_left1[1] - 10)),
                                                                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (100, 255, 255), 1,
                                                                cv2.LINE_AA)
                else:
                    x, y, w, h = cv2.boundingRect(c)
                    if w > 1 and w > 1:  # suppress small bounding boxes (optional)
                        # bbox_list.extend([x,y,x+w-1,y+h-1])
                        bbox_list.extend([x + 1, y + 1, w, h])  # MATLAB convention: start from 1 instead of 0
                        test_list = [co[0], co[1]]

                        cv2.rectangle(result_img, (x, y), (x + w - 1, y + h - 1), box_color[i], 2, cv2.LINE_8)

                        # 绘制字体
                        cv2.putText(result_img, args.thickness_dict[i], ((x + 15), y + 20), cv2.FONT_HERSHEY_TRIPLEX,
                                    0.8,
                                    color_dict[i], 1, cv2.LINE_8)
        i += 1
    for h_i in range(img.shape[1] - 1):
        for j in range(img.shape[1] - 1):
            if ((np.array(conturn_img[h_i, j])[0].all() == np.array([0, 0, 0])[0].all()) and (
                    np.array(conturn_img[h_i, j])[1].all() == np.array([0, 0, 0])[1].all()) and (
                    np.array(conturn_img[h_i, j])[2].all() == np.array([0, 0, 0])[2].all())):
                conturn_img[h_i, j] = [255, 255, 255]

    cv2.imwrite("contours.png", conturn_img)
    cv2.imshow("orgin", cv2.imread(args.image_path))
    cv2.imshow("predict", cv2.imread(imgpath_mask))
    cv2.imshow("test233", result_img)
    cv2.imwrite("result.bmp", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_angel():
    test_img = cv2.imread("./count_angel.png")

    # 对test_img进行求轮廓，或者说根据轮廓，来判定是否1L内存在2L，如果存在，就提取出来，否则舍弃，新建一个图为空

    # 根据颜色区间进行分割，然后观察是否为中空的
    lower_color1 = np.array([255, 255, 255])
    higher_color1 = np.array([255, 255, 255])
    color_1 = [lower_color1, higher_color1]
    lower_color2 = np.array([255, 0, 0])
    higher_color2 = np.array([255, 0, 0])
    color_2 = [lower_color2, higher_color2]


    # 分别提取出1L,2L的区间
    mask_1L = cv2.inRange(test_img, color_1[0], color_1[1])
    mask_2L = cv2.inRange(test_img, color_2[0], color_2[1])


    # 提取1L的轮廓，观察轮廓内是否有中空，有的话保存
    i = 0

    contours, hierarchy = cv2.findContours(mask_1L, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # 把各个轮廓分别画到一张图上
        # 将轮廓与原图进行拟合，观察是否有非0像素，存在非0像素的话，保存该轮廓
        i = i+1
        # 测试找质心
        M = cv2.moments(c)
        cX = int(M["m10"] / (M["m00"] + float("1e-8")))
        cY = int(M["m01"] / (M["m00"] + float("1e-8")))
        # cv2.circle(img, (cX, cY), 7, (255, 0, 255), -1)

        # 检测轮廓的面积
        Min_area = 15
        if Min_area < cv2.contourArea(c):

            epsilon = 0.04 * cv2.arcLength(c, True)  # 多边形拟合的距离参数，下一个函数要用到。原理见代码后链接
            approx = cv2.approxPolyDP(c, epsilon, True)  # 轮廓近似。将圆润曲线折线化，以此得到该图像的角点坐标。
            corners = len(approx)  # 得到角点数量
            if corners == 3:  # 三个角点的就是三角形
                mask_img = np.zeros((test_img.shape), np.uint8)
                area, point_triangle = cv2.minEnclosingTriangle(c)
                point_triangle = np.squeeze(point_triangle)
                cv2.drawContours(mask_img, [np.array([(int(point_triangle[0][0]), int(point_triangle[0][1])),
                                                      (int(point_triangle[1][0]), int(point_triangle[1][1])),
                                                      (int(point_triangle[2][0]), int(point_triangle[2][1]))])], 0,
                                 (255, 255, 255), 1)
                # cv2.imwrite("mask_{}.jpg".format(i))
                cv2.imshow("test",mask_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
def main():
    args = parse_args()
    get_twist_angel(args)

if __name__ == '__main__':

    main()