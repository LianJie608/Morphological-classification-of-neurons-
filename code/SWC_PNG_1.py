import os
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 坐标缩放函数
def scale_coordinates(coord, min_val, max_val):
    if max_val == min_val:  # 防止除以零
        return 0
    scaled_coord = (coord - min_val) / (max_val - min_val) * 30
    return scaled_coord


# 读取SWC文件
def parse_swc_file(file_path):
    neuron_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # 跳过注释行
            data = line.split()  # 按空格分割数据
            if len(data) >= 7:
                # 提取节点编号、节点类型、X、Y、Z坐标、半径和父节点编号
                node_id = int(data[0])
                x = float(data[2])
                y = float(data[3])
                z = float(data[4])
                parent_id = int(data[6])
                if node_id not in neuron_data:
                    neuron_data[node_id] = {'coordinates': (x, y, z), 'parent_id': parent_id, 'children': []}
                else:
                    neuron_data[node_id]['coordinates'] = (x, y, z)
                    neuron_data[node_id]['parent_id'] = parent_id
                if parent_id in neuron_data:
                    neuron_data[parent_id]['children'].append(node_id)

    # 提取坐标和相关信息
    coordinates_data = []
    for node_id, node_info in neuron_data.items():
        x, y, z = node_info['coordinates']
        parent_id = node_info['parent_id']
        coordinates_data.append([node_id, parent_id, x, y, z])

    # 转换为numpy数组以便计算min和max
    coordinates_array = np.array(coordinates_data)
    min_vals = np.min(coordinates_array[:, 2:], axis=0)
    max_vals = np.max(coordinates_array[:, 2:], axis=0)

    # 缩放坐标
    scaled_coordinates_array = np.copy(coordinates_array[:, 2:])
    for i in range(3):  # 分别处理x, y, z坐标
        scaled_coordinates_array[:, i] = scale_coordinates(coordinates_array[:, i + 2], min_vals[i], max_vals[i])

    # 可视化原始和缩放后的坐标
    fig = plt.figure(figsize=(12,7))

    # 原始坐标的可视化
    ax_orig = fig.add_subplot(121, projection='3d')
    for x, y, z in coordinates_array[:, 2:]:
        ax_orig.scatter(x, y, z, c='k', marker='o')
    ax_orig.set_xlabel('X')
    ax_orig.set_ylabel('Y')
    ax_orig.set_zlabel('Z')
    ax_orig.set_title('Original Coordinates')
    ax_orig.legend()

    # 缩放坐标的可视化
    ax_scaled = fig.add_subplot(122, projection='3d')
    for scaled_x, scaled_y, scaled_z in scaled_coordinates_array:
        ax_scaled.scatter(scaled_x, scaled_y, scaled_z,c='k', marker='o')
    ax_scaled.set_xlabel('Scaled X')
    ax_scaled.set_ylabel('Scaled Y')
    ax_scaled.set_zlabel('Scaled Z')
    ax_scaled.set_title('Scaled Coordinates')
    ax_scaled.legend()

    plt.tight_layout()
    plt.savefig('./swc_data/29643.png')
    # 返回缩放后的坐标数据
    return neuron_data, scaled_coordinates_array


def generate_voxel_data(s, e, num_steps=10):
    # 计算两点之间的向量
    vector = np.array(e) - np.array(s)
    # 计算两点之间的实际距离
    distance = np.linalg.norm(vector)
    # 如果距离为0，则不需要插值
    if distance == 0:
        return [s]
        # 计算每个维度的步长
    step_size = vector / num_steps
    # 初始化插值点列表
    interpolated_points = [s.tolist()]
    # 进行插值
    for i in range(1, num_steps + 1):
        interpolated_point = s + i * step_size
        # 注意：直接将interpolated_point转换为列表
        interpolated_points.append(interpolated_point.tolist())
        # 添加终点
    interpolated_points.append(e.tolist())
    return interpolated_points


#
# def linear_interpolation(S, E, num_steps=1):
#     """
#     Performs linear interpolation between two points S and E with a specified number of steps.
#
#     Args:
#         S: A tuple representing the starting point (Xs, Ys, Zs).
#         E: A tuple representing the ending point (Xe, Ye, Ze).
#         num_steps: The number of interpolation steps (excluding start and end points).
#
#     Returns:
#         A list of tuples representing the interpolated points.
#     """
#
#     Xs, Ys, Zs = S
#     Xe, Ye, Ze = E
#
#     # Calculate MaxDistance
#     MaxDistance = max(abs(Xs - Xe), abs(Ys - Ye), abs(Zs - Ze))
#
#     # Calculate intervals
#     IntervalX = (Xe - Xs) / (num_steps + 1)  # num_steps + 1 to include start and end points
#     IntervalY = (Ye - Ys) / (num_steps + 1)
#     IntervalZ = (Ze - Zs) / (num_steps + 1)
#
#     # Initialize lists
#     L = []
#     TemporaryX = Xs
#     TemporaryY = Ys
#     TemporaryZ = Zs
#
#     # Perform interpolation
#     for i in range(num_steps + 1):  # Include start and end points
#         L.append((TemporaryX, TemporaryY, TemporaryZ))
#         TemporaryX += IntervalX
#         TemporaryY += IntervalY
#         TemporaryZ += IntervalZ
#
#     return L

# 读取SWC文件并调用转换函数


neuron_data, scaled_coordinates_array = parse_swc_file('./swc_data/swc_raw/train/interneuron/GABAergic/29643.swc')

# Applying linear interpolation to generate voxel graphics of the neuron
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

# Perform linear interpolation between successive points and plot
for i in range(len(scaled_coordinates_array) - 1):
    interpolated_points = generate_voxel_data(scaled_coordinates_array[i], scaled_coordinates_array[i+1], num_steps=10)  # 使用10步插值
    for point in interpolated_points:
        ax.scatter(point[0], point[1], point[2], c='k', marker='o')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Neuron Voxel Graphics')

plt.tight_layout()
plt.savefig('./swc_data/29643_1.png')






# # 转换为核心矩阵并进行坐标缩放
# def convert_to_core_matrix_with_scaling(neuron_data):
#     # 从neuron_data中提取坐标和相关信息
#     coordinates_data = []
#     for node_id, node_info in neuron_data.items():
#         # 假设我们只关心坐标和父节点ID
#         x, y, z = node_info['coordinates']
#         parent_id = node_info['parent_id']
#         coordinates_data.append([node_id, parent_id, x, y, z])
#
#     # 将坐标数据转换为numpy数组以便计算min和max
#     coordinates_array = np.array(coordinates_data)
#
#     # 提取坐标的最小值和最大值
#     min_vals = np.min(coordinates_array[:, 2:], axis=0)
#     max_vals = np.max(coordinates_array[:, 2:], axis=0)
#
#     core_matrix = []
#     for row in coordinates_array:
#         node_id, parent_id, x, y, z = row
#         scaled_x = scale_coordinates(x, min_vals[0], max_vals[0])
#         scaled_y = scale_coordinates(y, min_vals[1], max_vals[1])
#         scaled_z = scale_coordinates(z, min_vals[2], max_vals[2])
#         core_matrix.append([node_id, parent_id, scaled_x, scaled_y, scaled_z])
#
#     return np.array(core_matrix)
# def visualize_swc_data_with_matplotlib(file_path,output_folder='./swc_data/'):
#     swc_data = read_swc_file(file_path)
#     # 假设swc_data中的每行是[x, y, z, radius, ...]的格式
#     # 我们只需要x, y, z坐标来绘制节点
#     x, y, z = swc_data[:, 0], swc_data[:, 1], swc_data[:, 2]
#
#     # 创建一个3D图形
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 绘制3D散点图
#     ax.scatter(x, y, z, c='k', marker='o')
#     # c参数设置颜色，marker参数设置标记的形状（这里是圆圈'o'）
#
#     # 添加轴标签
#     ax.set_xlabel('X Axis')
#     ax.set_ylabel('Y Axis')
#     ax.set_zlabel('Z Axis')
#
#     # 保存图像
#     # 从文件路径中提取文件名（不包括扩展名）
#     base_name = os.path.splitext(os.path.basename(file_path))[0]
#     file_name = f"{base_name}.png"
#
#     # 保存图像
#     plt.savefig(os.path.join(output_folder, file_name))
#
# # 调用函数来可视化SWC文件中的数据
# visualize_swc_data_with_matplotlib('./swc_data/swc_raw/train/Glia/astrocyte/Not reported/77329.swc')  # 替换为你的SWC文件路径
#
