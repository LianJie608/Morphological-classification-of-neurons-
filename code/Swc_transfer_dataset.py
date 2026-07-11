import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 解析SWC文件的函数
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
                node_type = int(data[1])
                x = float(data[2])
                y = float(data[3])
                z = float(data[4])
                radius = float(data[5])  # 提取节点的半径信息
                parent_id = int(data[6])

                if node_id not in neuron_data:
                    neuron_data[node_id] = {'coordinates': (x, y, z), 'parent_id': parent_id, 'children': []}
                else:
                    neuron_data[node_id]['coordinates'] = (x, y, z)
                    neuron_data[node_id]['parent_id'] = parent_id
                if parent_id in neuron_data:
                    neuron_data[parent_id]['children'].append(node_id)
    return neuron_data


#提取节点的坐标信息
def extract_coordinates(neuron_data):
    coordinates = {}
    for node in neuron_data:
        node_id, _, x, y, z, _, parent_id = node
        coordinates[node_id] = (x, y, z)
        if parent_id not in coordinates:
            coordinates[parent_id] = (None, None, None)  # 如果父节点坐标不存在，则设为None
    return coordinates

#线性插值
def linear_interpolation(point1, point2, num_points):
    # 计算两个点之间的线性插值
    x_values = np.linspace(point1[0], point2[0], num_points)
    y_values = np.linspace(point1[1], point2[1], num_points)
    z_values = np.linspace(point1[2], point2[2], num_points)
    interpolated_points = np.column_stack((x_values, y_values, z_values))
    return interpolated_points

# 对节点坐标进行等比压缩的函数
def scale_coordinates(neuron_data, scale_factor):
    for node_id, data in neuron_data.items():
        x, y, z = data['coordinates']
        scaled_x = x * scale_factor
        scaled_y = y * scale_factor
        scaled_z = z * scale_factor
        neuron_data[node_id]['coordinates'] = (scaled_x, scaled_y, scaled_z)


# 处理SWC文件并生成三维插值后的神经元结构图数据集
def process_swc_files(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.swc'):
            file_path = os.path.join(input_folder, file_name)
            neuron_data = parse_swc_file(file_path)

            # 进行插值处理
            # 创建一个临时列表来存储要添加的插值点
            interpolated_nodes_to_add = []

            # 输出神经元节点坐标、父节点坐标和子节点坐标
            for node_id, data in neuron_data.items():
                x, y, z = data['coordinates']
                parent_id = data['parent_id']
                children = data['children']

                # 输出当前节点的坐标
                print(f"Node {node_id}: ({x}, {y}, {z}), Parent: {parent_id}, Children: {children}")

                # 输出父节点的坐标
                if parent_id in neuron_data:
                    parent_x, parent_y, parent_z = neuron_data[parent_id]['coordinates']
                    # print(f"Parent ({parent_id}): ({parent_x}, {parent_y}, {parent_z})")

                    # 输出子节点的坐标
                    if children:
                        print(f" Children:")
                        for child_id in children:
                            child_x, child_y, child_z = neuron_data[child_id]['coordinates']
                            print(f" Node {child_id}: ({child_x}, {child_y}, {child_z})")


            #                 # 进行插值并添加到临时列表中
            #                 num_interpolated_points = 3  # 插值点的数量
            #                 interpolated_points = linear_interpolation(np.array([x, y, z]),
            #                                                            np.array([child_x, child_y, child_z]),
            #                                                            num_interpolated_points)
            #                 for i, point in enumerate(interpolated_points):
            #                     interpolated_node_id = f"interpolated_{node_id}_{child_id}_{i + 1}"
            #                     interpolated_nodes_to_add.append(
            #                         (interpolated_node_id, {'coordinates': (point[0], point[1], point[2])}))
            #
            # # 将临时列表中的插值点添加到neuron_data中
            # for node_id, data in interpolated_nodes_to_add:
            #     neuron_data[node_id] = data

            # 对所有节点的坐标进行等比压缩
            scale_factor = 0.5  # 假设缩小一半
            scale_coordinates(neuron_data, scale_factor)
            
            
            
            #VOXEL_SIZE = (2.68, 2.68, 2.68)  # (dx, dy, dz) 单位：µm/voxel 
            #TARGET_SHAPE = (112, 112, 112)  


            #segments = []
            #for node_id, data in neuron_data.items():
                #x, y, z = data['coordinates']
                #parent_id = data['parent_id']
                #if parent_id != -1 and parent_id in neuron_data:
                    #px, py, pz = neuron_data[parent_id]['coordinates']  # 父节点坐标
                    #segments.append(((x, y, z), (px, py, pz)))  # 存储为 ((x,y,z), (px,py,pz))
                    #volume = np.zeros(TARGET_SHAPE, dtype=np.uint8)  # shape = (Z, Y, X)

            #for (x0, y0, z0), (x1, y1, z1) in segments:
                # 插值生成 ~10–50 个点（自适应长度，避免过疏或过密）
                #num_points = max(10, int(np.linalg.norm([x1-x0, y1-y0, z1-z0]) / min(VOXEL_SIZE)))
                #points = linear_interpolation((x0, y0, z0), (x1, y1, z1), num_points)
    
                # 将物理坐标 → 体素索引：SWC(X,Y,Z) → volume(Z,Y,X)
                # 公式：voxel_z = round(z / dz), voxel_y = round(y / dy), voxel_x = round(x / dx)
            #for x, y, z in points:
                #iz = int(round(z / VOXEL_SIZE[2]))  # Z → 第0维
                #iy = int(round(y / VOXEL_SIZE[1]))  # Y → 第1维
                #ix = int(round(x / VOXEL_SIZE[0]))  # X → 第2维
            # 边界检查（防止越界）
                #if 0 <= iz < TARGET_SHAPE[0] and 0 <= iy < TARGET_SHAPE[1] and 0 <= ix < TARGET_SHAPE[2]:
                    #volume[iz, iy, ix] = 1

            # # 输出插值点的坐标
            # for node_id, data in neuron_data.items():
            #     if isinstance(node_id, str) and node_id.startswith('interpolated'):
            #         x, y, z = data['coordinates']
            #         print(f"Interpolated Node {node_id}: ({x}, {y}, {z})")

            # 提取插值前和插值后的节点坐标
            pre_interpolation_coordinates = np.array([data['coordinates'] for node_id, data in neuron_data.items() if not isinstance(node_id, str)])
            post_interpolation_coordinates = np.array([data['coordinates'] for node_id, data in neuron_data.items() if isinstance(node_id, str) and node_id.startswith('interpolated')])


            # 创建3D图形窗口并绘制图像
            fig = plt.figure(figsize=(12, 6))

            # # 绘制插值之后的神经元结构
            # ax2 = fig.add_subplot(111, projection='3d')
            # ax2.set_title('After Compression')
            # ax2.scatter(post_interpolation_coordinates[:, 0], post_interpolation_coordinates[:, 1],
            #             post_interpolation_coordinates[:, 2], c='k', marker='o')
            # ax2.set_xlabel('X')
            # ax2.set_ylabel('Y')
            # ax2.set_zlabel('Z')


            # # # 绘制插值之前的神经元结构
            ax1 = fig.add_subplot(111, projection='3d')
            # ax1.set_title('Before Compression')
            pre_interpolation_coordinates_scaled = np.array(
                [data['coordinates'] for node_id, data in neuron_data.items() if not isinstance(node_id, str)])
            ax1.scatter(pre_interpolation_coordinates_scaled[:, 0], pre_interpolation_coordinates_scaled[:, 1],
                        pre_interpolation_coordinates_scaled[:, 2], c='k', marker='o')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')

            # # 调整子图位置，使其在正中央
            fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

            # 设置坐标轴比例相等，使得图形在正中央
            ax1.set_box_aspect([1, 1, 1])  # 设置坐标轴比例为1:1:1

            # 保存图像
            plt.savefig(os.path.join(output_folder, file_name.replace('.swc', '.png')))


# 设置输入和输出文件夹路径 Not reported
#input_folder = './swc_data/swc_v3/test/Glia/astrocyte/Not reported'
#input_folder = './swc_data/swc_v3/test/Glia/microglia/Iba1-positive'

#input_folder = './swc_data/swc_v3/test/interneuron/basket'
#input_folder = './swc_data/swc_v3/test/interneuron/GABAergic'
#input_folder = './swc_data/swc_v3/test/interneuron/Nitrergic/Not reported'

#input_folder = './swc_data/swc_v3/test/principal cell/ganglion/Not reported'
#input_folder = './swc_data/swc_v3/test/principal cell/granule'
#input_folder = './swc_data/swc_v3/test/principal cell/medium spiny/Not reported'
#input_folder = './swc_data/swc_v3/test/principal cell/Parachromaffin/Pheochromocytoma line 12'
#input_folder = './swc_data/swc_v3/test/principal cell/Purkinje/Not reported'
# input_folder = './swc_data/swc_v3/test/principal cell/pyramidal'

#input_folder = './swc_data/swc_v3/test/sensory receptor'


##输出文件路径
#output_folder = './swc_data/3D_Img_fixed_final/test/Glia/astrocyte'
#output_folder = './swc_data/3D_Img_fixed_final/test/Glia/microglia'

#output_folder = './swc_data/3D_Img_fixed_final/test/interneuron/basket'
#output_folder = './swc_data/3D_Img_fixed_final/test/interneuron/GABAergic'
#output_folder = './swc_data/3D_Img_fixed_final/test/interneuron/Nitrergic'

#output_folder = './swc_data/3D_Img_fixed_final/test/principal cell/ganglion'
#output_folder = './swc_data/3D_Img_fixed_final/test/principal cell/granule'
#output_folder = './swc_data/3D_Img_fixed_final/test/principal cell/medium spiny'
#output_folder = './swc_data/3D_Img_fixed_final/test/principal cell/Parachromaffin'
#output_folder = './swc_data/3D_Img_fixed_final/test/principal cell/Purkinje'
# output_folder = './swc_data/3D_Img_fixed_final/test/principal cell/pyramidal'

#output_folder = './swc_data/3D_Img_fixed_final/test/sensory receptor'



##人类脑神经元数据集
input_folder = './swc_data/human/induced pluripotent stem cell-(iPSC)-derived'
#

output_folder = './swc_data/3D_Img_human/induced pluripotent stem cell-(iPSC)-derived'

# 调用处理SWC文件的函数
process_swc_files(input_folder, output_folder)

