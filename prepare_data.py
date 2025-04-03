import os
import numpy as np
import h5py
from PIL import Image
from sklearn.model_selection import train_test_split
import typer
from collections import defaultdict

def convert_data(input_dir, output_dir, grayscale=False):
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train_npz')
    test_dir = os.path.join(output_dir, 'test_vol_h5')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 输入目录
    image_dir = os.path.join(input_dir, 'JPEGImages')
    label_dir = os.path.join(input_dir, 'SegmentationClassNpy')

    # 获取所有图像文件名
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 划分训练集和验证集
    train_files, val_files = train_test_split(image_files, test_size=0.1, random_state=42)

    print(f"训练集数量: {len(train_files)}")
    print(f"验证集数量: {len(val_files)}")

    # 处理训练集
    for filename in train_files:
        base_name = os.path.splitext(filename)[0]

        # 读取图像
        img_path = os.path.join(image_dir, filename)
        img = np.array(Image.open(img_path))

        if grayscale:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

        # 归一化像素值从0-255到0-1
        img = img.astype(np.float32) / 255.0

        # 读取对应的标签
        label_path = os.path.join(label_dir, f"{base_name}.npy")
        label = np.load(label_path)

        # 保存为npz文件
        output_path = os.path.join(train_dir, f"{base_name}.npz")
        np.savez(output_path, image=img, label=label)

        print(f"已处理训练图像: {filename}")

    # 处理验证集 - 按大小分组
    size_groups = defaultdict(list)

    # 首先按大小分组图像
    for filename in val_files:
        base_name = os.path.splitext(filename)[0]

        # 读取图像获取尺寸
        img_path = os.path.join(image_dir, filename)
        with Image.open(img_path) as img:
            size = f"{img.width}x{img.height}"

        # 将文件名添加到对应大小的组
        size_groups[size].append(filename)

    # 对每个大小的组分别处理
    for size, filenames in size_groups.items():
        images = []
        labels = []
        case_names = []

        for filename in filenames:
            base_name = os.path.splitext(filename)[0]

            # 读取图像
            img_path = os.path.join(image_dir, filename)
            img = np.array(Image.open(img_path))

            if grayscale:
                img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

            # 归一化像素值从0-255到0-1
            img = img.astype(np.float32) / 255.0

            # 读取对应的标签
            label_path = os.path.join(label_dir, f"{base_name}.npy")
            label = np.load(label_path)

            # 添加到列表
            images.append(img)
            labels.append(label)
            case_names.append(base_name)

        # 保存为h5文件，按大小命名
        output_path = os.path.join(test_dir, f"{size}.h5")
        with h5py.File(output_path, 'w') as f:
            # 创建数据集
            f.create_dataset('images', data=np.array(images))
            f.create_dataset('labels', data=np.array(labels))
            # 保存文件名作为case_names
            dt = h5py.special_dtype(vlen=str)
            case_names_dataset = f.create_dataset('case_names', (len(case_names),), dtype=dt)
            for i, name in enumerate(case_names):
                case_names_dataset[i] = name

        print(f"已处理验证图像组 {size}, 包含 {len(filenames)} 张图像")

    print("数据转换完成！")

if __name__ == "__main__":
    typer.run(convert_data)
