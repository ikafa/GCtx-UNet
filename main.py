import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import _C
from networks.GCtx_UNet import GCViT_Unet as ViT_seg


def load_model_and_segment(image_tensor, config_path='configs/GCViT_xxtiny_224_lite.yaml', 
                           model_path='model_out/epoch_149.pth', 
                           img_size=224, num_classes=9):
    """
    加载预训练模型并对输入图像进行分割
    
    参数:
        image_tensor: 输入图像张量，形状应为 [1, 1, H, W] 或 [1, 3, H, W]
        config_path: 模型配置文件路径
        model_path: 模型权重文件路径
        img_size: 模型输入图像大小
        num_classes: 分割类别数量
        
    返回:
        分割结果张量
    """
    # 准备参数配置
    config = _C.clone()
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ViT_seg(config, img_size=img_size, num_classes=num_classes).to(device)
    
    # 加载预训练权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    msg = net.load_state_dict(checkpoint, strict=False)
    print(f"模型加载状态: {msg}")
    
    # 设置为评估模式
    net.eval()
    
    # 确保输入图像在正确的设备上
    image_tensor = image_tensor.to(device)
    
    # 如果需要，调整图像大小
    if image_tensor.shape[2] != img_size or image_tensor.shape[3] != img_size:
        print(f"警告：输入图像大小 {image_tensor.shape[2:]} 与模型期望的 {img_size}x{img_size} 不匹配")
        # 可以在这里添加图像调整代码
    
    # 使用模型进行推理
    with torch.no_grad():
        output = net(image_tensor)
    
    # 获取分割结果（取概率最高的类别）
    segmentation = torch.argmax(output, dim=1)
    
    return segmentation

def load_model_for_segmentation(config_path='configs/GCViT_xxtiny_224_lite.yaml',
                               model_path='model_out/epoch_149.pth',
                               img_size=224, num_classes=9):
    """
    加载模型用于分割任务
    
    参数:
        config_path: 模型配置文件路径
        model_path: 模型权重文件路径
        img_size: 模型输入图像大小
        num_classes: 分割类别数量
        
    返回:
        加载好的模型
    """
    import torch
    
    # 仅创建设备和模型结构，不进行推理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 这里只初始化模型，不进行推理
    # 在load_model_and_segment函数中初始化模型的部分代码
    # 注意：我们需要从load_model_and_segment提取初始化模型的逻辑
    config = _C.clone()
    
    # 初始化模型
    net = ViT_seg(config, img_size=img_size, num_classes=num_classes).to(device)
    
    # 加载预训练权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    msg = net.load_state_dict(checkpoint, strict=False)
    print(f"模型加载状态: {msg}")
    
    # 设置为评估模式
    net.eval()
    
    return net, device

def segment_image_with_model(image_tensor, model, device):
    """
    使用已加载的模型对图像进行分割
    
    参数:
        image_tensor: 输入图像张量
        model: 预加载的模型
        device: 计算设备
        
    返回:
        分割结果
    """
    # 确保输入图像在正确的设备上
    image_tensor = image_tensor.to(device)
    
    # 使用模型进行推理
    with torch.no_grad():
        output = model(image_tensor)
    
    # 获取分割结果（取概率最高的类别）
    segmentation = torch.argmax(output, dim=1)
    
    return segmentation

def segment_and_save(image_path, output_dir='results', 
                    config_path='configs/GCViT_xxtiny_224_lite.yaml',
                    model_path='model_out/epoch_149.pth',
                    img_size=224, num_classes=9):
    """
    加载RGB图像，进行分割，并保存结果
    
    参数:
        image_path: 输入图像的路径或包含图像路径列表的txt文件
        output_dir: 输出结果的目录
        config_path: 模型配置文件路径
        model_path: 模型权重文件路径
        img_size: 模型输入图像大小
        num_classes: 分割类别数量
    """
    # 检查是否为批处理模式（输入为txt文件）
    if image_path.endswith('.txt'):
        # 对于批处理，先加载一次模型
        print("批处理模式：加载模型...")
        model, device = load_model_for_segmentation(
            config_path=config_path,
            model_path=model_path,
            img_size=img_size,
            num_classes=num_classes
        )
        
        with open(image_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        results = []
        for img_path in image_paths:
            print(f"处理图像: {img_path}")
            try:
                # 处理单个图像时传入预加载的模型
                result = process_single_image(
                    img_path, output_dir, config_path, model_path, 
                    img_size, num_classes, preloaded_model=(model, device)
                )
                results.append(result)
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
        
        return results
    else:
        # 单图像处理模式
        return process_single_image(image_path, output_dir, config_path, model_path, img_size, num_classes)

def process_single_image(image_path, output_dir='results', 
                        config_path='configs/GCViT_xxtiny_224_lite.yaml',
                        model_path='model_out/epoch_149.pth',
                        img_size=224, num_classes=9,
                        preloaded_model=None):
    """
    处理单张图像的辅助函数
    
    参数:
        preloaded_model: 可选的预加载模型和设备元组 (model, device)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 图像名称（不包含路径和扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 加载RGB图像
    image = Image.open(image_path).convert('RGB')
    
    # 保存原始图像大小，用于后续缩放
    original_size = image.size  # (width, height)
    
    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度 [1, 3, H, W]
    
    # 根据是否有预加载模型决定如何进行分割
    if preloaded_model is not None:
        model, device = preloaded_model
        segmentation_result = segment_image_with_model(image_tensor, model, device)
    else:
        # 对于单一图像处理，仍然使用原始方法加载模型和分割
        segmentation_result = load_model_and_segment(
            image_tensor, 
            config_path=config_path,
            model_path=model_path,
            img_size=img_size,
            num_classes=num_classes
        )
    
    # 转换为numpy数组
    segmentation_np = segmentation_result.cpu().numpy()[0]  # 移除批次维度 [H, W]
    
    # 将分割结果缩放回原始图像大小
    # 注意: 这里使用PIL的resize方法，使用最近邻插值以保持类别标签的整数性质
    segmentation_pil = Image.fromarray(segmentation_np.astype(np.uint8))
    segmentation_pil = segmentation_pil.resize(original_size, Image.NEAREST)
    segmentation_np = np.array(segmentation_pil)
    
    # 保存分割结果为npy文件
    npy_path = os.path.join(output_dir, f"{image_name}_rust_seg.npy")
    np.save(npy_path, segmentation_np)
    print(f"分割结果已保存为: {npy_path}")
    
    # 自定义颜色映射
    color_map = {
        0: (0, 0, 0),       # "#000000"（黑色）
        1: (178, 247, 239), # "#B2F7EF"（浅蓝绿色）
        2: (255, 201, 113), # "#FFC971"（浅橙黄色）
        3: (255, 182, 39),  # "#FFB627"（橙黄色）
        4: (255, 149, 5),   # "#FF9505"（橙色）
        5: (226, 113, 29),  # "#E2711D"（深橙色）
        6: (204, 88, 3),    # "#CC5803"（棕橙色）
    }
    # 类别7及以上使用红色
    red_color = (255, 0, 0)  # "#FF0000"（红色）
    
    # 创建RGB输出图像
    color_segmentation = np.zeros((segmentation_np.shape[0], segmentation_np.shape[1], 3), dtype=np.uint8)
    
    # 为每个类别填充颜色
    for class_idx in range(num_classes):
        mask = (segmentation_np == class_idx)
        if class_idx in color_map:
            color_segmentation[mask] = color_map[class_idx]
        else:
            # 类别7及以上使用红色
            color_segmentation[mask] = red_color
    
    # 保存彩色分割图像
    color_image_path = os.path.join(output_dir, f"{image_name}_rust_vis.png")
    color_image = Image.fromarray(color_segmentation)
    color_image.save(color_image_path)
    print(f"彩色分割图像已保存为: {color_image_path}")
    
    return segmentation_np, color_segmentation

if __name__ == "__main__":
    # 使用示例
    import sys
    segment_and_save(sys.argv[1])