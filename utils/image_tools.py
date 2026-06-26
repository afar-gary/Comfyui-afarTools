import torch,folder_paths    # type: ignore
import torch.nn.functional as F      # type: ignore
import comfy.utils   # type: ignore
import os,sys,json,math  # type: ignore
from nodes import CLIPTextEncode
import comfy_extras.nodes_sam3 as sam3_module   # type: ignore

import types
import importlib.util
import comfy.model_management   # type: ignore

import numpy as np   # type: ignore
from .utils import aftools
from scipy.ndimage import binary_closing, binary_fill_holes     # type: ignore

# ==========================================
# 智能双引擎：Mask 孔洞填充
# ==========================================
try:
    import cv2   # type: ignore
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

def mask_fill_hole(mask_4d):
    """
    智能双引擎 Mask 孔洞填充：
    - 若有 cv2，使用工业级 floodFill (极速且 100% 稳定)。
    - 若无 cv2，使用优化的 max_pool2d 纯 PyTorch 方案 (无同步开销，<20ms，彻底修复边界 BUG)。
    """
    if HAS_CV2:
        return _fill_hole_cv2(mask_4d)
    else:
        return _fill_hole_scipy(mask_4d)

# ==========================================
# 辅助函数：基于 SciPy 的多阈值迭代孔洞填充 (社区验证终极方案)
# ==========================================
def _fill_hole_scipy(mask_4d):
    """
    填充 Mask 内部不与图像边界相连的闭合孔洞。
    采用 comfyui-inpaint-cropandstitch 验证过的多阈值迭代 + SciPy 形态学算法。
    完美兼容经过 expand/blur 后的渐变 Mask，彻底根除边界 BUG 与性能瓶颈。
    无需额外安装依赖 (scipy 为 ComfyUI 核心内置库)。
    
    参数:
        mask_4d: Tensor, shape [B, 1, H, W], 值域 0.0 - 1.0
    
    返回:
        filled_mask_4d: Tensor, shape [B, 1, H, W], 孔洞被平滑填充
    """
    B, C, H, W = mask_4d.shape
    device = mask_4d.device
    dtype = mask_4d.dtype
    
    results = []
    # 逐张处理 Batch
    for i in range(B):
        # 1. 转为 numpy [H, W]
        mask_np = mask_4d[i, 0].cpu().numpy()
        
        # 2. 多阈值迭代填充 (从严格到宽松，保留渐变边缘)
        thresholds = [1.0, 0.99, 0.97, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        for threshold in thresholds:
            # 二值化
            thresholded_mask = mask_np >= threshold
            
            # 闭运算 (先膨胀后腐蚀)，闭合微小缝隙。border_value=1 确保边界不会被误判为孔洞
            closed_mask = binary_closing(thresholded_mask, structure=np.ones((3, 3)), border_value=1)
            
            # 填充孔洞 (SciPy 底层 C 实现，极其稳健，无边界 BUG)
            filled_mask = binary_fill_holes(closed_mask)            
            
            # 将填充的部分以当前阈值合并回原 mask，保留平滑过渡
            mask_np = np.maximum(mask_np, np.where(filled_mask, threshold, 0.0))
            
        # 3. 转回 tensor
        results.append(torch.from_numpy(mask_np.astype(np.float32)))
        
    # 4. 堆叠并恢复 [B, 1, H, W] 形状，确保设备与 dtype 一致
    return torch.stack(results, dim=0).unsqueeze(1).to(device=device, dtype=dtype)

def _fill_hole_cv2(mask_tensor):
    """
    填充 Mask 内部不与图像边界相连的闭合孔洞。
    直接采用工业级标准的 cv2.floodFill 算法，彻底杜绝纯 PyTorch 模拟带来的边界 BUG。
    
    参数:
        mask_tensor: Tensor, shape [B, 1, H, W] 或 [B, H, W], 值域 0.0 - 1.0
    
    返回:
        filled_mask: Tensor, shape [B, 1, H, W], 孔洞被填充为 1.0
    """
    # 统一处理为 [B, H, W] 维度
    if mask_tensor.dim() == 4:
        mask_tensor = mask_tensor.squeeze(1)
        
    B, H, W = mask_tensor.shape
    device = mask_tensor.device
    dtype = mask_tensor.dtype
    
    filled_masks = []
    for i in range(B):
        # 1. 转为 numpy 并二值化 (>0.5 视为前景 1，否则为 0)
        mask_np = (mask_tensor[i].cpu().numpy() > 0.5).astype(np.uint8)
        
        # 2. 在遮罩周围添加 1 像素的边框（值为 0），确保外部区域绝对连通到 (0,0)
        padded_mask = np.pad(mask_np, pad_width=1, mode='constant', constant_values=0)
        ph, pw = padded_mask.shape # ph = H+2, pw = W+2
        
        # 3. 创建 floodFill 所需的 mask，OpenCV 硬性要求：必须比 image 大 2
        mask_ff = np.zeros((ph + 2, pw + 2), np.uint8)
        
        # 4. 从 (0,0) 点进行 floodFill，将连通的外部背景填充为 1
        # 注意：cv2.floodFill 会就地修改 floodfill 数组
        floodfill = padded_mask.copy()
        cv2.floodFill(floodfill, mask_ff, (0, 0), 1)
        
        # 5. floodfill == 0 的地方，既不是原前景，也不是连通的外部背景 -> 内部孔洞
        holes = (floodfill == 0).astype(np.uint8)
        
        # 6. 移除 1 像素边框，恢复原始尺寸 (H, W)
        holes = holes[1:-1, 1:-1]
        
        # 7. 原 mask 加上空洞部分，并限制在 0-1 之间
        filled_np = np.clip(mask_np + holes, 0, 1).astype(np.float32)
        
        # 8. 转回 tensor
        filled_tensor = torch.from_numpy(filled_np).to(device=device, dtype=dtype)
        filled_masks.append(filled_tensor)
        
    # 9. 堆叠并恢复通道维度 [B, 1, H, W]
    filled_mask = torch.stack(filled_masks, dim=0).unsqueeze(1)
    return filled_mask


# ==========================================
# 辅助函数：高效锐化 (Unsharp Mask)
# ==========================================
def apply_sharpness(image, sharp_amount):
    """
    对图像进行锐化处理。
    使用 Unsharp Mask (反锐化掩模) 算法：Sharpened = Original + Amount * (Original - Blurred)
    注意：此函数仅设计用于多通道 IMAGE 张量 [B, H, W, C]，绝不应用于单通道 MASK。
    """
    if sharp_amount <= 0.0:
        return image
        
    device = image.device
    dtype = image.dtype
    B, H, W, C = image.shape
    
    # 使用 5x5 高斯核作为模糊基础，提取低频信息
    k_size = 5
    sigma = 1.0
    coords = torch.arange(k_size, dtype=dtype, device=device) - k_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    # 深度可分离卷积核 [C, 1, K, 1] 和 [C, 1, 1, K]
    kernel_v = g.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    kernel_h = g.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    
    img_permuted = image.permute(0, 3, 1, 2) # [B, C, H, W]
    
    pad = k_size // 2
    # 1. 垂直方向模糊
    padded_v = F.pad(img_permuted, (0, 0, pad, pad), mode='reflect')
    blurred_v = F.conv2d(padded_v, kernel_v, groups=C)
    # 2. 水平方向模糊
    padded_h = F.pad(blurred_v, (pad, pad, 0, 0), mode='reflect')
    blurred = F.conv2d(padded_h, kernel_h, groups=C)
    
    # 3. 锐化公式：原图 + 强度 * (原图 - 模糊图)
    sharpened = img_permuted + sharp_amount * (img_permuted - blurred)
    sharpened = torch.clamp(sharpened, 0.0, 1.0)
    
    return sharpened.permute(0, 2, 3, 1)


# ==========================================
# 辅助函数：智能颜色字符串解析器
# ==========================================
def parse_color_string(color_str: str):
    """
    支持格式：
    1. 预设单词: white, black, grey, gray, red, green, blue, yellow, cyan, magenta
    2. Hex 16进制: #FFFFFF, FFFFFF, #FFF, FFF
    3. 浮点/整数三元组: 1.0, 1.0, 1.0 或 255, 255, 255 (自动识别范围并归一化)
    4. 单值灰度: 0.5 或 128 (自动转为 R=G=B)
    返回: (r, g, b) 元组，值域严格在 0.0 - 1.0 之间
    """
    color_str = str(color_str).strip().lower()
    
    # 1. 预设颜色匹配   
    if color_str in ['black','bk']: return 0.0, 0.0, 0.0
    if color_str in ['white','w']: return 1.0, 1.0, 1.0
    if color_str in ['gray','grey','gr']: return 0.5, 0.5, 0.5
    if color_str in ['red','r']: return 1.0, 0.0, 0.0
    if color_str in ['green','g']: return 0.0, 1.0, 0.0
    if color_str in ['blue','b']: return 0.0, 0.0, 1.0
    if color_str in ['yellow','y']: return 1.0, 1.0, 0.0
    if color_str in ['cyan','c']: return 0.0, 1.0, 1.0
    if color_str in ['magenta','m']: return 1.0, 0.0, 1.0
        
    # 2. Hex 16进制匹配 (支持 3位 或 6位，带或不带 #)
    hex_str = color_str.lstrip('#')
    if all(c in '0123456789abcdef' for c in hex_str):
        if len(hex_str) == 3:
            hex_str = ''.join([c*2 for c in hex_str]) # F00 -> FF0000
        if len(hex_str) == 6:
            try:
                r = int(hex_str[0:2], 16) / 255.0
                g = int(hex_str[2:4], 16) / 255.0
                b = int(hex_str[4:6], 16) / 255.0
                return (r, g, b)
            except ValueError:
                pass
                
    # 3. 逗号分隔的数值匹配 (如 "1.0, 0.5, 0.0" 或 "255, 0, 0")
    if ',' in color_str:
        try:
            parts = [float(p.strip()) for p in color_str.split(',')]
            if len(parts) == 1:
                # 单值灰度
                val = max(0.0, min(1.0, parts[0] / 255.0 if parts[0] > 1.0 else parts[0]))
                return (val, val, val)
            elif len(parts) == 3:
                # 自动识别是 0-255 还是 0-1 范围
                if max(parts) > 1.0:
                    parts = [p / 255.0 for p in parts]
                # Clamp 确保不越界
                parts = [max(0.0, min(1.0, p)) for p in parts]
                return tuple(parts)
        except ValueError:
            pass
            
    # 4. 单个数值字符串 (无逗号，如 "0.5" 或 "255")
    try:
        val = float(color_str)
        if val > 1.0:
            val = val / 255.0
        val = max(0.0, min(1.0, val))
        return (val, val, val)
    except ValueError:
        pass
        
    # 兜底：解析失败默认返回黑色
    return (0.0, 0.0, 0.0)

# ==========================================
# 辅助函数：零依赖的高斯模糊 (用于 Mask 边缘模糊)
# ==========================================
# def gaussian_blur_2d(tensor, kernel_size=5, sigma=1.0):
#     """
#     tensor shape: [B, 1, H, W]
#     """
#     device = tensor.device
#     dtype = tensor.dtype
    
#     # 生成 1D 高斯核
#     coords = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
#     g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#     g = g / g.sum()
    
#     # 扩展为 2D 核: [1, 1, K, K]
#     kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    
#     # 反射填充以防止边缘变暗
#     pad = kernel_size // 2
#     padded = F.pad(tensor, (pad, pad, pad, pad), mode='reflect')
    
#     return F.conv2d(padded, kernel, groups=1)

# ==========================================
# 辅助函数：极致优化的可分离高斯模糊 (Separable Gaussian Blur)
# ==========================================
def gaussian_blur_2d_fast(tensor, kernel_size=5, sigma=1.0):
    """
    使用可分离卷积 (1D 水平 + 1D 垂直) 替代 2D 卷积，
    将复杂度从 O(K^2) 降至 O(2K)，彻底消除大模糊半径下的卡顿。
    tensor shape: [B, 1, H, W]
    """
    device = tensor.device
    dtype = tensor.dtype
    
    # 1. 确保输入张量内存连续，防止底层卷积触发低效的隐式拷贝路径
    tensor = tensor.contiguous()
    
    # 2. 生成 1D 高斯核
    coords = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    # 3. 重塑为 1D 卷积核形状
    # 水平核: [1, 1, 1, K]
    kernel_h = g.view(1, 1, 1, -1)
    # 垂直核: [1, 1, K, 1]
    kernel_v = g.view(1, 1, -1, 1)
    
    pad = kernel_size // 2
    
    # 4. 水平方向模糊 (仅对左右进行 reflect padding)
    padded_h = F.pad(tensor, (pad, pad, 0, 0), mode='reflect')
    blurred_h = F.conv2d(padded_h, kernel_h, groups=1)
    
    # 5. 垂直方向模糊 (仅对上下进行 reflect padding)
    padded_v = F.pad(blurred_h, (0, 0, pad, pad), mode='reflect')
    blurred_v = F.conv2d(padded_v, kernel_v, groups=1)
    
    return blurred_v

# ==========================================
# 辅助函数：绝对安全的尺寸计算 (修复单一维度约束与最小尺寸BUG)
# ==========================================
def calc_target_size(scaleMode,ref_W, ref_H, preset,flip,megapixels,out_short,out_longer,out_W, out_H,step):
    """
    逻辑：
    1. 双非0：严格按用户输入的 out_W, out_H。
    2. 仅 out_W > 0：宽度设为 out_W，高度保持参考高度 ref_H 不变。
    3. 仅 out_H > 0：高度设为 out_H，宽度保持参考宽度 ref_W 不变。
    4. 双 0：保持参考尺寸 ref_W, ref_H。
    
    安全钳制：确保结果 >= 64，<= 8192，且严格为 8 的倍数。
    """
    # 1. 确定基础目标尺寸
    t_W,t_H = 0,0
    longer = max(ref_W,ref_H)
    short = min(ref_W,ref_H)
    divisor = math.gcd(ref_W,ref_H)
    image1_ratio_str = f'{round(ref_W/divisor)}:{round(ref_H/divisor)}'
    match scaleMode:
        case 'totalPixels':
            w_ratio, h_ratio = round(ref_W/divisor),round(ref_H/divisor)
            total_pixels = megapixels * 1024 * 1024
            # longest_edge = 4096
            # if total_pixels > longest_edge**2:
            #     total_pixels = longest_edge**2
            scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
            t_W = round(w_ratio * scale / step) * step
            t_H = round(h_ratio * scale / step) * step
        case 'shortEdge':
            if out_short>64:
                if short == ref_W:
                    t_W, t_H = out_short, out_short//round(ref_W/divisor)*round(ref_H/divisor)
                else:
                    t_W, t_H = out_short//round(ref_H/divisor)*round(ref_W/divisor),out_short
            else:
                t_W, t_H = ref_W, ref_H
        case 'longerEdge':
            if out_longer>64:
                if longer == ref_W:
                    t_W, t_H = out_longer, out_longer//round(ref_W/divisor)*round(ref_H/divisor)
                else:
                    t_W, t_H = out_longer//round(ref_H/divisor)*round(ref_W/divisor),out_longer
            else:
                t_W, t_H = ref_W, ref_H
        case 'preset':
            p = preset.split(' ')[1].split('x')
            t_W, t_H = int(p[0]),int(p[1])
        case 'custom':
            if out_W > 0 and out_H > 0:
                t_W, t_H = out_W, out_H
            elif out_W > 0:
                t_W, t_H = out_W, ref_H  # 只约束宽度，高度不变
            elif out_H > 0:
                t_W, t_H = ref_W, out_H  # 只约束高度，宽度不变
            else:
                t_W, t_H = ref_W, ref_H  # 都不约束
    if flip:
        temp = t_W
        t_W = t_H
        t_H = temp 
    

        
    # 2. 安全钳制 (Clamp)：四舍五入到 8 的倍数，并限制在 [64, 8192] 范围内
    # 使用 (x + 4) // 8 * 8 确保标准的四舍五入到 8 的倍数
    t_W = max(64, min(8192, int((t_W + 4) // step * step)))
    t_H = max(64, min(8192, int((t_H + 4) // step * step)))
    
    return t_W, t_H


# ==========================================
# 辅助函数：for corp mask resize
# ==========================================
def crop_target_size(scaleMode,ref_W, ref_H, megapixels,out_short,out_longer,step):
    """
    逻辑：
    1. 双非0：严格按用户输入的 out_W, out_H。
    2. 仅 out_W > 0：宽度设为 out_W，高度保持参考高度 ref_H 不变。
    3. 仅 out_H > 0：高度设为 out_H，宽度保持参考宽度 ref_W 不变。
    4. 双 0：保持参考尺寸 ref_W, ref_H。
    
    安全钳制：确保结果 >= 64，<= 8192，且严格为 8 的倍数。
    """
    # 1. 确定基础目标尺寸
    t_W,t_H = 0,0
    longer = max(ref_W,ref_H)
    short = min(ref_W,ref_H)
    divisor = math.gcd(ref_W,ref_H)
    w_ratio, h_ratio = round(ref_W/divisor),round(ref_H/divisor)
    # image1_ratio_str = f'{w_ratio}:{h_ratio}'
    match scaleMode:
        case 'totalPixels':
            total_pixels = megapixels * 1024 * 1024
            # longest_edge = 4096
            # if total_pixels > longest_edge**2:
            #     total_pixels = longest_edge**2
            if megapixels != 0:
                scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
                t_W = round(w_ratio * scale / step) * step
                t_H = round(h_ratio * scale / step) * step
            else:
                 t_W, t_H = ref_W, ref_H
        case 'shortEdge':
            if out_short>120:
                if short == ref_W:
                    t_W, t_H = round(out_short/step)*step, round(out_short/w_ratio*h_ratio/step)*step
                else:
                    t_W, t_H = round(out_short/h_ratio*w_ratio/step)*step,round(out_short/step)*step
            else:
                t_W, t_H = ref_W, ref_H
        case 'longerEdge':
            if out_longer>120:
                if longer == ref_W:
                    t_W, t_H = round(out_longer/step)*step, round(out_longer/w_ratio*h_ratio/step)*step
                else:
                    t_W, t_H = round(out_longer/h_ratio*w_ratio/step)*step,round(out_longer/step)*step
            else:
                t_W, t_H = ref_W, ref_H
    
    return t_W, t_H





# ==========================================
# 辅助函数：Mask 的膨胀 (Expand) 与 腐蚀 (Contract)
# ==========================================
# def mask_morphology(tensor, expand_pixels=0):
#     """
#     tensor shape: [B, 1, H, W], values 0.0 to 1.0
#     expand_pixels > 0: 膨胀 (Dilate)
#     expand_pixels < 0: 腐蚀 (Erode)
#     """
#     if expand_pixels == 0:
#         return tensor
        
#     kernel_size = abs(expand_pixels) * 2 + 1
#     padding = kernel_size // 2
    
#     # 复制边缘填充
#     padded = F.pad(tensor, (padding, padding, padding, padding), mode='replicate')
    
#     if expand_pixels > 0:
#         # 膨胀：使用最大池化
#         return F.max_pool2d(padded, kernel_size=kernel_size, stride=1, padding=0)
#     else:
#         # 腐蚀：使用负数的最大池化 (等效于最小池化)
#         return -F.max_pool2d(-padded, kernel_size=kernel_size, stride=1, padding=0)

# ==========================================
# 辅助函数：极致优化的可分离形态学 (Separable Morphology)
# ==========================================
def mask_morphology_fast(tensor, expand_pixels=0):
    """
    使用可分离池化 (1D 水平 + 1D 垂直) 替代 2D 池化，
    将膨胀/腐蚀的复杂度从 O(K^2) 降至 O(2K)，彻底消除大半径下的卡顿。
    tensor shape: [B, 1, H, W]
    """
    if expand_pixels == 0:
        return tensor
        
    device = tensor.device
    dtype = tensor.dtype
    tensor = tensor.contiguous()
    
    K = abs(expand_pixels) * 2 + 1
    pad = K // 2
    
    if expand_pixels > 0:
        # 膨胀 (Dilation) -> Max Pooling
        # 1. 水平方向
        h_padded = F.pad(tensor, (pad, pad, 0, 0), mode='replicate')
        h_res = F.max_pool2d(h_padded, kernel_size=(1, K), stride=1, padding=0)
        # 2. 垂直方向
        v_padded = F.pad(h_res, (0, 0, pad, pad), mode='replicate')
        return F.max_pool2d(v_padded, kernel_size=(K, 1), stride=1, padding=0)
    else:
        # 腐蚀 (Erosion) -> Min Pooling (即对负数求 Max Pooling，再取负)
        # 1. 水平方向
        h_padded = F.pad(-tensor, (pad, pad, 0, 0), mode='replicate')
        h_res = F.max_pool2d(h_padded, kernel_size=(1, K), stride=1, padding=0)
        # 2. 垂直方向
        v_padded = F.pad(h_res, (0, 0, pad, pad), mode='replicate')
        return -F.max_pool2d(v_padded, kernel_size=(K, 1), stride=1, padding=0)

# ==========================================
# 辅助函数：RGB 与 HSV 互转 (用于高级混合模式)
# ==========================================
def rgb_to_hsv(rgb):
    max_c = torch.max(rgb, dim=-1).values
    min_c = torch.min(rgb, dim=-1).values
    delta = max_c - min_c
    delta_safe = torch.where(delta == 0, torch.ones_like(delta), delta)
    
    h = torch.zeros_like(max_c)
    h = torch.where(max_c == rgb[..., 0], (rgb[..., 1] - rgb[..., 2]) / delta_safe, h)
    h = torch.where(max_c == rgb[..., 1], 2.0 + (rgb[..., 2] - rgb[..., 0]) / delta_safe, h)
    h = torch.where(max_c == rgb[..., 2], 4.0 + (rgb[..., 0] - rgb[..., 1]) / delta_safe, h)
    h = (h / 6.0) % 1.0
    h = torch.where(delta == 0, torch.zeros_like(h), h)
    
    s = torch.where(max_c == 0, torch.zeros_like(max_c), delta / max_c)
    v = max_c
    return torch.stack([h, s, v], dim=-1)

def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    c = v * s
    x = c * (1.0 - torch.abs((h * 6.0) % 2.0 - 1.0))
    m = v - c
    
    h6 = (h * 6.0).to(torch.int64) % 6
    
    # 纯张量向量化操作，无 Python 循环，绝对稳健且高效
    r = torch.where(h6 == 0, c, torch.where(h6 == 1, x, torch.where(h6 == 2, 0.0, torch.where(h6 == 3, 0.0, torch.where(h6 == 4, x, c)))))
    g = torch.where(h6 == 0, x, torch.where(h6 == 1, c, torch.where(h6 == 2, c, torch.where(h6 == 3, x, torch.where(h6 == 4, 0.0, 0.0)))))
    b = torch.where(h6 == 0, 0.0, torch.where(h6 == 1, 0.0, torch.where(h6 == 2, x, torch.where(h6 == 3, c, torch.where(h6 == 4, c, x)))))
    
    return torch.stack([r + m, g + m, b + m], dim=-1)

# ==========================================
# 辅助函数：全功能图层混合模式核心逻辑 (工业级标准)
# ==========================================
def apply_blend_mode(base, blend_color, mode, strength, mask_expanded):
    """
    base: [B, H, W, 3]
    blend_color: [1, 1, 1, 3]
    mask_expanded: [B, H, W, 1]
    strength: float
    """
    # 【核心修复】：将单色 blend_color 广播到与 base 完全相同的形状 [B, H, W, 3]
    # 这彻底解决了后续 HSV 转换和 stack 时的维度不匹配 Bug
    blend = blend_color.expand_as(base)
    
    # 计算混合后的颜色层
    if mode == "normal":
        blended = blend
    elif mode == "multiply":
        blended = base * blend
    elif mode == "screen":
        blended = 1.0 - (1.0 - base) * (1.0 - blend)
    elif mode == "overlay":
        blended = torch.where(base < 0.5, 2.0 * base * blend, 1.0 - 2.0 * (1.0 - base) * (1.0 - blend))
    elif mode == "soft_light":
        blended = torch.where(
            blend < 0.5,
            base - (1.0 - 2.0 * blend) * base * (1.0 - base),
            base + (2.0 * blend - 1.0) * (torch.sqrt(torch.clamp(base, 0.0, 1.0)) - base)
        )
    elif mode == "hard_light":
        blended = torch.where(blend < 0.5, 2.0 * base * blend, 1.0 - 2.0 * (1.0 - base) * (1.0 - blend))
    elif mode == "linear_dodge": # Add
        blended = torch.clamp(base + blend, 0.0, 1.0)
    elif mode == "linear_burn":
        blended = torch.clamp(base + blend - 1.0, 0.0, 1.0)
    elif mode == "color_dodge":
        blended = torch.clamp(base / (1.0 - blend + 1e-5), 0.0, 1.0)
    elif mode == "color_burn":
        blended = 1.0 - torch.clamp((1.0 - base) / (blend + 1e-5), 0.0, 1.0)
    elif mode == "difference":
        blended = torch.abs(base - blend)
    elif mode == "subtract":
        blended = torch.clamp(base - blend, 0.0, 1.0)
    elif mode == "darker": # Darken
        blended = torch.min(base, blend)
    elif mode == "lighter": # Lighten
        blended = torch.max(base, blend)
    elif mode == "vivid_light":
        blended = torch.where(
            blend < 0.5,
            1.0 - torch.clamp((1.0 - base) / (2.0 * blend + 1e-5), 0.0, 1.0),
            torch.clamp(base / (2.0 * (1.0 - blend) + 1e-5), 0.0, 1.0)
        )
    elif mode == "linear_light":
        blended = torch.clamp(base + 2.0 * blend - 1.0, 0.0, 1.0)
    elif mode == "pin_light":
        blended = torch.where(
            blend < 0.5,
            torch.min(base, 2.0 * blend),
            torch.max(base, 2.0 * (blend - 0.5))
        )
    elif mode == "hard_mix":
        blended = (torch.clamp(base + blend, 0.0, 1.0) >= 1.0).to(base.dtype)
    elif mode == "hue":
        hsv_b = rgb_to_hsv(base)
        hsv_c = rgb_to_hsv(blend)
        hsv_res = torch.stack([hsv_c[..., 0], hsv_b[..., 1], hsv_b[..., 2]], dim=-1)
        blended = hsv_to_rgb(hsv_res)
    elif mode == "color":
        hsv_b = rgb_to_hsv(base)
        hsv_c = rgb_to_hsv(blend)
        hsv_res = torch.stack([hsv_c[..., 0], hsv_c[..., 1], hsv_b[..., 2]], dim=-1)
        blended = hsv_to_rgb(hsv_res)
    elif mode == "luminosity":
        hsv_b = rgb_to_hsv(base)
        hsv_c = rgb_to_hsv(blend)
        hsv_res = torch.stack([hsv_b[..., 0], hsv_b[..., 1], hsv_c[..., 2]], dim=-1)
        blended = hsv_to_rgb(hsv_res)
    else:
        blended = base # Fallback

    # 按 Alpha (mask * strength) 进行线性插值混合
    alpha = mask_expanded * strength
    result = base * (1.0 - alpha) + blended * alpha
    return torch.clamp(result, 0.0, 1.0)


# ==========================================
# 【终极兼容版】RMBG 2.0 去背景辅助函数
# 完全复刻 ComfyUI-RMBG 插件的稳健加载逻辑，无需 modelscope/timm
# ==========================================
_RMBG_MODEL_CACHE = None
_RMBG_DEVICE = None

def _get_rmbg_model():
    global _RMBG_MODEL_CACHE, _RMBG_DEVICE
    if _RMBG_MODEL_CACHE is None:
        try:
            from transformers import PreTrainedModel, AutoModelForImageSegmentation # type:ignore
            
            _RMBG_DEVICE = comfy.model_management.get_torch_device()
            
            # 1. 推导本地模型路径
            ckpt_paths = folder_paths.get_folder_paths("checkpoints")
            models_dir = os.path.dirname(ckpt_paths[0]) if ckpt_paths else os.path.join(folder_paths.base_path, "models")
            cache_dir = os.path.join(models_dir, "RMBG", "RMBG-2.0")
            
            if not os.path.isdir(cache_dir):
                raise FileNotFoundError(f"未找到 RMBG 模型目录: {cache_dir}")

            print(f"[afar_tools] Loading RMBG-2.0 from: {cache_dir}")

            # 2. 优先尝试：硬核动态加载 (复刻 ComfyUI-RMBG 插件逻辑，完美避开 config.json 校验)
            model_loaded = False
            try:
                birefnet_path = os.path.join(cache_dir, "birefnet.py")
                birefnet_config_path = os.path.join(cache_dir, "BiRefNet_config.py")

                if os.path.exists(birefnet_config_path) and os.path.exists(birefnet_path):
                    config_spec = importlib.util.spec_from_file_location("BiRefNetConfig", birefnet_config_path)
                    config_module = importlib.util.module_from_spec(config_spec)
                    sys.modules["BiRefNetConfig"] = config_module
                    config_spec.loader.exec_module(config_module)

                    with open(birefnet_path, 'r', encoding='utf-8') as f:
                        birefnet_content = f.read()
                    birefnet_content = birefnet_content.replace(
                        "from .BiRefNet_config import BiRefNetConfig",
                        "from BiRefNetConfig import BiRefNetConfig"
                    )

                    module_name = f"custom_birefnet_{hash(birefnet_path)}"
                    module = types.ModuleType(module_name)
                    sys.modules[module_name] = module
                    exec(birefnet_content, module.__dict__)

                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, PreTrainedModel) and attr != PreTrainedModel:
                            BiRefNetConfig = getattr(config_module, "BiRefNetConfig")
                            _RMBG_MODEL_CACHE = attr(BiRefNetConfig())
                            
                            weights_path = os.path.join(cache_dir, "model.safetensors")
                            if os.path.exists(weights_path):
                                import safetensors.torch    # type:ignore
                                _RMBG_MODEL_CACHE.load_state_dict(safetensors.torch.load_file(weights_path))
                            else:
                                _RMBG_MODEL_CACHE.load_state_dict(torch.load(os.path.join(cache_dir, "pytorch_model.bin"), map_location="cpu"))
                            
                            model_loaded = True
                            print("[afar_tools] RMBG-2.0 loaded via dynamic module execution.")
                            break
            except Exception as dynamic_e:
                print(f"[afar_tools] Dynamic loading failed ({dynamic_e}), falling back to standard transformers...")

            # 3. 回退方案：标准 transformers 加载 (配合自动修复 config.json)
            if not model_loaded:
                config_path = os.path.join(cache_dir, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    if 'model_type' not in config_data or config_data['model_type'] is None:
                        config_data['model_type'] = "custom"
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config_data, f, indent=2)

                _RMBG_MODEL_CACHE = AutoModelForImageSegmentation.from_pretrained(
                    cache_dir, trust_remote_code=True, local_files_only=True
                )
                print("[afar_tools] RMBG-2.0 loaded via standard transformers fallback.")

            # 4. 模型后置处理
            _RMBG_MODEL_CACHE.eval()
            for param in _RMBG_MODEL_CACHE.parameters():
                param.requires_grad = False
            torch.set_float32_matmul_precision('high')
            _RMBG_MODEL_CACHE.to(_RMBG_DEVICE)
            
        except Exception as e:
            raise RuntimeError(f"[afar_tools] 加载 RMBG-2.0 模型彻底失败: {e}")
            
    return _RMBG_MODEL_CACHE, _RMBG_DEVICE

def _run_rmbg_inference(image_tensor):
    """
    使用 RMBG-2.0 对图像进行去背景推理。
    严格处理设备对齐与维度匹配，确保输出正确的 [B, H, W] Mask。
    """
    model, device = _get_rmbg_model()
    
    # 【致命 Bug 修复】：提前记录原始的高和宽，因为 image_tensor 是 [B, H, W, C]
    orig_h, orig_w = image_tensor.shape[1], image_tensor.shape[2]
    
    original_device = image_tensor.device
    image_tensor = image_tensor.to(device)
    
    # 1. 预处理：Permute -> Resize 1024x1024 -> ImageNet Normalize
    img_permuted = image_tensor.permute(0, 3, 1, 2) # [B, C, H, W]
    img_resized = F.interpolate(img_permuted, size=(1024, 1024), mode='bilinear', align_corners=False)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=img_resized.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=img_resized.dtype).view(1, 3, 1, 1)
    img_normalized = (img_resized - mean) / std
    
    # 2. 模型推理
    with torch.no_grad():
        outputs = model(img_normalized)
        if isinstance(outputs, list) and len(outputs) > 0:
            preds = outputs[-1].sigmoid()
        elif isinstance(outputs, dict) and 'logits' in outputs:
            preds = outputs['logits'].sigmoid()
        elif isinstance(outputs, torch.Tensor):
            preds = outputs.sigmoid()
        else:
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    preds = v.sigmoid()
                    break
            else:
                raise RuntimeError("无法识别的模型输出格式")
        
    # 【致命 Bug 修复】：使用记录的 orig_h, orig_w 进行 Resize，彻底杜绝变成 (W, 3) 的竖线
    mask_resized = F.interpolate(preds, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    
    return mask_resized.squeeze(1).to(original_device) # [B, H, W]






# ==========================================
# 需求 1：图像扩展与 Mask 生成 (终极整合版)
# ==========================================
class ImageMask_Pad_Resize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                
                # 1. 基础扩展量 (决定参考画布的总大小)
                "pad_boundary": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "四周统一的基础扩展量"}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                
                # 2. 偏移量 (决定原图在参考画布中的位置，可正可负)
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 8, "tooltip": "原图在参考画布中的水平额外偏移量 (正数向右，负数向左)"}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 8, "tooltip": "原图在参考画布中的垂直额外偏移量 (正数向下，负数向上)"}),

                # 3. 外观控制
                "color": ("STRING", {"default": "0.0", "multiline": False, "tooltip": "支持: 0.5(灰), 255,0,0, 1.0,0.0,0.0, #FF0000, red, white 等"}),
                'fill_hole':('BOOLEAN',{'default':False,'tooltip':'填充input_mask'}),
                "input_mask_blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "display": "number", "tooltip": "仅对输入mask模糊像素半径"}),
                "pad_mask_blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "display": "number", "tooltip": "仅对扩展区域边缘生效的模糊像素半径"}),
                
                # 4. 尺寸锁定 (第一层限制)
                'scaleMode':(['totalPixels','shortEdge','longerEdge','preset','custom'],{'default':'totalPixels'}),
                'megapixels':('FLOAT',{'default':1.0,'min':0.1,'max':16.0,'step':0.1,"tooltip": "以标准百万像素缩放。"}),
                "shortEdge": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "tooltip": "最终输出以较短边为准。64 = 不缩放"}),
                "longerEdge": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "tooltip": "最终输出以较长边为准。64 = 不缩放"}),
                "preset": (aftools.get_resolution_preset(s), {"default":aftools.get_resolution_preset(s)[0], "tooltip": "按分辨率预置应用宽高"}),
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "最终输出宽度。0 = 不限制，非0 = 强制锁定该尺寸"}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "最终输出高度。0 = 不限制，非0 = 强制锁定该尺寸"}),
                "multiple": ([8,16,32,64,128,256],{'default':8}),
                'flip':('BOOLEAN',{'default':False,'tooltip':'翻转宽高'}),
                
                # 5. 约束方式
                "crop": (["disabled", "center"], {"default": "disabled", "tooltip": "disabled: 强制拉伸/压缩参考画布至目标尺寸; center: 保持比例缩放并裁剪参考画布中心"}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "lanczos", "tooltip": "参考画布约束到目标尺寸时使用的插值算法"}),
                
            },
            "optional": {
                "mask": ("MASK", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "afar_tools"

    def execute(self, image, pad_boundary, pad_top, pad_bottom, pad_left, pad_right, 
                offset_x, offset_y, color, input_mask_blur,pad_mask_blur, 
                scaleMode,preset,flip,megapixels,shortEdge,longerEdge,width, height,multiple,
                crop, upscale_method, mask=None,fill_hole=False):
        
        B, H, W, C = image.shape
        device = image.device
        dtype = image.dtype

        # 1. 解析颜色
        r, g, b = parse_color_string(color)

        # 2. 计算参考画布尺寸 (严格由 pad 决定，不受 offset 影响，确保 offset 能实现“移出边界被裁剪”的效果)
        left_pad_total = pad_boundary + pad_left
        right_pad_total = pad_boundary + pad_right
        top_pad_total = pad_boundary + pad_top
        bottom_pad_total = pad_boundary + pad_bottom
        
        ref_W = W + left_pad_total + right_pad_total
        ref_H = H + top_pad_total + bottom_pad_total

        # 3. 计算原图在参考画布中的实际放置坐标 (base_pad + offset)
        total_x_offset = left_pad_total + offset_x
        total_y_offset = top_pad_total + offset_y

        # 4. 计算最终目标尺寸 (应用优先级逻辑 + 8的倍数安全校验)
        target_W, target_H = calc_target_size(scaleMode,ref_W, ref_H, preset,flip,megapixels,shortEdge,longerEdge,width, height,multiple)

        # ==========================================
        # 5. 构建参考画布 (Reference Canvas)
        # ==========================================
        
        # 5.1 参考图像：填充背景色
        ref_image = torch.zeros([B, ref_H, ref_W, C], device=device, dtype=dtype)
        color_tensor = torch.tensor([r, g, b], device=device, dtype=dtype)
        ref_image[:] = color_tensor.view(1, 1, 1, C)

        # 5.2 计算安全切片边界 (处理 offset 导致的原图部分或全部移出参考画布)
        # src: 原图中的有效区域, dst: 参考画布中的放置区域
        src_y_start = max(0, -total_y_offset)
        src_y_end = min(H, ref_H - total_y_offset)
        src_x_start = max(0, -total_x_offset)
        src_x_end = min(W, ref_W - total_x_offset)

        dst_y_start = max(0, total_y_offset)
        dst_y_end = dst_y_start + max(0, src_y_end - src_y_start)
        dst_x_start = max(0, total_x_offset)
        dst_x_end = dst_x_start + max(0, src_x_end - src_x_start)

        # 5.3 将原图有效区域放置到参考画布
        if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
            ref_image[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = \
                image[:, src_y_start:src_y_end, src_x_start:src_x_end, :]

        # 5.4 参考扩展 Mask：初始全 1 (全是扩展区)，原图实际占据的区域设为 0
        ref_pad_mask = torch.ones([B, ref_H, ref_W], device=device, dtype=dtype)
        if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
            ref_pad_mask[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = 0.0

        # 5.5 参考输入 Mask：仅放置在原图实际占据的区域，扩展区为 0
        ref_input_mask = torch.zeros([B, ref_H, ref_W], device=device, dtype=dtype)
        if mask is not None:
            input_mask = mask.to(device=device, dtype=dtype)
            if input_mask.shape[1] != H or input_mask.shape[2] != W:
                input_mask = F.interpolate(input_mask.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
            
            if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
                ref_input_mask[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    input_mask[:, src_y_start:src_y_end, src_x_start:src_x_end]

        # ==========================================
        # 6. 约束到目标尺寸 (Target Constraint) - 【Bug 修复点】
        # ==========================================
        if target_W != ref_W or target_H != ref_H:
            # 图像约束
            img_permuted = ref_image.permute(0, 3, 1, 2)
            target_image = comfy.utils.common_upscale(
                img_permuted, target_W, target_H, upscale_method, crop
            ).permute(0, 2, 3, 1)
            
            # 【修复】Mask 约束：必须使用 common_upscale 以完美继承 crop="center" 的裁剪逻辑！
            # 注意：Mask 强制使用 "bilinear" 插值以保证边缘平滑，crop 模式与图像保持一致。
            pad_mask_permuted = ref_pad_mask.unsqueeze(1) # [B, 1, H, W]
            input_mask_permuted = ref_input_mask.unsqueeze(1) # [B, 1, H, W]
            
            target_pad_mask = comfy.utils.common_upscale(
                pad_mask_permuted, target_W, target_H, "bilinear", crop
            ).squeeze(1) # 恢复为 [B, H, W]
            
            if fill_hole:
                input_mask_permuted = mask_fill_hole(input_mask_permuted)
            target_input_mask = comfy.utils.common_upscale(
                input_mask_permuted, target_W, target_H, "bilinear", crop
            ).squeeze(1)
        else:
            target_image = ref_image
            target_pad_mask = ref_pad_mask
            target_input_mask = ref_input_mask

        # ==========================================
        # 7. 隔离模糊与最终合并
        # ==========================================
        if input_mask_blur > 0:
            target_input_mask_4d = target_input_mask.unsqueeze(1)
            i_size = input_mask_blur * 2 + 1
            target_input_mask_4d = gaussian_blur_2d_fast(target_input_mask_4d, kernel_size=i_size, sigma=input_mask_blur / 3.0)
            target_input_mask = target_input_mask_4d.squeeze(1)
        if pad_mask_blur > 0:
            target_pad_mask_4d = target_pad_mask.unsqueeze(1)
            k_size = pad_mask_blur * 2 + 1
            target_pad_mask_4d = gaussian_blur_2d_fast(target_pad_mask_4d, kernel_size=k_size, sigma=pad_mask_blur / 3.0)
            target_pad_mask = target_pad_mask_4d.squeeze(1)

        final_mask = torch.max(target_pad_mask, target_input_mask)
        final_mask = torch.clamp(final_mask, 0.0, 1.0)

        return (target_image, final_mask)


# ==========================================
# 需求 2：基于 Mask 的裁剪、缩放与形态学处理 (集成 RMBG 兜底与后置去背景，完美支持 invert_mask)
# ==========================================
class CropByMask_Resize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                
                # 1. Mask 中间操作
                "fill_hole": ("BOOLEAN", {"default": False, "tooltip": "填充 Mask 内部不与边界相连的闭合孔洞"}),
                "mask_expand": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1, "tooltip": "膨胀(正)/腐蚀(负)"}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "Mask 边缘模糊半径"}),
                
                # 2. 边界扩展与对齐
                "crop_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "裁剪框放大比例"}),
                "crop_by_box": ("BOOLEAN", {"default": False, "tooltip": "强制 1:1 正方形裁切"}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "反转裁切后的局部 Mask (用于贴回时挖空，或去背景时反转主体/背景)"}),
                
                # 3. 二次缩放控制
                # "target_longer_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "tooltip": "输出较长边尺寸"}),
                
                'scaleMode':(['totalPixels','shortEdge','longerEdge'],{'default':'totalPixels'}),
                'megapixels':('FLOAT',{'default':1.0,'min':0.0,'max':16.0,'step':0.1,"tooltip": "以标准百万像素缩放。0 = 不缩放"}),
                "shortEdge": ("INT", {"default": 1024, "min": 120, "max": 4096, "step": 8, "tooltip": "最终输出以较短边为准。120 = 不缩放"}),
                "longerEdge": ("INT", {"default": 1024, "min": 120, "max": 4096, "step": 8, "tooltip": "最终输出以较长边为准。120 = 不缩放"}),
                "multiple": ([8,16,32,64,128,256],{'default':16,"tooltip": "尺寸对齐倍数"}),
                
                # 4. 预览混合控制
                "mask_preview_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "hard_light", 
                                "linear_dodge", "linear_burn", "color_dodge", "color_burn", 
                                "vivid_light", "linear_light", "pin_light", "hard_mix",
                                "difference", "subtract", "darker", "lighter", 
                                "hue", "color", "luminosity"], {"default": "normal", "tooltip": "预览混合模式"}),
                "mask_color": ("STRING", {"default": "#00ff00", "multiline": False, "tooltip": "预览颜色"}),
                "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "混合强度"}),                

                # 5. RMBG 去背景控制 (后置处理)
                "removeBG": ("BOOLEAN", {"default": False, "tooltip": "开启后，使用 RMBG-2.0 对裁切缩放后的图像进行精准去背景"}),
                "bg_type": (['alpha', 'color'], {"default": 'alpha', "tooltip": "alpha: 输出透明背景(4通道); color: 输出指定颜色背景(3通道)"}),
                "bg_color": ("STRING", {"default": "#ffffff", "multiline": False, "tooltip": "当 bg_type 为 color 时生效的背景颜色"}),
            },
            "optional": {
                "mask": ("MASK", {"default": None, "tooltip": "手动接入的 Mask。若未连接，将自动使用 RMBG-2.0 生成初始 Mask"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "DICT", "IMAGE")
    RETURN_NAMES = ("original_image", "crop_image", "crop_mask", "crop_data", "crop_preview")
    FUNCTION = "execute"
    CATEGORY = "afar_tools"

    def execute(self, image, fill_hole, mask_expand, mask_blur, invert_mask, 
                scaleMode,megapixels,shortEdge,longerEdge,multiple,
                crop_factor, crop_by_box, 
                mask_preview_mode, mask_color, blend_strength, removeBG, bg_type, bg_color, mask=None):
        B, H, W, C = image.shape
        device = image.device
        dtype = image.dtype

        # ==========================================
        # Step 0: Mask 来源判定 (前置 RMBG 兜底)
        # ==========================================
        if mask is None:
            print("[afar_tools] No mask provided, using RMBG-2.0 to generate initial mask.")
            mask_tensor = _run_rmbg_inference(image) # [B, H, W]
        else:
            mask_tensor = mask
            if mask_tensor.shape[1] != H or mask_tensor.shape[2] != W:
                mask_tensor = F.interpolate(mask_tensor.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)

        # ==========================================
        # Step 1: Mask 中间操作 (Ops 管线，此时不反转)
        # ==========================================
        mask_4d = mask_tensor.unsqueeze(1).to(device=device, dtype=dtype)
        
        if fill_hole:
            mask_4d = mask_fill_hole(mask_4d)
            
        processed_mask_4d = mask_morphology_fast(mask_4d, expand_pixels=mask_expand)
        if mask_blur > 0:
            k_size = mask_blur * 2 + 1
            processed_mask_4d = gaussian_blur_2d_fast(processed_mask_4d, kernel_size=k_size, sigma=mask_blur / 3.0)
            
        processed_mask = processed_mask_4d.squeeze(1)

        # ==========================================
        # Step 2: 计算 Crop Data 与 裁切 (基于正向 Mask)
        # ==========================================
        valid_indices = torch.nonzero(processed_mask > 0.1)
        if valid_indices.numel() == 0:
            raw_x, raw_y, raw_w, raw_h = 0, 0, W, H
            center_x, center_y = W / 2.0, H / 2.0
        else:
            x_min_raw = int(valid_indices[:, 2].min().item())
            x_max_raw = int(valid_indices[:, 2].max().item()) + 1
            y_min_raw = int(valid_indices[:, 1].min().item())
            y_max_raw = int(valid_indices[:, 1].max().item()) + 1
            raw_w = x_max_raw - x_min_raw
            raw_h = y_max_raw - y_min_raw
            center_x = (x_min_raw + x_max_raw) / 2.0
            center_y = (y_min_raw + y_max_raw) / 2.0

        new_w = raw_w * crop_factor
        new_h = raw_h * crop_factor
        if crop_by_box:
            max_side = max(new_w, new_h)
            new_w = new_h = max_side

        new_w = max(int(multiple), int((new_w + int(multiple) - 1) // int(multiple) * int(multiple)))
        new_h = max(int(multiple), int((new_h + int(multiple) - 1) // int(multiple) * int(multiple)))
        max_valid_w = max(int(multiple), (W // int(multiple)) * int(multiple))
        max_valid_h = max(int(multiple), (H // int(multiple)) * int(multiple))
        target_w = min(new_w, max_valid_w)
        target_h = min(new_h, max_valid_h)

        x_min = int(center_x - target_w / 2.0)
        x_max = x_min + target_w
        y_min = int(center_y - target_h / 2.0)
        y_max = y_min + target_h

        if x_min < 0: x_min = 0; x_max = target_w
        elif x_max > W: x_max = W; x_min = W - target_w
        if y_min < 0: y_min = 0; y_max = target_h
        elif y_max > H: y_max = H; y_min = H - target_h

        crop_data = {
            "original_size": (int(H), int(W)),
            "crop_box": (int(x_min), int(y_min), int(target_w), int(target_h))
        }

        image_crop = image[:, y_min:y_max, x_min:x_max, :]
        mask_crop_4d = processed_mask_4d[:, :, y_min:y_max, x_min:x_max]

        # ==========================================
        # Step 3: 对裁切后的局部 Mask 执行 Invert
        # ==========================================
        if invert_mask:
            mask_crop_4d = 1.0 - mask_crop_4d

        # ==========================================
        # Step 4: 缩放 (Scaling)
        # ==========================================
        current_w = x_max - x_min
        current_h = y_max - y_min
        scale_w, scale_h = crop_target_size(scaleMode,current_w, current_h, megapixels,shortEdge,longerEdge,multiple)

        # if current_w >= current_h:
        #     scale_w = target_longer_size
        #     scale_h = int(round(target_longer_size * (current_h / current_w)))
        # else:
        #     scale_h = target_longer_size
        #     scale_w = int(round(target_longer_size * (current_w / current_h)))
            
        # safe_scale_w = max(int(multiple), int((scale_w + int(multiple) - 1) // int(multiple) * int(multiple)))
        # safe_scale_h = max(int(multiple), int((scale_h + int(multiple) - 1) // int(multiple) * int(multiple)))

        img_permuted = image_crop.permute(0, 3, 1, 2)
        image_scale = comfy.utils.common_upscale(img_permuted, scale_w, scale_h, "bilinear", "disabled").permute(0, 2, 3, 1)
        
        mask_scale_4d = comfy.utils.common_upscale(mask_crop_4d, scale_w, scale_h, "bilinear", "disabled")
        mask_scale = torch.clamp(mask_scale_4d.squeeze(1), 0.0, 1.0)

        # ==========================================
        # Step 5: Image 去背景处理 (后置 RMBG，完美响应 invert_mask)
        # ==========================================
        if removeBG:
            print("[afar_tools] Applying RMBG-2.0 background removal to cropped image.")
            rmbg_mask = _run_rmbg_inference(image_scale) # [B, H, W]
            
            # 【核心修复】：如果开启了 invert_mask，去背景的 Mask 也需要相应翻转
            # 这样“填充颜色”或“透明”的效果才会正确作用于翻转后的区域
            effective_rmbg_mask = (1.0 - rmbg_mask) if invert_mask else rmbg_mask
            
            if bg_type == 'alpha':
                rmbg_mask_expanded = effective_rmbg_mask.unsqueeze(-1) # [B, H, W, 1]
                image_scale = torch.cat([image_scale, rmbg_mask_expanded], dim=-1) # [B, H, W, 4]
                
            elif bg_type == 'color':
                r, g, b = parse_color_string(bg_color)
                bg_tensor = torch.tensor([r, g, b], device=device, dtype=dtype).view(1, 1, 1, 3)
                rmbg_mask_expanded = effective_rmbg_mask.unsqueeze(-1)
                # Alpha 混合公式
                image_scale = image_scale * rmbg_mask_expanded + bg_tensor * (1.0 - rmbg_mask_expanded)
                image_scale = torch.clamp(image_scale, 0.0, 1.0)

        # ==========================================
        # Step 6: 生成 Crop Preview
        # ==========================================
        preview_image = image_scale
        if removeBG and bg_type == 'alpha' and image_scale.shape[-1] == 4:
            rgb_part = image_scale[..., :3]
            alpha_part = image_scale[..., 3:]
            preview_image = rgb_part * alpha_part

        r, g, b = parse_color_string(mask_color)
        blend_color_t = torch.tensor([r, g, b], device=device, dtype=dtype).view(1, 1, 1, 3)
        mask_expanded = mask_scale.unsqueeze(-1) # [B, H, W, 1]
        
        crop_preview = apply_blend_mode(preview_image, blend_color_t, mask_preview_mode, blend_strength, mask_expanded)

        # ==========================================
        # Step 7: 返回结果
        # ==========================================
        return (image, image_scale, mask_scale, crop_data, crop_preview)


# ==========================================
# 需求 2 升级版 v2：基于 Mask/SAM3/RMBG 的智能裁剪、缩放与形态学处理 (终极容错版)
# ==========================================
class CropByMask_Resize_sam3:
    @classmethod
    def INPUT_TYPES(s):
        # 1. 获取 checkpoints 目录的文件列表
        checkpoint_files = folder_paths.get_filename_list("checkpoints")
        
        # 2. 尝试获取 sam3 目录的文件列表
        sam3_files = []
        if "sam3" in folder_paths.folder_names_and_paths:
            try:
                sam3_files = folder_paths.get_filename_list("sam3")
            except Exception:
                pass
                
        if not sam3_files:
            try:
                ckpt_paths = folder_paths.get_folder_paths("checkpoints")
                if ckpt_paths:
                    models_dir = os.path.dirname(ckpt_paths[0])
                    sam3_dir = os.path.join(models_dir, "sam3")
                    if os.path.isdir(sam3_dir):
                        physical_sam3_files = [
                            f for f in os.listdir(sam3_dir) 
                            if f.lower().endswith(('.safetensors', '.ckpt', '.pt', '.pth', '.bin'))
                        ]
                        sam3_files = physical_sam3_files
            except Exception:
                pass

        # 3. 合并并去重
        all_sam_candidates = checkpoint_files + sam3_files
        sam_checkpoints = sorted([f for f in all_sam_candidates if 'sam' in f.lower()])

        if not sam_checkpoints:
            sam_checkpoints = ["No_SAM_Models_Found"]

        return {
            "required": {
                "image": ("IMAGE",),
                
                # 1. Mask 中间操作
                "fill_hole": ("BOOLEAN", {"default": False, "tooltip": "填充 Mask 内部不与边界相连的闭合孔洞"}),
                "mask_expand": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1, "tooltip": "膨胀(正)/腐蚀(负)"}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "Mask 边缘模糊半径"}),
                
                # 2. 边界扩展与对齐
                "crop_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "裁剪框放大比例"}),
                "crop_by_box": ("BOOLEAN", {"default": False, "tooltip": "强制 1:1 正方形裁切"}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "反转裁切后的局部 Mask (用于贴回时挖空，或去背景时反转主体/背景)"}),
                
                # 3. 二次缩放控制
                # "target_longer_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8, "tooltip": "输出较长边尺寸"}),
                
                'scaleMode':(['totalPixels','shortEdge','longerEdge'],{'default':'totalPixels'}),
                'megapixels':('FLOAT',{'default':1.0,'min':0.0,'max':16.0,'step':0.1,"tooltip": "以标准百万像素缩放。0 = 不缩放"}),
                "shortEdge": ("INT", {"default": 1024, "min": 120, "max": 4096, "step": 8, "tooltip": "最终输出以较短边为准。120 = 不缩放"}),
                "longerEdge": ("INT", {"default": 1024, "min": 120, "max": 4096, "step": 8, "tooltip": "最终输出以较长边为准。120 = 不缩放"}),
                # "preset": (aftools.get_resolution_preset(s), {"default":aftools.get_resolution_preset(s)[0], "tooltip": "按分辨率预置应用宽高"}),
                # "width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "最终输出宽度。0 = 不限制，非0 = 强制锁定该尺寸"}),
                # "height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "最终输出高度。0 = 不限制，非0 = 强制锁定该尺寸"}),
                "multiple": ([8,16,32,64,128,256],{'default':16,"tooltip": "尺寸对齐倍数"}),
                # 'flip':('BOOLEAN',{'default':False,'tooltip':'翻转宽高'}),




                
                # 4. 预览混合控制
                "mask_preview_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "hard_light", 
                                "linear_dodge", "linear_burn", "color_dodge", "color_burn", 
                                "vivid_light", "linear_light", "pin_light", "hard_mix",
                                "difference", "subtract", "darker", "lighter", 
                                "hue", "color", "luminosity"], {"default": "normal", "tooltip": "预览混合模式"}),
                "mask_color": ("STRING", {"default": "#00ff00", "multiline": False, "tooltip": "预览颜色"}),
                "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "混合强度"}),

                # 5. SAM3 Detect 控制参数
                "prompt": ("STRING", {"default": "", "multiline": False, "tooltip": "SAM3 文本提示词 (若为空或模型不存在，则自动降级使用 RMBG-2.0)"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "SAM3 检测置信度阈值"}),
                "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1, "tooltip": "SAM3 解码器细化迭代次数"}),
                "individual_masks": ("BOOLEAN", {"default": False, "tooltip": "SAM3 输出每个对象的独立掩码"}),
                "sam_model_name": (sam_checkpoints, {"tooltip": "选择 SAM3.1 模型 (自动检索并去重)"}),

                # 6. RMBG 去背景控制 (后置处理)
                "removeBG": ("BOOLEAN", {"default": True, "tooltip": "开启后，使用 RMBG-2.0 对裁切缩放后的图像进行精准去背景"}),
                "bg_type": (['alpha', 'color'], {"default": 'color', "tooltip": "alpha: 输出透明背景(4通道); color: 输出指定颜色背景(3通道)"}),
                "bg_color": ("STRING", {"default": "#ffffff", "multiline": False, "tooltip": "当 bg_type 为 color 时生效的背景颜色"}),
            },
            "optional": {
                "mask": ("MASK", {"default": None, "tooltip": "手动接入的 Mask (优先级最高)。若未连接，将根据模型存在性与 prompt 决定使用 SAM3 或 RMBG"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "DICT", "IMAGE", "BOUNDING_BOX")
    RETURN_NAMES = ("original_image", "crop_image", "crop_mask", "crop_data", "crop_preview", "bboxes")
    FUNCTION = "execute"
    CATEGORY = "afar_tools"

    def execute(self, image, fill_hole, mask_expand, mask_blur, invert_mask, 
                scaleMode,megapixels,shortEdge,longerEdge,multiple,
                crop_factor, crop_by_box, mask_preview_mode, mask_color, blend_strength,
                prompt, threshold, refine_iterations, individual_masks, sam_model_name, removeBG, bg_type, bg_color, mask=None):
        
        B, H, W, C = image.shape
        device = image.device
        dtype = image.dtype
        
        bboxes = [[] for _ in range(B)] 
        mask_tensor = None

        # ==========================================
        # Step 0: Mask 来源智能判定管线 (终极容错版)
        # ==========================================
        if mask is not None:
            # 优先级 1: 用户手动接入的 mask
            mask_tensor = mask
            if mask_tensor.shape[1] != H or mask_tensor.shape[2] != W:
                mask_tensor = F.interpolate(mask_tensor.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
        else:
            # 优先级 2 & 3: 检查 SAM3 模型是否存在以及 prompt 是否为空
            
            # 1. 提前校验模型文件是否存在
            ckpt_paths = folder_paths.get_folder_paths("checkpoints")
            models_dir = os.path.dirname(ckpt_paths[0]) if ckpt_paths else os.path.join(folder_paths.base_path, "models")
            
            sam3_model_exists = False
            if os.path.exists(os.path.join(models_dir, sam_model_name)):
                sam3_model_exists = True
            elif os.path.exists(os.path.join(models_dir, "sam3", sam_model_name)):
                sam3_model_exists = True
                
            prompt_clean = str(prompt).strip()
            
            # 2. 只有在模型存在 且 prompt 不为空时，才尝试执行 SAM3
            if sam3_model_exists and prompt_clean:
                print(f"[afar_tools] No manual mask, using SAM3 with prompt: '{prompt_clean}'")
                try:
                    # 确定具体路径
                    if os.path.exists(os.path.join(models_dir, sam_model_name)):
                        ckpt_file = os.path.join(models_dir, sam_model_name)
                    else:
                        ckpt_file = os.path.join(models_dir, "sam3", sam_model_name)
                        
                    model, clip = comfy.sd.load_checkpoint_guess_config(
                        ckpt_file, output_vae=True, output_clip=True, 
                        embedding_directory=folder_paths.get_folder_paths("embeddings")
                    )[:2]
                    
                    clip_encode_node = CLIPTextEncode()
                    conditioning, = clip_encode_node.encode(clip, prompt_clean)
                    
                    mask_sam3, detected_bboxes = sam3_module.SAM3_Detect.execute(
                        model=model, image=image, conditioning=conditioning, 
                        threshold=threshold, refine_iterations=refine_iterations, individual_masks=individual_masks
                    )
                    
                    if isinstance(detected_bboxes, list) and len(detected_bboxes) > 0:
                        bboxes = detected_bboxes
                        
                    # 维度规范化
                    if mask_sam3.dim() == 4:
                        mask_sam3 = mask_sam3.squeeze(1)
                    elif mask_sam3.dim() == 2:
                        mask_sam3 = mask_sam3.unsqueeze(0)
                    if mask_sam3.dim() == 3 and mask_sam3.shape[0] != B:
                        mask_sam3 = mask_sam3.max(dim=0, keepdim=True).values
                        mask_sam3 = mask_sam3.repeat(B, 1, 1)
                        
                    mask_tensor = mask_sam3.to(device=device, dtype=dtype)
                    
                except Exception as e:
                    print(f"[afar_tools] SAM3 execution failed ({e}), falling back to RMBG-2.0.")
                    mask_tensor = _run_rmbg_inference(image)
            else:
                # 优先级 3: 模型不存在 或 prompt 为空，直接 RMBG 兜底
                if not sam3_model_exists:
                    print(f"[afar_tools] SAM3 model '{sam_model_name}' not found. Falling back to RMBG-2.0.")
                else:
                    print("[afar_tools] No manual mask and empty prompt, using RMBG-2.0 as fallback.")
                mask_tensor = _run_rmbg_inference(image)

        # ==========================================
        # Step 1: Mask 中间操作 (Ops 管线，此时不反转)
        # ==========================================
        mask_4d = mask_tensor.unsqueeze(1).to(device=device, dtype=dtype)
        
        if fill_hole:
            mask_4d = mask_fill_hole(mask_4d)
            
        processed_mask_4d = mask_morphology_fast(mask_4d, expand_pixels=mask_expand)
        if mask_blur > 0:
            k_size = mask_blur * 2 + 1
            processed_mask_4d = gaussian_blur_2d_fast(processed_mask_4d, kernel_size=k_size, sigma=mask_blur / 3.0)
            
        processed_mask = processed_mask_4d.squeeze(1)

        # ==========================================
        # Step 2: 计算 Crop Data 与 裁切 (基于正向 Mask)
        # ==========================================
        valid_indices = torch.nonzero(processed_mask > 0.1)
        if valid_indices.numel() == 0:
            raw_x, raw_y, raw_w, raw_h = 0, 0, W, H
            center_x, center_y = W / 2.0, H / 2.0
        else:
            x_min_raw = int(valid_indices[:, 2].min().item())
            x_max_raw = int(valid_indices[:, 2].max().item()) + 1
            y_min_raw = int(valid_indices[:, 1].min().item())
            y_max_raw = int(valid_indices[:, 1].max().item()) + 1
            raw_w = x_max_raw - x_min_raw
            raw_h = y_max_raw - y_min_raw
            center_x = (x_min_raw + x_max_raw) / 2.0
            center_y = (y_min_raw + y_max_raw) / 2.0

        new_w = raw_w * crop_factor
        new_h = raw_h * crop_factor
        if crop_by_box:
            max_side = max(new_w, new_h)
            new_w = new_h = max_side

        new_w = max(int(multiple), int((new_w + int(multiple) - 1) // int(multiple) * int(multiple)))
        new_h = max(int(multiple), int((new_h + int(multiple) - 1) // int(multiple) * int(multiple)))
        max_valid_w = max(int(multiple), (W // int(multiple)) * int(multiple))
        max_valid_h = max(int(multiple), (H // int(multiple)) * int(multiple))
        target_w = min(new_w, max_valid_w)
        target_h = min(new_h, max_valid_h)

        x_min = int(center_x - target_w / 2.0)
        x_max = x_min + target_w
        y_min = int(center_y - target_h / 2.0)
        y_max = y_min + target_h

        if x_min < 0: x_min = 0; x_max = target_w
        elif x_max > W: x_max = W; x_min = W - target_w
        if y_min < 0: y_min = 0; y_max = target_h
        elif y_max > H: y_max = H; y_min = H - target_h

        crop_data = {
            "original_size": (int(H), int(W)),
            "crop_box": (int(x_min), int(y_min), int(target_w), int(target_h))
        }

        image_crop = image[:, y_min:y_max, x_min:x_max, :]
        mask_crop_4d = processed_mask_4d[:, :, y_min:y_max, x_min:x_max]

        # ==========================================
        # Step 3: 对裁切后的局部 Mask 执行 Invert
        # ==========================================
        if invert_mask:
            mask_crop_4d = 1.0 - mask_crop_4d

        # ==========================================
        # Step 4: 缩放 (Scaling)
        # ==========================================
        # 4. 计算最终目标尺寸 (应用优先级逻辑 + 8的倍数安全校验)

        current_w = x_max - x_min
        current_h = y_max - y_min
        scale_w, scale_h = crop_target_size(scaleMode,current_w, current_h, megapixels,shortEdge,longerEdge,multiple)
        # if target_W >= target_H:
        #     scale_w = target_longer_size
        #     scale_h = int(round(target_longer_size * (target_H / target_W)))
        # else:
        #     scale_h = target_longer_size
        #     scale_w = int(round(target_longer_size * (target_W / target_H)))

        # if current_w >= current_h:
        #     scale_w = target_longer_size
        #     scale_h = int(round(target_longer_size * (current_h / current_w)))
        # else:
        #     scale_h = target_longer_size
        #     scale_w = int(round(target_longer_size * (current_w / current_h)))
            
        # safe_scale_w = max(int(multiple), int((scale_w + int(multiple) - 1) // int(multiple) * int(multiple)))
        # safe_scale_h = max(int(multiple), int((scale_h + int(multiple) - 1) // int(multiple) * int(multiple)))

        img_permuted = image_crop.permute(0, 3, 1, 2)
        image_scale = comfy.utils.common_upscale(img_permuted, scale_w, scale_h, "bilinear", "disabled").permute(0, 2, 3, 1)
        
        mask_scale_4d = comfy.utils.common_upscale(mask_crop_4d, scale_w, scale_h, "bilinear", "disabled")
        mask_scale = torch.clamp(mask_scale_4d.squeeze(1), 0.0, 1.0)







        # ==========================================
        # Step 5: Image 去背景处理 (后置 RMBG，完美响应 invert_mask)
        # ==========================================
        if removeBG:
            print("[afar_tools] Applying RMBG-2.0 background removal to cropped image.")
            rmbg_mask = _run_rmbg_inference(image_scale) # [B, H, W]
            
            # 如果开启了 invert_mask，去背景的 Mask 也需要相应翻转
            effective_rmbg_mask = (1.0 - rmbg_mask) if invert_mask else rmbg_mask
            
            if bg_type == 'alpha':
                rmbg_mask_expanded = effective_rmbg_mask.unsqueeze(-1) # [B, H, W, 1]
                image_scale = torch.cat([image_scale, rmbg_mask_expanded], dim=-1) # [B, H, W, 4]
                
            elif bg_type == 'color':
                r, g, b = parse_color_string(bg_color)
                bg_tensor = torch.tensor([r, g, b], device=device, dtype=dtype).view(1, 1, 1, 3)
                rmbg_mask_expanded = effective_rmbg_mask.unsqueeze(-1)
                image_scale = image_scale * rmbg_mask_expanded + bg_tensor * (1.0 - rmbg_mask_expanded)
                image_scale = torch.clamp(image_scale, 0.0, 1.0)

        # ==========================================
        # Step 6: 生成 Crop Preview
        # ==========================================
        preview_image = image_scale
        if removeBG and bg_type == 'alpha' and image_scale.shape[-1] == 4:
            rgb_part = image_scale[..., :3]
            alpha_part = image_scale[..., 3:]
            preview_image = rgb_part * alpha_part

        r, g, b = parse_color_string(mask_color)
        blend_color_t = torch.tensor([r, g, b], device=device, dtype=dtype).view(1, 1, 1, 3)
        mask_expanded = mask_scale.unsqueeze(-1) # [B, H, W, 1]
        
        crop_preview = apply_blend_mode(preview_image, blend_color_t, mask_preview_mode, blend_strength, mask_expanded)

        # ==========================================
        # Step 7: 返回结果
        # ==========================================
        return (image, image_scale, mask_scale, crop_data, crop_preview, bboxes)

# ==========================================
# 需求 3：自动化贴回与 Blend 调节 (含 4通道 RGBA 兼容与边缘二次柔化)
# ==========================================
class CropByMask_Restore:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE", {"tooltip": "原始大图"}),
                "crop_image": ("IMAGE", {"tooltip": "需求2输出的局部处理图 (支持 3通道 RGB 或 4通道 RGBA)"}),
                "crop_mask": ("MASK", {"tooltip": "需求2输出的局部 Mask"}),
                "crop_data": ("DICT", {"tooltip": "需求2输出的坐标与尺寸字典 (crop_data)"}),
                
                # 融合控制
                "sharp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "【仅对crop_image生效】还原后的局部图锐化强度 (0.0 = 不锐化)"}),
                "blend_edge": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "【仅对crop_mask生效】还原后的局部 Mask 边缘二次高斯模糊程度 (内部已乘2增强)"}),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "hard_light", 
                                "linear_dodge", "linear_burn", "color_dodge", "color_burn", 
                                "vivid_light", "linear_light", "pin_light", "hard_mix",
                                "difference", "subtract", "darker", "lighter", 
                                "hue", "color", "luminosity"], 
                               {"default": "normal", "tooltip": "贴回时使用的图层混合模式"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "整体贴回融合强度 (0.0=完全保留原图, 1.0=完全贴回)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("final_image", "original_image", "final_mask")
    FUNCTION = "execute"
    CATEGORY = "afar_tools"

    def execute(self, original_image, crop_image, crop_mask, crop_data, sharp, blend_edge, blend_mode, blend_strength):
        B, H, W, C = original_image.shape
        device = original_image.device
        dtype = original_image.dtype

        # ==========================================
        # 【核心修复】智能处理 4通道 (RGBA) 输入
        # ==========================================
        if crop_image.shape[-1] == 4:
            # 1. 提取 Alpha 通道 [B, H, W, 1]
            alpha_channel = crop_image[..., 3:4]
            # 2. 提取 RGB 通道用于后续贴回 [B, H, W, 3]
            crop_image = crop_image[..., :3]
            
            # 3. 将 RMBG 的 Alpha 通道与原有的 crop_mask 融合
            # 这样既保留了需求2中的 expand/blur 效果，又应用了 RMBG 的精准边缘
            mask_expanded = crop_mask.unsqueeze(-1) # [B, H, W, 1]
            crop_mask = (mask_expanded * alpha_channel).squeeze(-1)
            crop_mask = torch.clamp(crop_mask, 0.0, 1.0)

        # ==========================================
        # Step 1: 强制规范化 crop_mask 为严格的 3D [B, H, W]
        # ==========================================
        if crop_mask.dim() == 4:
            if crop_mask.shape[1] == 1:
                crop_mask = crop_mask.squeeze(1)
            elif crop_mask.shape[-1] == 1:
                crop_mask = crop_mask.squeeze(-1)
        elif crop_mask.dim() == 2:
            crop_mask = crop_mask.unsqueeze(0)
        
        # 确保 crop_mask 的 batch 维度与 original_image 一致
        if crop_mask.shape[0] != B:
            if crop_mask.shape[0] < B:
                crop_mask = crop_mask.repeat(B // crop_mask.shape[0] + 1, 1, 1)[:B]
            else:
                crop_mask = crop_mask[:B]

        # ==========================================
        # Step 2: 从 crop_data 提取核心坐标与尺寸
        # ==========================================
        x_min, y_min, crop_w, crop_h = crop_data["crop_box"]
        x_min, y_min, crop_w, crop_h = int(x_min), int(y_min), int(crop_w), int(crop_h)

        # ==========================================
        # Step 3: 将局部图和 Mask 还原到裁剪时的尺寸
        # ==========================================
        # 1. 还原 crop_image (此时保证是 3 通道)
        img_permuted = crop_image.permute(0, 3, 1, 2)
        restored_img = comfy.utils.common_upscale(
            img_permuted, crop_w, crop_h, "lanczos", "disabled"
        ).permute(0, 2, 3, 1)
        
        # 2. 还原 crop_mask
        mask_permuted = crop_mask.unsqueeze(1) # [B, 1, h, w]
        restored_mask = comfy.utils.common_upscale(
            mask_permuted, crop_w, crop_h, "bilinear", "disabled" 
        ).squeeze(1) # [B, crop_h, crop_w]

        # ==========================================
        # Step 4: 锐化与边缘模糊
        # ==========================================
        if sharp > 0.0:
            restored_img = apply_sharpness(restored_img, sharp)

        if blend_edge > 0.0:
            effective_blend_edge = blend_edge * 4.0
            blur_radius = int(effective_blend_edge * 50)
            
            if blur_radius > 0:
                k_size = blur_radius * 2 + 1
                sigma = max(0.1, blur_radius / 3.0)
                
                restored_mask_4d = restored_mask.unsqueeze(1)
                restored_mask_4d = gaussian_blur_2d_fast(restored_mask_4d, kernel_size=k_size, sigma=sigma)
                restored_mask = restored_mask_4d.squeeze(1)
                
            restored_mask = torch.clamp(restored_mask, 0.0, 1.0)

        # ==========================================
        # Step 5: 创建同尺寸画布并进行安全贴回 (自动 Offset)
        # ==========================================
        # 此时 original_image 和 restored_img 都是 3 通道，完美匹配
        patch_canvas = torch.zeros_like(original_image)
        mask_canvas = torch.zeros([B, H, W], device=device, dtype=dtype)
        
        src_y_start = max(0, -y_min)
        src_y_end = min(crop_h, H - y_min)
        src_x_start = max(0, -x_min)
        src_x_end = min(crop_w, W - x_min)
        
        dst_y_start = max(0, y_min)
        dst_y_end = dst_y_start + max(0, src_y_end - src_y_start)
        dst_x_start = max(0, x_min)
        dst_x_end = dst_x_start + max(0, src_x_end - src_x_start)
        
        if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
            patch_canvas[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = \
                restored_img[:, src_y_start:src_y_end, src_x_start:src_x_end, :]
            
            mask_canvas[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                restored_mask[:, src_y_start:src_y_end, src_x_start:src_x_end]

        # ==========================================
        # Step 6: 应用 Blend 模式与强度，生成最终图像
        # ==========================================
        final_mask_expanded = mask_canvas.unsqueeze(-1)
        final_image = apply_blend_mode(
            base=original_image, 
            blend_color=patch_canvas, 
            mode=blend_mode, 
            strength=blend_strength, 
            mask_expanded=final_mask_expanded
        )

        # ==========================================
        # Step 7: 最终输出前的维度兜底检查
        # ==========================================
        if mask_canvas.dim() == 4:
            if mask_canvas.shape[1] == 1:
                mask_canvas = mask_canvas.squeeze(1)
            elif mask_canvas.shape[-1] == 1:
                mask_canvas = mask_canvas.squeeze(-1)
        if mask_canvas.shape[0] != B:
            mask_canvas = mask_canvas[:B] if mask_canvas.shape[0] > B else mask_canvas.repeat(B, 1, 1)[:B]

        return (final_image, original_image, mask_canvas)


# ==========================================
# 需求 4：独立的 Mask 孔洞填充与形态学处理节点
# ==========================================
class Fill_Mask_Holes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "输入的原始 Mask"}),
                "mask_expand": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1, "tooltip": "填充后对 Mask 进行膨胀(正数)或腐蚀(负数)"}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "对 Mask 边缘进行高斯模糊的像素半径"}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "在所有操作完成后，反转 Mask (1.0 - mask)"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = "afar_tools"

    def execute(self, mask, mask_expand, mask_blur, invert_mask):
        device = mask.device
        dtype = mask.dtype

        # ==========================================
        # 1. 强制规范化输入 mask 为严格的 3D [B, H, W]
        # ==========================================
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)
            
        B, H, W = mask.shape

        # ==========================================
        # 2. 核心处理流程 (Fill Hole -> Expand -> Blur -> Invert)
        # ==========================================
        # 2.1 填充孔洞 (双引擎：cv2 优先，scipy 兜底，需要 4D 输入)
        mask_4d = mask.unsqueeze(1) # [B, 1, H, W]
        mask_4d = mask_fill_hole(mask_4d)

        # 2.2 膨胀/腐蚀 (极速可分离形态学)
        mask_4d = mask_morphology_fast(mask_4d, expand_pixels=mask_expand)

        # 2.3 边缘模糊 (极速可分离高斯模糊)
        if mask_blur > 0:
            k_size = mask_blur * 2 + 1
            mask_4d = gaussian_blur_2d_fast(mask_4d, kernel_size=k_size, sigma=mask_blur / 3.0)

        # 2.4 反转
        if invert_mask:
            mask_4d = 1.0 - mask_4d

        # ==========================================
        # 3. 强制规范化输出为严格的 3D [B, H, W] 并限制值域
        # ==========================================
        final_mask = torch.clamp(mask_4d.squeeze(1), 0.0, 1.0)
        
        # 兜底维度检查，确保绝对不会输出 4D 导致预览节点崩溃
        if final_mask.dim() == 4:
            final_mask = final_mask.squeeze(1)
        if final_mask.dim() == 2:
            final_mask = final_mask.unsqueeze(0)

        return (final_mask,)

# ==========================================
# 需求 5：Image 与 Mask 混合预览节点 (含完整 Mask 后处理与最终 Mask 输出)
# ==========================================
class Image_Mask_Preview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "原始底图"}),
                "mask": ("MASK", {"tooltip": "用于混合的 Mask (若尺寸不一致会自动双线性缩放对齐)"}),
                
                # 【严格按照指定顺序排列的 Mask 后处理链】
                "fill_holes": ("BOOLEAN", {"default": False, "tooltip": "填充 Mask 内部不与边界相连的闭合孔洞，使其变为实心"}),
                "mask_expand": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1, "tooltip": "对 Mask 进行膨胀(正数)或腐蚀(负数)，改变作用范围"}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "对 Mask 边缘进行高斯模糊的像素半径，生成平滑羽化过渡"}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "在所有形态学处理后，反转 Mask (1.0 - mask)"}),
                
                "mask_preview_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "hard_light", 
                                "linear_dodge", "linear_burn", "color_dodge", "color_burn", 
                                "vivid_light", "linear_light", "pin_light", "hard_mix",
                                "difference", "subtract", "darker", "lighter", 
                                "hue", "color", "luminosity"], 
                               {"default": "normal", "tooltip": "类似 PS 的图层混合模式"}),
                "mask_color": ("STRING", {"default": "#00ff00", "multiline": False, "tooltip": "混合颜色，支持 #00ff00, 0,1,0, green 等"}),
                "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "混合强度 (0.0=全底图, 1.0=全混合色)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("final_image", "final_mask")
    FUNCTION = "execute"
    CATEGORY = "afar_tools"

    def execute(self, image, mask, fill_holes, mask_expand, mask_blur, invert_mask, mask_preview_mode, mask_color, blend_strength):
        B, H, W, C = image.shape
        device = image.device
        dtype = image.dtype

        # ==========================================
        # 1. 强制规范化 mask 维度并对齐 Batch
        # ==========================================
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)
            
        if mask.shape[0] != B:
            mask = mask.repeat(B // mask.shape[0] + 1, 1, 1)[:B] if mask.shape[0] < B else mask[:B]

        # ==========================================
        # 2. 空间尺寸对齐 (若 mask 与 image 宽高不同)
        # ==========================================
        if mask.shape[1] != H or mask.shape[2] != W:
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)

        # ==========================================
        # 3. 核心 Mask 后处理流水线 (严格保持 4D [B, 1, H, W] 以提升计算效率)
        # ==========================================
        mask_4d = mask.unsqueeze(1)
        
        # 3.1 填充孔洞
        if fill_holes:
            mask_4d = mask_fill_hole(mask_4d)
            
        # 3.2 膨胀 / 腐蚀
        mask_4d = mask_morphology_fast(mask_4d, expand_pixels=mask_expand)
        
        # 3.3 边缘模糊
        if mask_blur > 0:
            k_size = mask_blur * 2 + 1
            mask_4d = gaussian_blur_2d_fast(mask_4d, kernel_size=k_size, sigma=mask_blur / 3.0)
            
        # 3.4 反转
        if invert_mask:
            mask_4d = 1.0 - mask_4d

        # 恢复为 3D [B, H, W] 并钳制值域
        final_mask = torch.clamp(mask_4d.squeeze(1), 0.0, 1.0)

        # ==========================================
        # 4. 颜色解析与混合渲染
        # ==========================================
        r, g, b = parse_color_string(mask_color)
        blend_color_t = torch.tensor([r, g, b], device=device, dtype=dtype).view(1, 1, 1, 3)
        
        mask_expanded = final_mask.unsqueeze(-1) # [B, H, W, 1]
        
        final_image = apply_blend_mode(
            base=image, 
            blend_color=blend_color_t, 
            mode=mask_preview_mode, 
            strength=blend_strength, 
            mask_expanded=mask_expanded
        )

        # ==========================================
        # 5. 最终输出前的维度兜底检查 (确保 100% 兼容 ComfyUI 预览节点)
        # ==========================================
        if final_mask.dim() == 4:
            final_mask = final_mask.squeeze(1)
        elif final_mask.dim() == 2:
            final_mask = final_mask.unsqueeze(0)
        if final_mask.shape[0] != B:
            final_mask = final_mask.repeat(B, 1, 1)[:B] if final_mask.shape[0] < B else final_mask[:B]

        return (final_image, final_mask)