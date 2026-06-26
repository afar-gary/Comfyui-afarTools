

import nodes,comfy,torch    # type:ignore
from comfy_api.latest import ComfyExtension, IO, UI  # type:ignore


class ResizeAndPadImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ResizeAndPadImage",
            search_aliases=["fit to size"],
            display_name="Resize And Pad Image",
            category="image/transform",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("target_width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("target_height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Combo.Input("padding_color", options=["white", "black"], advanced=True),
                IO.Combo.Input("interpolation", options=["area", "bicubic", "nearest-exact", "bilinear", "lanczos"], advanced=True),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, target_width, target_height, padding_color, interpolation,crop='disabled') -> IO.NodeOutput:
        batch_size, orig_height, orig_width, channels = image.shape

        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        scale = min(scale_w, scale_h)

        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        image_permuted = image.permute(0, 3, 1, 2)
        resized = comfy.utils.common_upscale(image_permuted, new_width, new_height, interpolation, crop)

        # ================= 核心修改区域 =================
        
        # 1. 解析自定义颜色 (返回 0.0-1.0 的 r, g, b)
        r, g, b = cls.parse_color(padding_color)
        
        # 2. 根据图像通道数构建颜色张量
        if channels == 4:
            # 如果是 RGBA，背景 Alpha 通道默认设为 1.0 (不透明)
            color_vals = [r, g, b, 1.0]
        else:
            color_vals = [r, g, b]
            
        color_tensor = torch.tensor(color_vals, dtype=image.dtype, device=image.device)
        
        # 3. 创建背景画布 (使用 zeros 代替 full)
        padded = torch.zeros(
            (batch_size, channels, target_height, target_width),
            dtype=image.dtype,
            device=image.device
        )
        
        # 4. 利用广播机制将颜色填充到整个画布
        # color_tensor.view(1, C, 1, 1) 会将其形状变为 (1, 通道数, 1, 1)
        # padded[:] = ... 会自动将其广播到 (B, C, H, W) 并赋值
        padded[:] = color_tensor.view(1, channels, 1, 1)
        
        # ================================================

        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        # 将缩放后的图像粘贴到画布中心
        padded[:, :, y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        output = padded.permute(0, 2, 3, 1)
        return IO.NodeOutput(output)

    @classmethod
    def empty_image(cls,width, height,color='#000000'):
        r, g, b = cls.parse_color(color)
        dtype = comfy.model_management.intermediate_dtype()
        device = comfy.model_management.intermediate_device()
        r = torch.full([1, height, width, 1], r, device=device, dtype=dtype)
        g = torch.full([1, height, width, 1], g, device=device, dtype=dtype)
        b = torch.full([1, height, width, 1], b, device=device, dtype=dtype)
        image = torch.cat((r, g, b), dim=-1)
        return (image, )


    @classmethod
    def parse_color(cls, color_str: str):
        """解析颜色字符串，支持 'black', 'white' 以及 Hex (如 '#00FF00')"""
        color_str = str(color_str).strip().lower()
        
        # 预设颜色
        if color_str in ['black','bk']: return 0.0, 0.0, 0.0
        if color_str in ['white','w']: return 1.0, 1.0, 1.0
        if color_str in ['gray','grey','gr']: return 0.5, 0.5, 0.5
        if color_str in ['red','r']: return 1.0, 0.0, 0.0
        if color_str in ['green','g']: return 0.0, 1.0, 0.0
        if color_str in ['blue','b']: return 0.0, 0.0, 1.0
        if color_str in ['yellow','y']: return 1.0, 1.0, 0.0
        if color_str in ['magenta','m']: return 1.0, 0.0, 1.0
        
        # 映射 0-1 至255 解析十进制 颜色
        if ',' in color_str:
            color_list = [p.strip() for p in color_str.split(',')]
            try:
                r = float(color_list[0])
                g = float(color_list[1])
                b = float(color_list[2])
                r = max(0.0, min(1.0, r))
                g = max(0.0, min(1.0, g))
                b = max(0.0, min(1.0, b))
                return r,g,b
            except ValueError:
                pass
        # 解析 Hex 颜色
        color_str16 = color_str.lstrip('#')
        if len(color_str16) == 6:
            try:
                r = int(color_str16[0:2], 16) / 255.0
                g = int(color_str16[2:4], 16) / 255.0
                b = int(color_str16[4:6], 16) / 255.0
                return r, g, b
            except ValueError:
                pass

                
        # 解析失败默认返回黑色
        return 0.0, 0.0, 0.0




    resize_and_pad = execute  # TODO: remove