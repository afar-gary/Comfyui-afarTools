from .utils import image_tools,ksamplers

# from ay import ay
from .utils.utils import aftools
from .utils.ksamplers import pipe_unite_ksampler
import folder_paths # type: ignore

from server import PromptServer # type: ignore
from aiohttp import web # type: ignore
import server # type: ignore
import os
import platform,subprocess

@server.PromptServer.instance.routes.get("/aftools/update_params")
@server.PromptServer.instance.routes.post("/aftools/update_params")
async def update_params(request):
     # ================== 处理 POST 保存请求 ==================
    if request.method == 'POST':
        action = request.rel_url.query.get("action", None)
        if action == 'save':
            try:
                print(f'✅ 触发保存')
                data = await request.json()
                data_dict = data.get('variables', {})
                clip_type = data_dict.get('clip_type')
                print(f' -- clip_type:{clip_type}')
                for k,v in data_dict.items():
                    print(f' -- data:{k} -> {v}')

                # 【核心修复】清洗数据：把前端传来的字符串 "None"、"null" 统一转为 Python 的 None
                cleaned_variables = {}
                for key, value in data_dict.items():
                    if value is None or value == "None" or value == "null":
                        cleaned_variables[key] = None
                    else:
                        cleaned_variables[key] = value

                print(f"✅ 后端接收到保存请求 - clip_type: {clip_type}")
                # print(f"✅ 后端接收到清洗后的变量: {cleaned_variables}")        



                aftools().parameter_save(cleaned_variables)
                new_params = {"status": "success", "message": "保存成功"}
                return web.json_response(new_params)
            except Exception as e:
                print(f"❌ 后端保存出错: {e}")
                return web.json_response({"status": "error", "message": str(e)}, status=500)
    # ================== 处理 GET 加载预设请求 ==================
    elif request.method == 'GET':
        preset = request.rel_url.query.get("preset",None)
        if not preset or preset == "undefined":
            print("⚠️ 后端拦截：前端传来的预设名为空或 undefined")
            return web.json_response({"error": "预设名为空"}, status=400)
        try:      
            new_params = {}  
            # print(f"✅ /aftools/update_params 当前预设: {preset}")
            # print(f"✅ 后端路由 /aftools/update_params 已被访问！当前预设: {preset}")

            # 获取当前pipe_unite_ksampler_dev类的default值
            vars_default_dict = aftools().parameter_set_default(pipe_unite_ksampler.INPUT_TYPES())
            
            if preset == 'not found':
                print(f'❌ 预设文件不存在')
                new_params = {}
                # return web.json_response(new_params)
            elif preset == 'default':
                new_params = vars_default_dict
                # return web.json_response(new_params)
            else:        
                new_params = aftools().parameter_set(preset,vars_default_dict)
                # print(f"✅ 后端成功获取参数: {new_params.keys()}")

            if new_params:
                return web.json_response(new_params)
            else:
                return web.json_response({})

            
        except Exception as e:
            print(f"❌ 后端加载预设出错: {e}")
            return web.json_response({"error": str(e)}, status=500)
       
    # 如果都不是，返回兜底响应
    return web.json_response({"error": f"未知的请求方式: {request.method}"}, status=400)

@server.PromptServer.instance.routes.get("/aftools/open_folder")
async def open_folder_handler(request):
    folder_path = request.rel_url.query.get("path", None)
    
    if not folder_path:
        return web.json_response({"error": "未提供路径"}, status=400)

    # 【核心修复】将前端传来的相对路径，拼接成 ComfyUI 根目录下的绝对路径
    # folder_paths.base_path 就是 ComfyUI 的根目录
    absolute_path = os.path.join(folder_paths.base_path, folder_path)
    
    # 为了保险起见，把路径中的斜杠统一规范化（兼容 Windows 和 Linux/macOS）
    absolute_path = os.path.normpath(absolute_path)

    print(f"📂 尝试打开文件夹绝对路径: {absolute_path}") # 可以在后台黑框里看到实际去打开的路径

    if not os.path.exists(absolute_path):
        return web.json_response({"error": f"路径不存在: {absolute_path}"}, status=400)

    try:
        # 根据不同的操作系统调用不同的命令打开文件夹
        if platform.system() == "Windows":
            os.startfile(absolute_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", absolute_path])
        else:  # Linux
            subprocess.Popen(["xdg-open", absolute_path])
            
        return web.json_response({"status": "success", "message": "已尝试打开文件夹"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


NODE_CLASS_MAPPINGS = {    
    'ImageMask_Pad_Resize':image_tools.ImageMask_Pad_Resize,    
    'Fill_Mask_Holes':image_tools.Fill_Mask_Holes,    
    'Image_Mask_Preview':image_tools.Image_Mask_Preview,    
    'CropByMask_Resize':image_tools.CropByMask_Resize,    
    'CropByMask_Resize_sam3':image_tools.CropByMask_Resize_sam3,    
    'CropByMask_Restore':image_tools.CropByMask_Restore,    
    'pipe_loras_pack':ksamplers.pipe_loras_pack,
    'pipe_unite_loader':ksamplers.pipe_unite_loader,
    'pipe_unite_ksampler':ksamplers.pipe_unite_ksampler,  
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'ImageMask_Pad_Resize':'afar ImageMask_Pad_Resize v1.7',
    'Fill_Mask_Holes':'afar Fill_Mask_Holes v1.1',
    'Image_Mask_Preview':'afar Image_Mask_Preview v1.1',
    'CropByMask_Resize':'afar CropByMask_Resize v2.0',
    'CropByMask_Resize_sam3':'afar CropByMask_Resize_sam3 v1.3',
    'CropByMask_Restore':'afar CropByMask_Restore v1.6',
    'pipe_loras_pack':'afar Until_Loras_Stack v1.5',
    'pipe_unite_loader':'afar Unite_Loader v3.3',
    'pipe_unite_ksampler':'afar Unite_Ksampler v3.3',
    
}

WEB_DIRECTORY = './web'

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']




