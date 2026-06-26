import os,sys,importlib.util,json,math,textwrap,copy,gc
import folder_paths,nodes,torch
import comfy,comfy.sd,comfy_api
import comfy.utils
import logging
# import comfy.model_management
import comfy_api.latest._io as io
# from comfy.comfy_types import IO
import comfy_extras.nodes_flux as nodes_flux
import comfy_extras.nodes_cond as nodes_cond
import comfy_extras.nodes_chroma_radiance as nodes_chroma_radiance
import comfy_extras.nodes_model_patch as nodes_model_patch
import comfy_extras.nodes_edit_model as nodes_edit_model
import comfy_extras.nodes_model_advanced as nodes_model_advanced
import comfy_extras.nodes_cfg as nodes_cfg
import comfy_extras.nodes_custom_sampler as nodes_custom_sampler
import comfy_extras.nodes_post_processing as nodes_post_processing
import comfy_extras.nodes_mask as nodes_mask
import comfy_extras.nodes_differential_diffusion as nodes_differential_diffusion
# import comfy_extras.nodes_images as nodes_images

from ..utils import node_qwen_diy as diy
from ..utils import node_pad as pad
from ..utils.node_qwen_diy import QwenEditTextEncode_EditUtils as qwenEU
from ..utils.node_fls import FLSSamplerNodeV4 as fls
from ..utils.node_krea2_diy import ConditioningKrea2Rebalance as Krea2Rebalance




_gguf_exist = False
gg = None
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    custom_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    gguf_dir = os.path.join(custom_dir, 'gguf')
    file_path = os.path.join(gguf_dir, 'pig.py')
    file_exists = os.path.isfile(file_path)
    
    # print(f'--current_dir: {current_dir}')
    # print(f'--gguf_dir: {gguf_dir}')
    # print(f'--file_exists: {file_exists}')
    
    if file_exists:
        # ==========================================
        # 【核心修复】：临时劫持 sys.path 和 sys.modules
        # 完美解决相对导入报错，且不破坏环境中的官方 gguf 库
        # ==========================================
        
        # 1. 备份现场
        original_sys_path = sys.path.copy()
        # 备份所有以 gguf 开头的模块（防止官方库被覆盖）
        gguf_modules_backup = {k: v for k, v in sys.modules.items() if k == 'gguf' or k.startswith('gguf.')}
        
        # 2. 清理缓存：防止 Python 使用已经加载的官方 gguf 缓存
        for k in list(sys.modules.keys()):
            if k == 'gguf' or k.startswith('gguf.'):
                del sys.modules[k]
                
        # 3. 提升优先级：将 custom_dir 插入到 sys.path 最前面
        if custom_dir in sys.path:
            sys.path.remove(custom_dir)
        sys.path.insert(0, custom_dir)
        
        try:
            # 4. 执行标准导入（此时 Python 会完美处理 pig.py 内的所有相对/绝对导入）
            from gguf import pig as gg  # type:ignore
            _gguf_exist = True
            print("✅ 成功加载本地 gguf/pig.py")
        finally:
            # 5. 【关键】无论成功失败，必须恢复现场！绝不影响其他依赖官方 gguf 的插件
            sys.path = original_sys_path
            
            # 清理掉刚导入的本地 gguf 模块缓存
            for k in list(sys.modules.keys()):
                if k == 'gguf' or k.startswith('gguf.'):
                    del sys.modules[k]
            
            # 把官方的 gguf 模块还回去
            sys.modules.update(gguf_modules_backup)            
    else:
        _gguf_exist = False
        print("⚠️ 未找到 pig.py 文件")
except Exception as e:
    _gguf_exist = False
    print(f"❌ 加载 gguf/pig.py 失败: {e}")
    print(f"无法支持 gguf 模型，屏蔽并切回 safetensors 模式。")
    print(f"需手动刷新界面。已加载 gguf 模型会报红，删除节点重新添加即可。")


# items set
PIPE_TYPE_MODEL = "models"  # 全局唯一类型标识
PIPE_TYPE_LORA = "loras"  # 全局唯一类型标识
SIGNATURE = "_aftools_"   # 运行时签名标记
SOLUTJION_TYPE=['totalPixels','longerEdge','preset','custom']
ASPECTRATIO = ['1:1','1:2','2:3','3:4','5:7','9:16','9:21','10:21']
ROUND_MUTILPLE = ['None',8,16,32,64,128,512]
CN_TYPE = ['normal', 'zimage']
KSAMPLE_TYPE = ['normal','normal double','advanced single','advanced double[base/turbo]','custom']
UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
# 统一精度控制入口（推荐 UI 下拉选项）
DTYPE_UNIFIED = ["default", "float32", "float16", "bfloat16", "fp8_e4m3fn_fast", "fp8_e5m2"]
ROUND16_LIST=['flux','flux2']
ROUND8_LIST=['qwen_image','omnigen2','lumina2']


class aftools:   
    
    # IO
    def get_model_diffusion_list(s):
        # if model_type == 'Checkpoints':
        #     print(f'----Checkpoints')
        # else:
        #     print(f'----diffusion')
        m1 = folder_paths.get_filename_list('checkpoints') if 'checkpoints' in folder_paths.folder_names_and_paths else []
        m2 = folder_paths.get_filename_list('diffusion_models') if 'diffusion_models' in folder_paths.folder_names_and_paths else []
        m1_clear = [str(name).strip() for name in m1] if m1 else []
        m2_clear = [str(name).strip() for name in m2] if m2 else []
        if _gguf_exist:
            g1 = folder_paths.get_filename_list('model_gguf')
            g1_clear = [str(name).strip() for name in g1] if g1 else []
            return ['None'] + m1_clear + m2_clear + g1_clear
        else:
            return ['None'] + m1_clear + m2_clear
            
    def get_model_clip_list(s):
        c2 = folder_paths.get_filename_list('text_encoders') if 'text_encoders' in folder_paths.folder_names_and_paths else []
        c2_clear = [str(name).strip() for name in c2] if c2 else []
        if _gguf_exist:
            c1 = folder_paths.get_filename_list('clip') if 'clip' in folder_paths.folder_names_and_paths else []
            g1 = folder_paths.get_filename_list('clip_gguf') 
            c1_clear = [str(name).strip() for name in c1] if c1 else []
            g1_clear = [str(name).strip() for name in g1] if g1 else []

            combined_clip_list = c1_clear + c2_clear + g1_clear
        else:
            combined_clip_list = c2_clear
        return ['None'] + combined_clip_list[:]
    
    def get_model_vae_list(s):        
        if _gguf_exist:
            vaes = []
            vaes += folder_paths.get_filename_list('vae')
            vaes += folder_paths.get_filename_list('vae_gguf')
            approx_vaes = folder_paths.get_filename_list('vae_approx')
            sdxl_taesd_enc = False
            sdxl_taesd_dec = False
            sd1_taesd_enc = False
            sd1_taesd_dec = False
            sd3_taesd_enc = False
            sd3_taesd_dec = False
            f1_taesd_enc = False
            f1_taesd_dec = False
            for v in approx_vaes:
                if v.startswith('taesd_decoder.'):
                    sd1_taesd_dec = True
                elif v.startswith('taesd_encoder.'):
                    sd1_taesd_enc = True
                elif v.startswith('taesdxl_decoder.'):
                    sdxl_taesd_dec = True
                elif v.startswith('taesdxl_encoder.'):
                    sdxl_taesd_enc = True
                elif v.startswith('taesd3_decoder.'):
                    sd3_taesd_dec = True
                elif v.startswith('taesd3_encoder.'):
                    sd3_taesd_enc = True
                elif v.startswith('taef1_encoder.'):
                    f1_taesd_dec = True
                elif v.startswith('taef1_decoder.'):
                    f1_taesd_enc = True
            if sd1_taesd_dec and sd1_taesd_enc:
                vaes.append('taesd')
            if sdxl_taesd_dec and sdxl_taesd_enc:
                vaes.append('taesdxl')
            if sd3_taesd_dec and sd3_taesd_enc:
                vaes.append('taesd3')
            if f1_taesd_dec and f1_taesd_enc:
                vaes.append('taef1')
        else:
            vaes = folder_paths.get_filename_list('vae') if 'vae' in folder_paths.folder_names_and_paths else []
        vaes_clear = [str(name).strip() for name in vaes] if vaes else []
        vaes_clear.append("pixel_space")
        return ['None'] + vaes_clear
    
    def get_model_lora_list(s):
        l1 = folder_paths.get_filename_list('loras') if 'loras' in folder_paths.folder_names_and_paths else []
        l2 = folder_paths.get_filename_list('lora') if 'lora' in folder_paths.folder_names_and_paths else []
        l1_clear = [str(name).strip() for name in l1] if l1 else []
        l2_clear = [str(name).strip() for name in l2] if l2 else []
        return ['None'] + l1_clear + l2_clear
    
    def get_single_clip_types(s):
        # single_clip_types = ['stable_diffusion', 'stable_cascade', 'sd3', 'stable_audio', 'mochi', 'ltxv', 'pixart', 'cosmos', 'lumina2', 'wan', 'hidream', 'chroma', 'ace', 'omnigen2', 'qwen_image', 'hunyuan_image', 'flux2', 'ovis', 'longcat_image', 'cogvideox']
        single_clip_types, = nodes.CLIPLoader.INPUT_TYPES()['required']['type']
        # print(f'{clip_type(single_clip_types)} -> {list(single_clip_types)}')
        return single_clip_types
    def get_double_clip_types(s):
        # double_clip_types = ['sdxl', 'sd3', 'flux', 'hunyuan_video', 'hidream', 'hunyuan_image', 'hunyuan_video_15', 'kandinsky5', 'kandinsky5_image', 'ltxv', 'newbie', 'ace'] 
        double_clip_types, = nodes.DualCLIPLoader.INPUT_TYPES()['required']['type']
        return double_clip_types
    def get_clip_type_list(s):
        # show_t = set(s.get_single_clip_types() + s.get_double_clip_types())
        show_t = ['======SINGLE======'] + s.get_single_clip_types() + ['======DOUBLE======'] + s.get_double_clip_types()
        # print(f'{clip_type(show_t)} -> {show_t}')
        return show_t

    def get_resolution_preset(s):
        resolution_dict ={}
        try:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
            # parmas_dir = os.path.join(root,'presets')
            file_name = 'presets_resolution.txt'
            # file_path = os.path.join(parmas_dir,file_name)
            file_path = os.path.join(root,file_name)
            if os.path.exists(file_path):
                with open(file_path,'r',encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or ":" not in line:
                            continue
                        # 拆分比例 + 分辨率
                        ratio, res_line = line.rsplit(":", 1)
                        ratio = ratio.strip().replace("'",'')

                        res_line = res_line.rsplit(']')[0].replace('[','').replace("'",'').split(',')
                        resolution_dict[ratio] = res_line
                    # print(f'run test -> {file_path}')
                    # print(resolution_dict)
            else:
                resolution_dict ={
                    '1:1':['1024x1024','1600x1600'],
                    '3:2':['768x512','1256x840',],
                    '4:3':['640x480','960x720','1472x1104'],
                    '16:9':['832x480','1280x720','1366x768','1664x928'],
                    '21:9':['1568x672'],
                }
        except:
            # path_s = f'{root.rsplit('\\',2)[1]}\{root.rsplit('\\',2)[2]}\\'
            # print(f"warning: 根目录下未找到presets.txt -> {path_s}")
            pass

        list_s = []
        for k,v in resolution_dict.items():
            for x in v:
                list_s.append(f'[{k}] {x}')
        # print(f'read resolution -> {list_s}')
        return list_s

    def common_upscale(s,image, width, height, upscale_method='bicubic', crop='center'):
        image_permute = image.permute(0, 3, 1, 2)
        image2 = comfy.utils.common_upscale(image_permute,width, height,upscale_method, crop)
        image2 = image2.permute(0, 2, 3, 1)
        return image2

    def get_resolution(s,match_size,image1,image2,mask,resolution_type,flip,
                           ratio='1:1',megapixels=1.0,
                           longerEdge = 1024,resolution_preset='(1:1) 512x512',width=1024,height=1024,interpolation='area',roundstep=2,color= '#00ff00',crop='disable',clip_type=''):
        
        def _totalPixels(ratio='1:1',megapixels=1,step=8):
            ratios =ratio.split(':')
            w_ratio, h_ratio = int(ratios[0]),int(ratios[1])
            longest_edge = 2048
            total_pixels = megapixels * 1024 * 1024
            if total_pixels > longest_edge**2:
                total_pixels = longest_edge**2
            scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
            width_r = round(w_ratio * scale / step) * step
            height_r = round(h_ratio * scale / step) * step
            return (width_r,height_r)

        def _longerEdge(ratio='1:1',longerEdge=1024,step=8):
            ratios =ratio.split(':')
            w_ratio, h_ratio = int(ratios[0]),int(ratios[1])
            longest,shortest = max(w_ratio,h_ratio),min(w_ratio,h_ratio)

            if w_ratio > h_ratio:
                width_r = longerEdge//step*step
                height_r = int(longerEdge/longest*shortest//step*step)
            elif w_ratio == h_ratio:
                width_r = longerEdge//step*step
                height_r = longerEdge//step*step
            else:
                width_r =  int(longerEdge/longest*shortest//step*step)
                height_r = longerEdge//step*step
            return (width_r,height_r)
        

        if clip_type in ROUND16_LIST:
            step = 16
        elif clip_type in ROUND8_LIST:
            step = 8
        else:
            step = 2 if roundstep == 'None' else int(roundstep)

        image_f,mask_f = None,None
        image_input = None
        width_r,height_r=0,0
        if image1 is not None:
            image_input = image1
        elif image1 is None and image2 is not None:
            image_input = image2
            
        if image_input is not None:
            image1_w,image1_h = image_input.shape[2],image_input.shape[1]
            divisor = math.gcd(image1_w,image1_h)
            image1_ratio_str = f'{round(image1_w/divisor)}:{round(image1_h/divisor)}'
            # print(f'--image1_ratio_str:{image1_ratio_str}')
            if match_size:
                width_r,height_r = image1_w,image1_h
            else:
                if resolution_type == 'totalPixels':                
                    width_r,height_r = _totalPixels(image1_ratio_str,megapixels,step)
                elif resolution_type == 'longerEdge':
                    width_r,height_r = _longerEdge(image1_ratio_str,longerEdge,step)
                elif resolution_type == 'preset':
                    p = resolution_preset.split(' ')[1].split('x')
                    width_r = (int(p[0])//step)*step
                    height_r = (int(p[1])//step)*step    
                elif resolution_type == 'custom':
                    width_r,height_r = (width//step)*step,(height//step)*step
            if flip:
                temp = height_r
                height_r = width_r
                width_r = temp
            image_input_resize = pad.ResizeAndPadImage.execute(image_input,width_r, height_r,color, interpolation,crop)
            image_f = image_input_resize[0]
        else:
            if resolution_type =='totalPixels':
                width_r,height_r = _totalPixels(ratio,megapixels,step)
            elif resolution_type =='longerEdge':
                width_r,height_r = _longerEdge(ratio,longerEdge,step)
            elif resolution_type == 'preset':
                p = resolution_preset.split(' ')[1].split('x')
                width_r = (int(p[0])//step)*step
                height_r = (int(p[1])//step)*step 
            elif resolution_type =='custom':
                # width_r,height_r = width,height
                width_r,height_r = (width//step)*step,(height//step)*step
            if flip:
                temp = height_r
                height_r = width_r
                width_r = temp
            # image_f = pad.ResizeAndPadImage.empty_image(width_r,height_r,color)[0]


        if mask is not None:
            # print(f'-- mask.shape:{mask.shape}')
            if image_f is not None:
                B,H,W,C = image_f.shape
                device_f = image_f.device
                dtype_f = image_f.dtype
            else:
                B,H,W = (1,height_r,width_r)
                device_f = mask.device
                dtype_f = mask.dtype

            mask_4D = mask.to(device=device_f, dtype=dtype_f)
            if mask.dim()==3:
                mask_4D = mask.unsqueeze(1)
            elif mask.dim()==2:
                mask_4D = mask.unsqueeze(0).unsqueeze(0)
            if mask.shape[0]!= B:
                mask_f = mask.repeat(B // mask.shape[0] + 1, 1, 1)[:B] if mask.shape[0] < B else mask[:B]

            # print(f'-- mask_f.shape:{mask_4D.shape}')
            mask_4D_scale = comfy.utils.common_upscale(mask_4D, W, H, "bilinear", "disabled").permute(0, 2, 3, 1)
            # print(f'-- mask_4D_scale.shape:{mask_4D_scale.shape}')

            mask_f = mask_4D_scale.squeeze(-1)
            # print(f'-- mask_f.shape:{mask_f.shape}')

        return (image_f,mask_f,width_r, height_r)


    def get_samplers_type(s):
        sm = comfy.samplers.KSampler.SAMPLERS
        return sm
    
    def get_schedulers_type(s):
        sh = comfy.samplers.KSampler.SCHEDULERS
        return sh

    # 匹配 safetensors/gguf weigth type 
    def _get_loader_kwargs(unified_dtype: str, is_gguf: bool):
        """一行映射，直接返回对应加载器需要的参数字典"""
        if is_gguf:
            # GGUF 的 patch_dtype 不支持 FP8，选 FP8 时自动降级到 bfloat16
            dtype = "bfloat16" if "fp8" in unified_dtype else unified_dtype
            return {"patch_dtype": dtype}
        else:
            # Safetensors 直接透传，fp8_e4m3fn 会自动触发 fp8_optimizations
            return {"weight_dtype": unified_dtype}

    def _unpack_tuple(s,obj):
        obj_ex = None
        comfy_type_lsit = [list,dict,
                           comfy.sd.CLIP,comfy.sd.VAE,
                           comfy.model_patcher.ModelPatcherDynamic,
                           comfy_api.latest._io.NodeOutput
                           ]
        if type(obj) == tuple and len(obj) >=1:
            len_obj = len(obj)
            for i in range(len_obj):
                # print(f' -- tuple read: {i} {type(obj[i])}')
                if type(obj[i]) in comfy_type_lsit:
                    obj_ex = obj[i]
                    # print(f' -- tuple read: {obj if type(obj[i]) not in [list,dict] else 'mabey latent/clip.encode list [no show and skip]'}')
                    # print(f' -- tuple extract: {obj_ex if type(obj_ex) not in [list,dict] else 'mabey latent/clip.encode list [no show and skip]'}')
                    break
            return obj_ex    
        else:
            return obj

    def _check_model_layer(s,model_load):
        model_type = ''
        model_layer = []
        try:
            if type(model_load) != comfy.model_patcher.ModelPatcherDynamic:
                if len(model_load) > 1:
                    model_type = 'model_full'
                    for i in range(len(model_load)):
                        # print (f' -- model_layer {i} -> {type(model_load[i])}')
                        model_layer.append(1 if model_load[i] else 0)
                else:
                    model_type = 'model_single_gguf'
                    # print (f' -- model_layer 0 -> {type(model_load[0])}')
                    model_layer = [1,0,0]
            else:
                model_type = 'model_single'
                model_layer = [1,0,0]
            # print(f' -- model load type = [{model_type}]')
        except ValueError as e:
            print(f' -- mdoel wrong,model_layer < 0 ,{e}')
            return
        # print(f' --model_layer:{model_layer}')
        return model_layer





    # =========================================================
    # get class var
    def mode_info(s,value):
        comfy_root = f'{os.path.abspath(__file__).split('custom_nodes')[0]}'
        model_path = os.path.join(comfy_root,'models')
        # model path
        model_info = {}
        dir = []
        for d in os.listdir(model_path):
            if os.path.isdir(os.path.join(model_path,d)):
                dir.append(d)
                # print(dir)            
        target_dir = ''
        for i in range(len(dir)):
            target_dir = os.path.join(model_path,dir[i])
            target_file =f'{target_dir}\\{value}'    
            if os.path.isfile(target_file) and os.path.exists(target_file):
                path = target_file
                # model_info['var'] = scr_key
                # model_info['name'] = target_file.split('\\')[-1].split('.')[0]
                model_info['dir'] = dir[i]
                model_info['format'] = path.split('.')[-1]
                model_info['input'] = value
                model_info['path'] = path
                break

        # file = str(value).rsplit('\\')[-1]
        # print(f'{'-'*20}\ninput:[{scr_key}] - [{file}]\n{'-'*20}')
        # for k,v in model_info.items():
        #     print(f'{' '}{k}:{v}')
        return model_info

    def get_cls_slots_vars2(s,INPUT_TYPES,
                            model_1,model_2='None',clip_1='None',clip_2='None',vae='None',
                           lora_1='None',lora_2='None',control_model='None',):
        
        vars_main = {'model_1':model_1,'model_2':model_2,'clip_1':clip_1,'clip_2':clip_2,'vae':vae,
                     'lora_1':lora_1,'lora_2':lora_2,'control_model':control_model,
                     }
        required_dict = {}
        optional_dict = {}
        input_type_dict = {}
        for k,v in INPUT_TYPES.get('required').items():
            required_dict[k] = v[1].get('default') if isinstance(v,tuple) and len(v) >1 else None
        for k,v in INPUT_TYPES.get('optional').items():
            optional_dict[k] = v[1].get('default') if isinstance(v,tuple) and len(v) >1 else None        
        input_type_dict.update(required_dict)
        input_type_dict.update(optional_dict)

        model_info={}
        set_info={}
        for k,v in input_type_dict.items():
            if type(v) == str:
                if k in list(vars_main.keys()) and vars_main[k] != 'None':
                    if k == 'vae' and vars_main[k] == 'pixel_space':
                        model_info[k] = {"dir": '',"format": '',"input": 'pixel_space',"path":''}
                    else:
                        model_info[k] = s.mode_info(vars_main[k]) if vars_main[k] != 'None' else '-1'
                else:
                    if k == 'skip_ksample':
                        continue
                    set_info[k] = v
            else:
                if type(v) in [int,float,bool]:
                    set_info[k] = v

        datatree = {
            SIGNATURE:True,
            'models':model_info,
            'set':set_info,
        }
        return (datatree,)


    def get_cls_slots_vars(s,INPUT_TYPES,
                           model_1='None',model_2='None',clip_1='None',clip_2='None',vae='None',
                           lora_1='None',lora_2='None',control_model='None',
                           **kwargs):
        data_keys =  INPUT_TYPES
        # print(f'func input_data:{len(data_keys)},{type(data_keys)},{data_keys.keys()}')
        vars_main = {'model_1':model_1,'model_2':model_2,'clip_1':clip_1,'clip_2':clip_2,'vae':vae,
                     'lora_1':lora_1,'lora_2':lora_2,'control_model':control_model,
                     }
        input_vars = []
        for i in data_keys:
            if len(data_keys[i])>0:
                for v in data_keys[i].keys():
                    if v not in list(vars_main.keys()):
                        input_vars.append(v)
        vars_dict = {}

        if len(input_vars)>0:
            for k,v in vars_main.items():
                vars_dict[k] = v if v !='None' else 'None'
            # print(f'input_vars:{input_vars}')

            pass
        else:
            return None

        for i in input_vars:
            vars_dict[i]=kwargs.get(i)
        return vars_dict

    # get value
    def get_input_slots(s,input_dict):
        comfy_root = f'{os.path.abspath(__file__).split('custom_nodes')[0]}'
        model_path = os.path.join(comfy_root,'models')
        # print(model_path)
        scr_key = list(input_dict.keys())[0]
        scr_value = list(input_dict.values())[0] 

        # is_path,is_num = False,False
        # var_type = ''
        # if scr_value not in ['','None']:
        #     if type(scr_value) == str:
        #         var_type = 'path' if len(scr_value.split('\\'))>1 or scr_value.rsplit('.')[-1] in ['safetensors','pt','gguf'] else None
        #     else:
        #         var_type = 'num' 
    
        def mode_info(value):
            # model path
            model_info = {}
            dir = []
            for d in os.listdir(model_path):
                if os.path.isdir(os.path.join(model_path,d)):
                    dir.append(d)
                    # print(dir)            
            target_dir = ''
            for i in range(len(dir)):
                target_dir = os.path.join(model_path,dir[i])
                target_file =f'{target_dir}\\{value}'    
                if os.path.isfile(target_file) and os.path.exists(target_file):
                    path = target_file
                    # model_info['var'] = scr_key
                    # model_info['name'] = target_file.split('\\')[-1].split('.')[0]
                    model_info['dir'] = dir[i]
                    model_info['format'] = path.split('.')[-1]
                    model_info['input'] = value
                    model_info['path'] = path
                    break

            # file = str(value).rsplit('\\')[-1]
            # print(f'{'-'*20}\ninput:[{scr_key}] - [{file}]\n{'-'*20}')
            # for k,v in model_info.items():
            #     print(f'{' '}{k}:{v}')
            return model_info

        input_info = None
        if type(scr_value) == str:
            if len(scr_value.split('\\'))>1 or scr_value.rsplit('.')[-1] in ['safetensors','pt','gguf']:
               input_info = mode_info(scr_value)
            else:
                input_info = scr_value
        elif type(scr_value) in [int,float,bool]:
            input_info = scr_value
        return input_info

    # data read
    def get_slots_data(s,data_dict):
        data_keys = data_dict.keys()
        print(data_keys)
        pack_str,pack_num = {},{}
        # read slots
        for i in data_keys:
            scr_dict={}
            scr_dict[i] = data_dict[i]
            scr = data_dict[i]
            file_info = s.get_input_slots(scr_dict)
            pack_str[i]=file_info if file_info not in [None,'None','null',''] else '-1'
            # if type(scr) == str: #and len(scr.split('.'))>1:                
            #     file_info = s.get_input_slots(scr_dict)
            #     pack_str[i]=file_info
            # elif type(scr) == float or type(scr) == int:
            #     sub_pack_num = {
            #     }   
        datatree = {
            SIGNATURE:True,
            'content':pack_str,
        }
        return datatree

    # data check
    def check_slots_data_pipe(s,afpipe_in):
        content_pack = afpipe_in.get('content',{})

        model_pack = {}
        set_pack = {}
        model_load_dict = {}
        slots_load_dict = {}
        for i in content_pack:
            if type(content_pack[i]) == dict :
                # and str(i).split('_')[0].lower() in ['model','clip','vae','lora','controlnet'] and content_pack[i] != '-1' and \
                # len(str(content_pack[i]).split('.')) >1 :
                model_pack[i] = content_pack[i]
                # print(f'load model {i} -> {content_pack[i]}')
            elif content_pack[i] != '-1':
                set_pack[i] = content_pack[i]
                # print(f'load data {i} -> {content_pack[i]}')


        # load models check
        # if len(model_pack)>0:
        #     for slot in model_pack:
        #         slot_name = slot
        #         model_info = model_pack[slot]
        #         model_input = model_info['input']
        #         # model_format = model_info['format']
        #         # model_dir = model_info['dir']

        #         model_load_dict[slot_name] = model_input

        slots_load_dict = {
            SIGNATURE:True,
            'models': model_pack,
            'set': set_pack,
        }
        return (slots_load_dict,)    
        # print(f'load data  -> {slots_load_dict['models']}')


    # auto set preset labels
    def auto_updata_perset_label(s) ->dict:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
        parmas_dir = os.path.join(root,'presets')
        format = '.txt'

        txt_dict = {'default':''}
        txt_list = []
        # txt_dict['default'] = ''
        if os.path.exists(parmas_dir):
            with os.scandir(parmas_dir) as files:
                for file in files:
                    if file.is_file() and file.name.lower().endswith(format.lower()):
                        txt_dict[file.name.split('.')[0]] = file.path
                        # txt_list.append(file.name.split('.')[0])
                if len(txt_dict)==0:
                    txt_dict['not found'] = ''
        else:
            print(f'warning: 根目录下未找到配置文件 -> {parmas_dir.split('ComfyUI')[-1][1:]}')
            # return ['no preset'] +['default']
        # print(txt_dict)
        return txt_dict
    
    # set parameter to default
    def parameter_set_default(s,input_types_dict):
        vars_dict = {}
        # exclude = ['weight_mode','preset','round_multiple','use_vaedecode_tiled','tile_size','overlap','temporal_size','temporal_overlap']
        exclude = ['preset',]
        if len(input_types_dict)>0:
            temp = list(input_types_dict.keys())
            for n in temp:
                var = input_types_dict.get(n)
                for k,v in var.items():
                    if k not in exclude and len(v)>1 and isinstance(v[1],dict) and 'default' in v[1]:
                        vars_dict[k] = v[1]['default']
        return vars_dict
                
    # set parameter preset
    def parameter_set(s,preset_name,vars_default_dict):
        def infer_and_convert(val: str, prefer_int_for_float: bool = False):
            """智能推断并转换字符串类型"""
            if not isinstance(val, str):
                return val, type(val)
                
            s = val.strip()
            if not s:  # 空字符串保持原样或转为 None，按需调整
                return s, str

            # 1. 优先尝试 int（避免 9 被转成 9.0）
            try:
                return int(s), int
            except ValueError:
                pass

            # 2. 尝试 float（兼容 1.0, .5, 1e3, inf 等）
            try:
                f = float(s)
                # 可选：若业务希望 1.0 → 1，可在此处拦截
                if prefer_int_for_float and f.is_integer():
                    return int(f), int
                return f, float
            except ValueError:
                pass

            # 3. 常见布尔值映射
            if s.lower() in ('true', 'false', 'yes', 'no', '1', '0'):
                b = s.lower() in ('true', 'yes', '1')
                return b, bool

            # 4. 兜底保持字符串
            return s, str

        def get_model_input_path(model_name) -> str:
            comfy_root = f'{os.path.abspath(__file__).split('custom_nodes')[0]}'
            model_path = os.path.join(comfy_root,'models')
            # model path
            sreach_dir = ['checkpoints','diffusion_models','unet',
                          'text_encoders','clip','clip_gguf',
                          'vae',
                          'lora','loras','controlnet','model_patches']
            sreach_path = [os.path.join(model_path,sub) for sub in sreach_dir]

            for s_dir in sreach_path:
                if not os.path.isdir(s_dir):
                    continue
                for dirpath,dirnames,filenames in os.walk(s_dir):
                    if model_name in filenames:
                        full_path = os.path.join(dirpath,model_name)
                        dir = s_dir
                        input_path = full_path.split(dir)[-1][1:]
                        # print(f'run test -> {var}:{input_path}')
                        return input_path
            return None

        def read_section_from_prefix(file_path,clip_type) -> list:
            section_line =[]
            if not os.path.exists(file_path):
                print(f' -- read_section_from_prefix:文件不存在 -> {file_path}')
                return section_line
            
            is_recording = False
            with open(file_path,'r',encoding='utf-8') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith('=') :
                        if is_recording:
                            break
                        if len(stripped) == len(clip_type) or stripped == clip_type:
                            is_recording = True
                            continue
                    if is_recording :
                        section_line.append(line.strip()) 
            return section_line
                
        def read_line_from_file(file_path)->list:
            section_line =[]
            if not os.path.exists(file_path):
                print(f' -- read_section_from_prefix:文件不存在 -> {file_path}')
                return section_line
            
            with open(file_path,'r',encoding='utf-8') as f:
                for line in f:
                    section_line.append(line.strip()) 
            return section_line


        # 当标签=['no found','default']，不读取txt文件
        preset_file_dict = s.auto_updata_perset_label()
        # print(f' --preset_file_dict:{preset_file_dict}')
        if preset_name in ['no found','default']:
            return {'stack':'null'}
                            
        file_path = preset_file_dict[preset_name]
        # print(f' -- file path:{file_path}')
        top_prefix = ['=','-','+','_','*','/','\\','[','{','【']
        section_line=read_line_from_file(file_path)
        var_dict ={}
        if len(section_line)>0:
            for line in section_line:
                line = line.strip()
                if not line or ":" not in line and line.startswith(tuple(top_prefix)):
                    continue
                # 拆分参数
                if ':' in line:
                    var, value = line.split(":", 1)
                    value_t = infer_and_convert(value)[0]
                    # print(f' -- str infer_and_convert:{var}, {value} -> {value_t}')
                    if type(value_t)==str:
                        if value_t.split('.')[-1] in ['safetensors','gguf','pt','ckpt']:
                            input_path = get_model_input_path(value_t)
                            var_dict[var]=input_path
                        else:
                            var_dict[var]=value_t
                            # print(f' -- str but not model:{var_dict['resolution']}')
                            # print(f' -- str but not model:{var_dict.keys()}')
                    else:
                        var_dict[var]=value_t
        # print(f' -- func parameter load run')
        var_dict_f = {}
        prompt_profix = ['prompt','p','pos','positive','n','neg','negative','提示','提示词','正向','反向']
        if len(preset_name.split('_p'))>1:
            # 预设前缀为带_的关键字时，只更新提示变量，其它保持不变
            for p in var_dict:
                if p in ['pos','neg','clip_type']:
                    var_dict_f[p] = var_dict[p]
            pass
        else:
            for k,v in vars_default_dict.items():
                if k in var_dict:
                # if k in var_dict and str(var_dict[k]).strip() not in ['','None']:
                    var_dict_f[k] = var_dict[k]
                    # print(f'\t-> set:{var_dict_f[k]} = src:{var_dict[k]}')  
                    # print(f'\tset-> {k}:{var_dict[k]}')  
                else:
                    var_dict_f[k] = v
                    # print(f'\tdef   {k}:{v}')  
        return var_dict_f

    # save current preset ==========================？
    def parameter_save(s,save_dict):
        from datetime import datetime
        root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
        parmas_dir = os.path.join(root,'presets')
        dir_path = ''
        if os.path.exists(parmas_dir):
            dir_path = parmas_dir
        else:
            os.makedirs(parmas_dir)
            print(f' -- params目录不存在,\n -- 已创建 -> {parmas_dir}')

        now = datetime.now()
        # 格式化为指定格式的字符串
        clip_type = save_dict['clip_type']
        formatted_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f'{clip_type}_{formatted_time}.txt'
        save_path = os.path.join(dir_path,file_name)

        with open(save_path,'w',encoding='utf-8') as f:
            f.write(f'[{str(clip_type).upper()}]\n')
            for k,v in save_dict.items():
                if k.split('_')[0] in ['model','clip','vae','lora','control'] and type(v) == str and v.rsplit('.')[-1] in ['safetensors','gguf','pt','ckpt']:
                    v = v.rsplit('\\')[-1]
                f.write(f'{k}:{v}\n')
            print(f' -- save current preset success ~! \n\tpath:{dir_path}')
        return save_dict

        



    # loading=====================================================
    def model_load(s,model_pack,weight_mode):
        mod_load = None
        mod_model,mod_clip,mod_vae = None,None,None
        mod_includ_clip,mod_includ_vae = False,False
        if model_pack:
            mod_format = model_pack['format']
            mod_dir = model_pack['dir']
            mod_input = model_pack['input']

            # logging.info(f"准备加载模型，路径: {mod_input}")
            # ckpt_path=folder_paths.get_full_path("checkpoints", mod_input)
            # out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True)
            # print(f' --> out:{out}')

            # model_dir = ['checkpoint','diffusion_models','unet']
            if mod_dir == 'checkpoints':
                mod_load = nodes.CheckpointLoaderSimple().load_checkpoint(mod_input)
                mod_model = mod_load[0] if mod_load[0] else None
            else:
                if mod_format == 'gguf':
                    try:
                        # GGUF 的 patch_dtype 不支持 FP8，选 FP8 时自动降级到 bfloat16
                        dtype = "bfloat16" if "fp8" in weight_mode else weight_mode
                        mod_load = gg.LoaderGGUFAdvanced().load_model(mod_input,patch_dtype=dtype)
                    except ImportError as e:
                        print(e)
                else:
                    # model_diffuse_path = folder_paths.get_full_path(mod_dir, mod_input) if mod_input != 'None' else None
                    # mod_load = comfy.sd.load_diffusion_model(model_diffuse_path)
                    mod_load = nodes.UNETLoader().load_unet(mod_input,weight_dtype=weight_mode)
                if type(mod_load) == tuple and len(mod_load) >=1:
                    mod_model = mod_load[0]  
                else:                      
                    mod_model = mod_load    
                # print(f' <--> model tuple ?:{type(mod_model)} {mod_model}')

            # check layer and check clip and vae in models?
            load_type = ['model','model+clip','model+vae','model+clip+vae']
            model_type = load_type[0]
            model_layer = s._check_model_layer(mod_load)
            # print(f' --check_model_layer:{model_layer}')
            if model_layer[1] == 1 and model_layer[2] == 0:
                model_type = load_type[1]
                mod_clip = mod_load[1]
            elif model_layer[1] == 0 and model_layer[2] == 1:                
                model_type = load_type[2]
                mod_vae = mod_load[2]
            elif model_layer[1] == 1 and model_layer[2] == 1:
                model_type = load_type[3]
                mod_clip = mod_load[1]
                mod_vae = mod_load[2]            
        return model_type,(mod_model,mod_clip,mod_vae)

    def clip_load(s,clip_1_pack,clip_2_pack,clip_type,layer_skip=-1,device='default'):
        clip_dir = ['clip','clips','text_encoders']
        if clip_1_pack:
            clip1_format = clip_1_pack['format']
            clip1_dir = clip_1_pack['dir']
            clip1_input = clip_1_pack['input']

        if clip_2_pack:
            clip2_format = clip_2_pack['format']
            clip2_dir = clip_2_pack['dir']
            clip2_input = clip_2_pack['input']

        clip_mix = None
        Warning_txt = 'clip1 clip2 不能相同 !!!'
        # Warning_txt = 'clip1 clip2 the same !!!'
        if _gguf_exist:
            if clip_1_pack and clip_2_pack == None:
                # print(f'--------gguf----------clip_1')        
                clip_mix = gg.ClipLoaderGGUF().load_clip(clip1_input,clip_type,device)
            elif clip_1_pack == None and clip_2_pack:
                # print(f'--------gguf----------clip_2')        
                clip_mix = gg.ClipLoaderGGUF().load_clip(clip2_input,clip_type,device)
            elif clip_1_pack and clip_2_pack:
                if clip1_input == clip2_input:
                    raise ValueError (Warning_txt)
                # print(f'--------gguf----------clip 1 + clip_2')        
                clip_mix = gg.DualClipLoaderGGUF().load_clip(clip1_input,clip2_input,clip_type,device)
        else:
            # print(f'--------saftensors---------gguf-false')        
            if clip_1_pack and clip_2_pack == None and clip1_format != 'gguf':
                clip_mix = nodes.CLIPLoader().load_clip(clip1_input,clip_type)
            elif clip_1_pack and clip_2_pack \
                and clip1_format != 'gguf' and clip2_format != 'gguf':
                if clip1_input == clip2_input:
                    raise ValueError (Warning_txt)
                clip_mix = nodes.DualCLIPLoader().load_clip(clip1_input,clip2_input,clip_type)

        if len(clip_mix)>=1:
            if clip_1_pack and clip_2_pack:
                print(f' -- double clip --')
            else:
                print(f' -- single clip --')
            clip_mix = s._unpack_tuple(clip_mix)

        return clip_mix

    def set_layer_skip(s,clip,stop_at_clip_layer,clip_type='sdxl'):
        # items = ['stable_diffusion','stable_cascade','sd3','sdxl',]
        clip_f = None 
        if clip:
            # model_format,model_dir_name = s.get_model_and_dir_type(clip)
            # if clip_type in items and model_dir_name == 'checkpoints':
            clip_f = s._unpack_tuple(nodes.CLIPSetLastLayer().set_last_layer(clip,stop_at_clip_layer))
        return clip_f

    def set_lcm_for_sd_xl(s,model=None,sampling='lcm',zsnr=False):
        model_f = None 
        if model is not None:
            model_f = nodes_model_advanced.ModelSamplingDiscrete().patch(model, sampling, zsnr)[0]
        return model_f


    def text_encode(s,clip_cls,text):
        cond,pooled,pooled_output = None,None,None
        # 0确保为空时encode长度也一至
        if not text and not text.strip():
            text = ' '
        # 1编译
        tokens = clip_cls.tokenize(text)
        encode = clip_cls.encode_from_tokens_scheduled(tokens)
        # # # flux 在pooled_dict中添加guidance的键值对
        # if clip_type in ['flux','flux2']:
        #     # 2解包，零化pooled，有仅零化，无则用1填充，并添加guidance键值对
        #     if isinstance(encode,tuple) and len(encode) == 2:
        #         cond,pooled = encode
        #     else:
        #         cond = encode
        #         pooled = None
        #     if pooled is None:
        #         if cond and len(cond)>0 and len(cond[0])>0:
        #             pooled = torch.zeros_like(cond[0][0])
        #         else:
        #             pooled = torch.zeros_like((1,2048)) 
        #     pooled_output = {'pooled_output':pooled}
        #     pooled_output={'guidance':guiance}
        #     # 3恢复结构
        #     text_encode = [[cond,pooled_output]]
        # else:
        #     text_encode = encode
        return (encode,)

    def text_encode_zero(s,text_encode,neg_zero=True):
        con = []
        if text_encode and neg_zero:
            cond = []
            pooled = text_encode[0][1].copy()
            pooled_output = pooled.get('pooled_output',None)
            # pooled 零化
            if pooled_output is not None:
                pooled['pooled_output'] = torch.zeros_like(pooled_output)
            
            conditioning_lyrics = pooled.get('conditioning_lyrics',None)
            if conditioning_lyrics is not None:
                pooled['conditioning_lyrics'] = torch.zeros_like(conditioning_lyrics)
            # cond 零化
            cond = torch.zeros_like(text_encode[0][0])
            # 恢复结构
            con.append([cond,pooled])  
        return (con,)

    def text_encode_flux(s,clip_type,clip_cls,text_l,text_t5,guidance):
        encode = None
        if clip_type in ['flux']:
            # 编译clip_l模型
            tokens_l = clip_cls.tokenize(text_l)
            # 编译t5xxl模型
            tokens_t5 = clip_cls.tokenize(text_t5)
            # 再用t5编译的值修改tokens元组中的键值对
            tokens_l['t5xxl'] = tokens_t5['t5xxl']
            encode = clip_cls.encode_from_tokens_scheduled(tokens_l,add_dict={'guidance':guidance})
        return (encode,)

    def set_chroma(s,model_cls,clip_cls,min_padding,min_length):
        model_f,clip_f =None,None
        if model_cls is not None and clip_cls is not None:
            model_f, = nodes_chroma_radiance.ChromaRadianceOptions.execute(model=model_cls,
                        preserve_wrapper=True,start_sigma=1.0,end_sigma=0.0,nerf_tile_size=-1,force_sequential_txt_ids=False)
            clip_f, = nodes_cond.T5TokenizerOptions().execute(clip_cls, min_padding, min_length)
        return model_f,clip_f


    def conditioning_set(s,clip_cls,pos_str,neg_str,clip_type,neg_zero=False,guidance=3.5,krea2_rebalance=4.0,per_layer_weights=None):
        pos_encode,neg_encode=None,None

        if clip_cls:
            if clip_type in ['flux']:
                pos_encode = s.text_encode_flux(clip_type,clip_cls,neg_str,pos_str,guidance)[0]
                neg_encode = s.text_encode_zero(pos_encode,True)[0]
            else:
                pos_encode = s.text_encode(clip_cls,pos_str)[0]
                if neg_zero:
                    neg_encode = s.text_encode_zero(pos_encode,neg_zero)[0]
                else:
                    neg_encode = s.text_encode(clip_cls,neg_str)[0]
                    
            # Krea2 Rebalance
            if clip_type == 'krea2':
               pos_encode= Krea2Rebalance.main(s,pos_encode, krea2_rebalance, per_layer_weights)[0]
            #    print(f'--krea2 pos:{type(pos_encode)}-{pos_encode}')

        return (pos_encode,neg_encode)


    def ref_encode_to_conditioning_for_flux2(s,vae_cls,image,text_conditioning):
        text_encode_f = None
        if vae_cls and image is not None:
            latent_encode = nodes.VAEEncode().encode(vae_cls,image)
            # if type(latent_encode) == tuple and len(latent_encode)>=1:
            #     latent_encode = latent_encode[0]
            txt_encode = nodes_edit_model.ReferenceLatent().execute(text_conditioning,latent_encode[0])
            # print(f' -- refrence conditioning:{len(txt_encode[0])} \n{txt_encode[0]}')

            # if type(txt_encode) == tuple and len(txt_encode)>=1:
            #     text_encode_f = txt_encode[0]
            # else:
            #     text_encode_f = txt_encode
            text_encode_f = txt_encode[0]
        return text_encode_f
    
    def ref_encode_to_conditioning_for_qwenedit(s,clip_cls,pos_txt,vae_cls,image1=None, image2=None, image3=None, image4=None):
        pos_encode,neg_encode = None,None
        t_enocde = diy.TextEncodeQwenImageEditPlus().execute(clip_cls, pos_txt, vae_cls, image1, image2, image3, image4)
        # print(f' -- qwen edit encode:{type(t_enocde)}\n{t_enocde}')
        pos_encode = t_enocde[0]
        neg_encode = s.text_encode_zero(pos_encode,True)[0]
        return (pos_encode,neg_encode)
    
    def ref_qwen_edit_from_Qweneditutils(s,clip_cls,pos_txt,vae_cls,image1=None, image2=None, image3=None, image4=None,mask=None,
                                          latent_width=1024,latent_height=1024):
        pos_encode,neg_encode,latent,main_image,noise_mask=None,None,None,None,None
                
        color = '#00ff00'
        resize_image = pad.ResizeAndPadImage().execute(image1,latent_width,latent_height,color,'area')
        image1_height = resize_image[0].shape[1]
        image1_width = resize_image[0].shape[2]
        ref_longest_edge = max(image1_width,image1_height)
        # print(f'resize_image type:{type(resize_image)} --image1_height:{image1_height} - image1_width:{image1_width} || ref_longest_edge:{ref_longest_edge}')

        t = diy.QwenEditTextEncode_EditUtils().encode(clip_cls,vae_cls,pos_txt,resize_image[0],image2,image3,image4,mask,ref_longest_edge)
                                                    
        pos_encode = t[0]
        neg_encode = s.text_encode_zero(pos_encode,True)[0]
        latent=t[1]
        main_image=t[3]
        noise_mask=t[4]
        return (pos_encode,neg_encode,latent,main_image,noise_mask)

    def latent_set(s, clip_type, image, mask, width, height, batch, vae_cls=None, use_edit_mode=False):
        #========================================
        # # empty latent
        # def generate(self, width, height, batch_size=1):
        #     latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
        #     return ({"samples": latent, "downscale_ratio_spacial": 8}, )
        
        # # empty flux2 latent
        # def execute(cls, width, height, batch_size=1) -> io.NodeOutput:
        #     latent = torch.zeros([batch_size, 128, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        #     return io.NodeOutput({"samples": latent})
        
        # # vae encode
        # def encode(self, vae, image):
        #     t = vae.encode(image)
        #     return ({"samples":t}, )
        
        # # set latent mask
        # def set_mask(self, latent, mask):
        #     s = latent.copy()
        #     s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        #     return (s,)
        #========================================
        # latent
        latent_t = None
        if image is not None and not use_edit_mode:
            latent_t = nodes.VAEEncode().encode(vae_cls,image)[0]
        else:            
            if clip_type == 'flux2':
                latent_t = nodes_flux.EmptyFlux2LatentImage().execute(width,height,batch)[0]
            else:
                latent_t = nodes.EmptyLatentImage().generate(width,height,batch)

        if mask is not None:
            if type(latent_t) == tuple and len(latent_t)>2:
                temp_latent = latent_t[0]
            else:
                temp_latent = latent_t
            n_latent = nodes.SetLatentNoiseMask().set_mask(temp_latent,mask[0])
            latent_t = n_latent[0]
            # print(f' --latent_t:{type(latent_t)}\n{latent_t}')
        
        if type(latent_t) == tuple and len(latent_t)>=1:
            latent_o = (latent_t)[0]
            # latent_o = s._unpack_tuple(latent_t)
        else:
            latent_o = latent_t

        return (latent_o,)

    def latent_upscale(s,latent_image, upscale_method="nearest-exact", scale_by=1.0):
        sample_u = nodes.LatentUpscaleBy().upscale(latent_image, upscale_method,scale_by)
        return sample_u[0]

    def vae_set(s,vae_pack):
        vae_f = None
        if vae_pack:
            vae_format = vae_pack['format']
            vae_dir = vae_pack['dir']
            vae_input = vae_pack['input']
            # vae_format,vae_dir = s.get_model_and_dir_type(vae)
            vae_f = s._unpack_tuple(nodes.VAELoader().load_vae(vae_input))
        return vae_f

    def lora_set(s,model,lora_pack,strength):
        model_f = None
        if lora_pack:
            # lora_format = lora_pack['format']
            lora_dir = lora_pack['dir']
            lora_input = lora_pack['input']
            lora_path = lora_pack['path']   # 完整路径/绝对路径
            # model_f = s._unpack_tuple(nodes.LoraLoaderModelOnly().load_lora_model_only(model,lora_input,strength))           

            lora = comfy.utils.load_torch_file(lora_path)
            model_f,c = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
            # print(f' - {type(model_f)}')
        return model_f


    def set_seed(s,auto_seed,seed):
        import math
        seed_f = None
        seed_random = math.floor(math.random() * 1000000000000);
        if auto_seed:
            seed_f = seed_random
            # seed_f = nodes.SetSeed().set_seed(seed)
        return seed_f


    def set_flux2_kvcache(s,model):
        model_cls = None
        if model is not None:
            model_cls = nodes_flux.FluxKVCache().execute(model)
        return model_cls

    def ksample_set(s,clip_type,ks_type,model_1_cls,
                        seed,positive, negative, latent_image,latent_width,latent_height, 
                        steps_1, cfg_1, sampler_name_1, scheduler_1, denoise_1=1.0, 
                        steps_2=8, cfg_2=1.0, sampler_name_2='euler',scheduler_2='simaple', denoise_2=1.0, 
                        use_fls=False,fovea_strength=3.0,sharpness=0.5,mask_inertia=0.85,
                        start_step=0,end_step=10000,
                        use_model_2=False,model_2_cls=None,
                        scale_by_1=1.0, scale_by_2=1.0,  
                        upscale_method_1="nearest-exact", upscale_method_2="nearest-exact",                               
                        # add_noise_1='enable',return_with_leftover_noise_1='enable', 
                        # add_noise_2='disable',return_with_leftover_noise_2='disable', 
                         ):
        # print(f' -- ks_type:{type(ks_type)} {ks_type}')
        k_sample_f,ksample_latent,scheduler_f = None,None,None
        k_type = KSAMPLE_TYPE




        # k_sample_f = nodes.KSampler().sample(                    
        #                 model_1_cls,seed, steps_1, cfg_1, sampler_name_1, scheduler_1, 
        #                 positive, negative, latent_image, 
        #                 denoise_1)



        if clip_type == 'flux2':
            scheduler_f_1 = nodes_flux.Flux2Scheduler().execute(steps_1,latent_width,latent_height)
            scheduler_f_2 = nodes_flux.Flux2Scheduler().execute(steps_1,latent_width,latent_height)
        else:
            scheduler_f_1 = scheduler_1
            scheduler_f_2 = scheduler_2

        if scale_by_1 !=1.0:
            latent_u_1 = s.latent_upscale(latent_image, upscale_method_1, scale_by_1)
        else:
            latent_u_1 = latent_image

        
        # KSAMPLE_TYPE = ['normal','normal double','advanced single','advanced double[base/turbo]','custom']
        match ks_type:
            # normal
            case _ if ks_type == k_type[0]:
                if not use_fls:
                    k_sample_f = nodes.KSampler().sample(                    
                        model_1_cls,seed, steps_1, cfg_1, sampler_name_1, scheduler_f_1, 
                        positive, negative, latent_u_1, 
                        denoise_1)
                else:
                    fls_sample,_ = fls().sample_fls(model_1_cls,seed, steps_1, cfg_1, sampler_name_1, scheduler_f_1, 
                        positive, negative, latent_u_1, 
                        denoise_1,
                        fovea_strength,sharpness,mask_inertia)
                    k_sample_f = (fls_sample,)
                    # print(f'-- fls k_sample_f:{k_sample_f}')

            # normal double
            case _ if ks_type == k_type[1]:
                if not use_fls:
                    k_sample_1_f = nodes.KSampler().sample(                    
                        model_1_cls,seed, steps_1, cfg_1, sampler_name_1, scheduler_f_1, 
                        positive, negative, latent_u_1, 
                        denoise_1)
                else:
                    fls_sample,_ = fls().sample_fls(model_1_cls,seed, steps_1, cfg_1, sampler_name_1, scheduler_f_1, 
                        positive, negative, latent_u_1, 
                        denoise_1,
                        fovea_strength,sharpness,mask_inertia)
                    k_sample_1_f = (fls_sample,)

                if scale_by_2 !=1.0:
                    latent_u_2 = s.latent_upscale(k_sample_1_f[0], upscale_method_2, scale_by_2)
                else:
                    latent_u_2 = k_sample_1_f[0]
                
                if use_model_2 and model_2_cls: 
                    model_2_t = model_2_cls
                    sampler_name_t = sampler_name_2
                    scheduler_t = scheduler_f_2
                else:
                    model_2_t = model_1_cls
                    sampler_name_t = sampler_name_1
                    scheduler_t = scheduler_f_1

                k_sample_f = nodes.KSampler().sample(                    
                        model_2_t,seed, steps_2, cfg_2, sampler_name_t, scheduler_t, 
                        positive, negative, latent_u_2, 
                        denoise_2)

            # advance single
            case _ if ks_type == k_type[2]:
                k_sample_f = nodes.KSamplerAdvanced().sample(
                    model_1_cls, 'enable', seed,steps_1, cfg_1, sampler_name_1, scheduler_f_1, 
                    positive, negative,latent_u_1, 
                    start_step, end_step, 'enable', denoise=1.0
                )

            # advance double base&turbo
            case _ if ks_type == k_type[3]:
                if use_model_2 and model_2_cls:
                    k_sample_1 = nodes.KSamplerAdvanced().sample(
                        model_1_cls, 'enable', seed, steps_1, cfg_1, sampler_name_1, scheduler_f_1, 
                        positive, negative,latent_u_1, 
                        start_step, end_step, 'enable', denoise=1.0
                    )
                    k_sample_f = nodes.KSamplerAdvanced().sample(
                        model_2_cls, 'disable', seed, steps_1, 1.0, sampler_name_2, scheduler_f_2, 
                        positive, negative,k_sample_1[0], 
                        end_step, steps_1, return_with_leftover_noise='disable', denoise=1.0
                    )
                else:
                    raise RuntimeError('The second diffuse model is not loaded or use model 2 is not open.')
 
        
        # print(f'-- ksample_latent:{ksample_latent}')
        ksample_latent = k_sample_f[0]
        return (ksample_latent,)


    def vaedecoder_set(s,vae,ksample_latent,use_vaedecode_tiled=False,tile_size=512,overlap=64,temporal_size=64,temporal_overlap=8):
        image = None
        latent_decode = ksample_latent
        # latent_decode = ksample_latent[0][0]
        if use_vaedecode_tiled:
            image = nodes.VAEDecodeTiled().decode(
                vae, latent_decode, tile_size, overlap, temporal_size, temporal_overlap)[0]
            # print(f' -- vae decode tiled on')
        else:
            image = nodes.VAEDecode().decode(vae,latent_decode)[0]
            # image = vae.decode(latent_decode['samples'])
            pass
        return image


    def ModelSamplingAuraFlow(s,model_cls,shift=1.73):
        model_f = nodes_model_advanced.ModelSamplingAuraFlow().patch_aura(model_cls,shift)[0]
        return model_f

    # 针对编辑模型，如2511/firered
    def CFGNorm(s,model_cls,strength=1):
        model_f = None
        model_f = nodes_cfg.CFGNorm().execute(model_cls,strength)[0]
        # print(type(model_f))
        return model_f
    
    def ModelSamplingFlux(s,model_cls,max_shift=1.15, base_shift=0.5, width=1024, height=1024):
        model_f = nodes_model_advanced.ModelSamplingFlux().patch(model_cls, max_shift, base_shift, width, height)[0]
        return model_f


    # controlnet
    def get_controlnet_type(s):
        items = ['Normal', 'Zimage']
        return items    
    # load controlnet model
    def get_model_controlnet_list(s):
        controlnet_normal_path = folder_paths.get_filename_list('controlnet') if 'controlnet' in folder_paths.folder_names_and_paths else []
        controlnet_model_patches_path = folder_paths.get_filename_list('model_patches') if 'model_patches' in folder_paths.folder_names_and_paths else []
        cn_normal_clear = [str(name).strip() for name in controlnet_normal_path] if controlnet_normal_path else []
        cn_normal_path_clear = [str(name).strip() for name in controlnet_model_patches_path] if controlnet_model_patches_path else []
        return ['None'] + cn_normal_clear + cn_normal_path_clear
    def load_controlnet_normal(s,model_controlnet):
        model_controlnet_f = None
        model_controlnet_f, = nodes.ControlNetLoader().load_controlnet(model_controlnet) if model_controlnet != 'None' else None
        return model_controlnet_f
    def load_controlnet_model_patch(s,model_controlnet):
        model_controlnet_f = None
        model_controlnet_f, = nodes_model_patch.ModelPatchLoader().load_model_patch(model_controlnet) if model_controlnet != 'None' else None
        return model_controlnet_f
    # appley controlnet判断
    def set_applycontrolnet_normal(s,ctlmodel_cls,vae,image_controlnet,strength,startPercent,endPercent,
                              prompt,prompt_nag,extra_concat=[]):
        prompt_c,prompt_c_nag = None,None
        if ctlmodel_cls != 'None':
            prompt_c,prompt_c_nag = nodes.ControlNetApplyAdvanced().apply_controlnet(prompt,prompt_nag,
                                                                            ctlmodel_cls,image_controlnet,
                                                                            strength,startPercent, endPercent, 
                                                                            vae=vae,extra_concat=extra_concat)
        return prompt_c,prompt_c_nag  
    def set_applycontrolnet_model_patch(s,model_controlnet,model,vae,image,strength, inpaint_image=None, mask=None):
        model_f = None
        if model_controlnet != 'None':
            model_f, = nodes_model_patch.QwenImageDiffsynthControlnet().diffsynth_controlnet(model,model_controlnet,vae,
                                                                                                    image,
                                                                                                    strength,inpaint_image,mask)
        return model_f

    def set_DifferentialDiffusion(s,model_cls,strength=1.0):
        model_f = None
        if model_cls is not None:
            model_f = nodes_differential_diffusion.DifferentialDiffusion().execute(model_cls,strength)[0]
        return model_f


    # 外挂lora解包
    def unpack_pipe_loras_stack(s,pipe):
        lora_list = []
        if isinstance(pipe,dict):
            loras_dict = pipe.get('content',None)
            if len(loras_dict)>0:
                for v in loras_dict.values():
                    lora_pack = {}
                    lora_strength = 0
                    lora_pack['dir'] = v['dir']
                    lora_pack['format'] = v['format']
                    lora_pack['input'] = v['input']
                    lora_pack['path'] = v['path']
                    lora_strength = v['strength']
                    lora_list.append((lora_pack,lora_strength))
        return lora_list

