from .utils import SOLUTJION_TYPE,ASPECTRATIO,KSAMPLE_TYPE
from .utils import PIPE_TYPE_MODEL,PIPE_TYPE_LORA,SIGNATURE,DTYPE_UNIFIED,UPSCALE_METHODS
from .utils import aftools


class pipe_loras_pack():
    af = aftools()
    @classmethod
    def INPUT_TYPES(s):
        return{
            'optional': {
                'data':(PIPE_TYPE_LORA,),
                            
                "lora_1": ('BOOLEAN',{'default':True}),
                'lora_1_model': (s.af.get_model_lora_list(),{'default': 'None'}),    
                'lora_1_strength': ('FLOAT', {'default': 1.0, 'min': -3.0, 'max': 5.0, 'step': 0.01}),         
                "lora_2": ('BOOLEAN',{'default':False}),
                'lora_2_model': (s.af.get_model_lora_list().copy(),{'default': 'None'}),             
                'lora_2_strength': ('FLOAT', {'default': 1.0, 'min': -3.0, 'max': 5.0, 'step': 0.01}),
                "lora_3": ('BOOLEAN',{'default':False}),
                'lora_3_model': (s.af.get_model_lora_list(),{'default': 'None'}),             
                'lora_3_strength': ('FLOAT', {'default': 1.0, 'min': -3.0, 'max': 5.0, 'step': 0.01}),
                "lora_4": ('BOOLEAN',{'default':False}),
                'lora_4_model': (s.af.get_model_lora_list().copy(),{'default': 'None'}),             
                'lora_4_strength': ('FLOAT', {'default': 1.0, 'min': -3.0, 'max': 5.0, 'step': 0.01}),
                "lora_5": ('BOOLEAN',{'default':False}),
                'lora_5_model': (s.af.get_model_lora_list(),{'default': 'None'}),             
                'lora_5_strength': ('FLOAT', {'default': 1.0, 'min': -3.0, 'max': 5.0, 'step': 0.01}),
                "lora_6": ('BOOLEAN',{'default':False}),
                'lora_6_model': (s.af.get_model_lora_list().copy(),{'default': 'None'}),             
                'lora_6_strength': ('FLOAT', {'default': 1.0, 'min': -3.0, 'max': 5.0, 'step': 0.01}),
                "lora_7": ('BOOLEAN',{'default':False}),
                'lora_7_model': (s.af.get_model_lora_list().copy(),{'default': 'None'}),             
                'lora_7_strength': ('FLOAT', {'default': 1.0, 'min': -3.0, 'max': 5.0, 'step': 0.01}),
                "lora_8": ('BOOLEAN',{'default':False}),
                'lora_8_model': (s.af.get_model_lora_list().copy(),{'default': 'None'}),             
                'lora_8_strength': ('FLOAT', {'default': 1.0, 'min': -3.0, 'max': 5.0, 'step': 0.01}),
                "lora_9": ('BOOLEAN',{'default':False}),
                'lora_9_model': (s.af.get_model_lora_list().copy(),{'default': 'None'}),             
                'lora_9_strength': ('FLOAT', {'default': 1.0, 'min': -3.0, 'max': 5.0, 'step': 0.01}),
                "lora_10": ('BOOLEAN',{'default':False}),
                'lora_10_model': (s.af.get_model_lora_list().copy(),{'default': 'None'}),             
                'lora_10_strength': ('FLOAT', {'default': 1.0, 'min': -3.0, 'max': 5.0, 'step': 0.01}),
            },
        }        
    RETURN_TYPES = (PIPE_TYPE_LORA,)
    RETURN_NAMES = ('data',)
    FUNCTION = 'load'
    CATEGORY = 'afar_tools'
    SEARCH_ALIASES = ['模型加载','Model Loader','Universal Loader','lora load','多模型加载','Multi lora Loader']

    def load(s,**kwargs):
        vars_dict = {}
        input_vars = s.INPUT_TYPES().get('optional',None)
        vars_name = [i for i in input_vars.keys() if input_vars is not None and i.split('_')[0] == 'lora']
        vars_dict = {n: kwargs.get(f'{n}') for n in vars_name}

        model_dict,model_pack_dict = {},{}
        for n in vars_dict:
            if n.count('_')==1 and vars_dict.get(n) ==True and vars_dict.get(f'{n}_model') != 'None':   
                lora_name = n
                model_input = vars_dict.get(f'{n}_model')
                model_strength = vars_dict.get(f'{n}_strength')
                lora_set = []       
                lora_set.append(model_input)
                lora_set.append(model_strength)
                model_dict[lora_name] = lora_set
                # print(f' -- get:{model_input}')

        # print(f' -- get:{len(model_dict)}')
        if len(model_dict)==0:
            return (None,)            
        
        print(f' - {model_dict}')
        for k,v in model_dict.items():
            model_info = s.af.mode_info(v[0])
            model_info['strength'] = vars_dict[f'{k}_strength']
            model_pack_dict[k] = model_info

        pipe_dict={
            SIGNATURE:True,
            'content':model_pack_dict,
        }       
        return (pipe_dict,)
        # return (pipe,model_1,model_2,clip_1,clip_2,vae,lora1,lora2,controlnet)


class pipe_unite_ksampler:
    af = aftools()
    @classmethod
    def INPUT_TYPES(s):
        return{
            'required':{
                'model_1': (s.af.get_model_diffusion_list(),{'default': 'None'}),
            },
            'optional': {
                'lora_1':(s.af.get_model_lora_list(),{'default': 'None'}),
                'lora_1_strength': ('FLOAT', {'default': 1.0, 'min': 0, 'max':10.0, 'step': 0.01}),
                'clip_1': (s.af.get_model_clip_list(),{'default': 'None'}),
                'clip_2': (s.af.get_model_clip_list().copy()+[''],{'default': 'None'}),
                'vae': (s.af.get_model_vae_list(),{'default': 'None'}),             

                'weight_mode': (DTYPE_UNIFIED, {'default': 'default'}),
                'clip_type': (s.af.get_clip_type_list(),{'default': s.af.get_clip_type_list()[0]}),
                
                'use_edit_mode':('BOOLEAN', {'default': False},),
                'use_model_2':('BOOLEAN', {'default': False},),                    
                'model_2': (s.af.get_model_diffusion_list().copy()+[''],{'default': 'None'}), 
                'lora_2': (s.af.get_model_lora_list().copy()+[''],{'default': 'None'},),
                'lora_2_strength': ('FLOAT', {'default': 1.0, 'min': 0, 'max':10.0, 'step': 0.01}),
                
                'use_controlnet':('BOOLEAN', {'default': False},),
                'control_model': (s.af.get_model_controlnet_list(),{'default': 'None'},),                     
                'control_strength': ('FLOAT', {'default': 1.0, 'min': -10.0, 'max': 10.0, 'step': 0.01}), 
                'control_startPercent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}), 
                'control_endPercent': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}), 
                'control_strength_model_2': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'preset': (list(s.af.auto_updata_perset_label().keys()),{'default':list(s.af.auto_updata_perset_label().keys())[0]}), 

                'pos': ('STRING', {'multiline': True, 'dynamicPrompts': True,'default':''},),
                'neg': ('STRING', {'multiline': True,'dynamicPrompts': True,'default':''},),                    
                'neg_zero':('BOOLEAN', {'default': True},),

                'match_size':('BOOLEAN', {'default': False,'tooltip':'auto match resolution from ref image1'},),
                'resolution': (SOLUTJION_TYPE,{'default':SOLUTJION_TYPE[0]}),
                'aspectRatio': (ASPECTRATIO, {'default':ASPECTRATIO[1],}),
                'megapixels': ('FLOAT', {'default': 1.0, 'min': 0.1, 'max': 16.0, 'step':0.1,}),
                'longerEdge': ('INT', {'default': 1536, 'min': 64, 'max': 2560, 'step': 2}),
                'solution_preset':(s.af.get_resolution_preset(), {'default':s.af.get_resolution_preset()[0]}),
                'width': ('INT', {'default': 512, 'min': 64, 'max': 2560, 'step': 2}),
                'height': ('INT', {'default': 1024, 'min': 64, 'max': 2560, 'step': 2}),
                'flip':('BOOLEAN', {'default': False,'tooltip':'flip the width and height in resolution'},),
                
                'batch': ('INT', {'default': 1, 'min': 1, 'max': 10, 'step': 1}),
                'layer_skip': ('INT', {'default': -1, 'min': -24, 'max': -1, 'step': 1, 'advanced': True}),
                # sampling, zsnr
                "lcm_sampling": (["eps", "v_prediction", "lcm", "x0", "img_to_img", "img_to_img_flow"], {'default': "lcm",'tooltip': 'ModelSamplingDiscrete的采样方式,lcm是基于eps的lcm采样,mg_to_img_flow是基于v_prediction的图像到图像的流式采样'},),
                'lcm_zsnr': ('BOOLEAN', {'default': False, 'advanced': True}),

                'use_kvcache': ('BOOLEAN', {'default': False,'tooltip':'use kvcache for flux2 kv'},),
                'fluxguidance': ('FLOAT', {'default': 3.5, 'min': -100, 'max': 100, 'step': 0.01, 'tooltip': 'for flux.'}),
                'shift': ('FLOAT', {'default': 3.1, 'min': 0.0, 'max': 100.0, 'step':0.01}),
                'cfgNorm':('FLOAT',{'default':1.0, 'min':0.0, 'max':100.0, 'step':0.01}), 
                               
                # krea2
                'krea2_rebalance':('FLOAT',{'default':1.0, 'min':0.0, 'max':10.0, 'step':0.01,'tooltip': 'for nsfw.'}),
                # 'per_layer_weights':('STRING',{'default':"1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0","multiline": False}),

                # fls
                'use_fls': ('BOOLEAN', {'default': False,'tooltip':'use fls ksample in normal'},),
                "fovea_strength": ("FLOAT",{"default": 3.0,"min": 0.0,"max": 10.0,"step": 0.1,"display": "slider",},),
                "sharpness": ("FLOAT",{"default": 0.5,"min": 0.0,"max": 3.0,"step": 0.05,"display": "slider",},),
                "mask_inertia": ("FLOAT",{"default": 0.85,"min": 0.0,"max": 0.99,"step": 0.01,"display": "slider",},),

                # ksampler
                'ksample_type': (KSAMPLE_TYPE,{'default':KSAMPLE_TYPE[0]},),
                'steps_1': ('INT', {'default': 8, 'min': 1, 'max': 10000,}),
                'cfg_1': ('FLOAT', {'default': 1, 'min': 0.0, 'max': 100.0, 'step':0.1, 'round': 0.01,}),
                'sampler_name_1': (s.af.get_samplers_type(), {'default':'euler'}),
                'scheduler_1': (s.af.get_schedulers_type(),{'default':'simple'}),
                'denoise_1': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,}),
                "scale_by_1": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                
                'steps_2': ('INT', {'default': 8, 'min': 1, 'max': 10000,}),
                'cfg_2': ('FLOAT', {'default': 1, 'min': 0.0, 'max': 100.0, 'step':0.1, 'round': 0.01,}),
                'sampler_name_2': (s.af.get_samplers_type().copy()+[''], {'default':'euler'}),
                'scheduler_2': (s.af.get_schedulers_type().copy()+[''],{'default':'simple'}),
                'denoise_2': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,}),
                "scale_by_2": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),

                'start_step': ('INT', {'default': 0, 'min': 0, 'max': 10000, 'advanced': True}),
                'end_step': ('INT', {'default': 10, 'min': 0, 'max': 10000, 'advanced': True}),
                # vae tiled
                'use_tiled': ('BOOLEAN', {'default': False},),
                'tile_size': ('INT', {'default': 512, 'min': 64, 'max': 4096, 'step': 32, 'advanced': True}),
                'overlap': ('INT', {'default': 64, 'min': 0, 'max': 4096, 'step': 32, 'advanced': True}),
                'temporal_size': ('INT', {'default': 64, 'min': 8, 'max': 4096, 'step': 4, 'tooltip': 'Only used for video VAEs: Amount of frames to decode at a time.', 'advanced': True}),
                'temporal_overlap': ('INT', {'default': 8, 'min': 4, 'max': 4096, 'step': 4, 'tooltip': 'Only used for video VAEs: Amount of frames to overlap.', 'advanced': True}),

                'image_1':('IMAGE',),
                'image_2':('IMAGE',),
                'image_3':('IMAGE',),
                'image_4':('IMAGE',),
                'mask':('MASK',),
                'lora_stack': (PIPE_TYPE_LORA,),
            },
            'hidden':{
                'seed': ('INT', {'default': 43, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': 'fixed'}),
                'auto_seed': ('BOOLEAN', {'default': False}),
            }
        }        
    RETURN_TYPES = ('IMAGE','MASK','IMAGE','IMAGE',)
    RETURN_NAMES = ('image','mask','refImage1','refImage2')
    FUNCTION = 'run'
    CATEGORY = 'afar_tools'
    SEARCH_ALIASES = ['模型加载','Model Loader','Universal Loader','通用加载器','多模型加载','Multi Model Loader']

    def run(s,
            model_1,model_2,clip_1,clip_2='None',vae='None',use_model_2=False,
            weight_mode='default',clip_type='None',preset='',
            lora_1='None',lora_1_strength='None',lora_2='None',lora_2_strength='None',
            pos='',neg = '',neg_zero=True,

            fluxguidance=3.5,use_kvcache=False,shift=3.1,cfgNorm=1,
            resolution=SOLUTJION_TYPE[0],width=1024,height=1024,flip=False,batch=1,match_size=False,
            solution_preset=af.get_resolution_preset()[0],
            aspectRatio='1:1',megapixels=1.0,longerEdge=1024,

            ksample_type='common',
            steps_1=8,cfg_1=1.0,sampler_name_1='euler',scheduler_1='sample',denoise_1=1.0,
            steps_2=8,cfg_2=1.0,sampler_name_2='euler',scheduler_2='sample',denoise_2=1.0,

            krea2_rebalance=1.0,

            use_fls=False,fovea_strength=3.0,sharpness=0.5,mask_inertia=0.85,
            
            start_step=0,end_step=10000,            
            scale_by_1=1.0,  
            scale_by_2=1.0,  
            upscale_method_1="nearest-exact",                            
            upscale_method_2="nearest-exact",

            use_tiled=False,tile_size=512,overlap=64,temporal_size=64,temporal_overlap=8,

            use_edit_mode=False,
            use_controlnet=False,
            image_1=None,image_2=None,
            image_3=None,image_4=None,
            # image_CN=None,
            mask=None,
            control_model='None',control_strength=1,control_startPercent=0,control_endPercent=1,control_strength_model_2=0,
            layer_skip=0,
            lcm_sampling = 'lcm', lcm_zsnr=False,
            lora_stack=None,seed=0,
            auto_seed=False,
             **kwargs):        
        # run pipe in 
        afpipe_loras_in=None
        if lora_stack !=None:    
            # print(f' -- lora_stack.get(SIGNATURE):{lora_stack.get(SIGNATURE)}')        
            if not isinstance(lora_stack,dict) or not lora_stack.get(SIGNATURE):
                raise ValueError(f"Only supports the '{SIGNATURE}' type output by this plugin.")
        afpipe_loras_in = s.af.unpack_pipe_loras_stack(lora_stack)  
        local_data =s.af.get_cls_slots_vars2(s.INPUT_TYPES(),model_1,model_2,clip_1,clip_2,vae,lora_1,lora_2,control_model)

        def load_check_list_advance(check_dict,lora_extras):
            models_dict = check_dict[0].get('models')

            model_cls_1,model_cls_2,clip_cls_1,clip_cls_2,latent_output,vae_cls,sample_latent = None,None,None,None,None,None,None
            pos_output,neg_output = None,None
            clip_cls = None
            image_output,mask_output = None,None
            control_net_cls = None
            refimage1_outpout = image_1 if image_1 is not None else None
            refimage2_outpout = image_2 if image_2 is not None else None

            if len(models_dict) ==0:
                # raise RuntimeError('没有加载diffuse大模型') 
                out_slot = [local_data,
                   image_output,model_cls_1,model_cls_2,clip_cls,
                   pos_output,neg_output,latent_output,vae_cls,sample_latent,
                   control_net_cls]
                return out_slot                        

            model_list = list(models_dict.keys()) if len(models_dict.keys())>0 else []
            # print(f'-- model_list:{type(model_list)}')

            mod_type_1,mod_1 = s.af.model_load(models_dict.get('model_1'),weight_mode) if 'model_1' in model_list else ['',None]
            # print(f'-- mod_1 load:{type(mod_1)} - combot:{mod_type_1}')

            # load model and check layers
            if mod_1:
                match mod_type_1:
                    case 'model+clip':
                        model_cls_1 = mod_1[0]
                        clip_cls_1 = mod_1[1]
                    case 'model+vae':
                        model_cls_1 = mod_1[0]
                        vae_cls = mod_1[2]
                    case 'model+clip+vae':
                        model_cls_1 = mod_1[0]
                        clip_cls_1 = mod_1[1]
                        vae_cls = mod_1[2]
                    case _:
                        model_cls_1 = mod_1[0]
                if use_model_2 :
                    mod_type_2,mod_2 = s.af.model_load(models_dict.get('model_2'),weight_mode) if 'model_2' in model_list and use_model_2 else ['',None]
                    if mod_2:
                        match mod_type_2:
                            case 'model+clip':
                                model_cls_2 = mod_2[0]
                                clip_cls_2 = mod_2[1]
                            case 'model+vae':
                                model_cls_2 = mod_2[0]
                                vae_cls = mod_2[2]
                            case 'model+clip+vae':
                                model_cls_2 = mod_2[0]
                                clip_cls_2 = mod_2[1]
                                vae_cls = mod_2[2]
                            case _:
                                model_cls_2 = mod_2[0]
                # kv cache for flux2 klein kv
                if clip_type == 'flux2' and use_kvcache:
                    model_cls_1 = s.af.set_flux2_kvcache(model_cls_1)[0]


            # load clip
            if clip_cls_1 == None and clip_cls_2 == None:
                clip_1_pack = models_dict.get('clip_1') if 'clip_1' in model_list else None
                clip_2_pack = models_dict.get('clip_2') if 'clip_2' in model_list else None
                clip_cls = s.af.clip_load(clip_1_pack,clip_2_pack,clip_type) if clip_1_pack or clip_2_pack else None 
                # print(f' --clip_1_pack {clip_1_pack}') 
            else:
                # raise RuntimeWarning('model1已经包含了 clip')
                if clip_cls_2 == None:
                    clip_cls = clip_cls_1
                    print(f' -warning- model 1 已经包含了 clip')
                else:
                    clip_cls = clip_cls_2
                    print(f' -warning- model 2 已经包含了 clip')

            # load vae
            if vae_cls == None:
                vae_pack = models_dict.get('vae') if 'vae' in model_list else None
                vae_cls = s.af.vae_set(vae_pack)
                # print(f' --vae {vae_pack}')
            else:
                if use_model_2:
                    print(f' -warning- model 2 已经包含了 vae')
                else:
                    print(f' -warning- model 1 已经包含了 vae')

            # load lora
            lora_1_pack = models_dict.get('lora_1') if 'lora_1' in model_list else None
            if lora_1_pack:
                model_cls_1 = s.af.lora_set(model_cls_1,lora_1_pack,lora_1_strength)
            if use_model_2:
                lora_2_pack = models_dict.get('lora_2') if 'lora_2' in model_list else None
                if lora_2_pack:
                    model_cls_2 = s.af.lora_set(model_cls_2,lora_2_pack,lora_2_strength)

            # load lora extras
            if lora_extras and len(lora_extras)>0:
                for lora_pack in lora_extras:
                    lora_extras_pack = lora_pack[0]
                    lora_extras_strength = lora_pack[1] 
                    print(f' -- load loras stack: [{lora_extras_pack.get('input')}]')
                    model_cls_1 = s.af.lora_set(model_cls_1,lora_extras_pack,lora_extras_strength)
                    if use_model_2 and lora_2_pack:
                        model_cls_2 = s.af.lora_set(model_cls_2,lora_extras_pack,lora_extras_strength)

            # chroma
            if clip_type == 'chroma':
                model_cls_1,clip_cls = s.af.set_chroma(model_cls_1,clip_cls,0,3)
                # print(f'-- run chroma:model:{model_cls_1} - clip:{clip_cls}')


            # lcm sampling
            if clip_type in ['sdxl','stable_diffusion','sd3']:
                model_cls_1 = s.af.set_lcm_for_sd_xl(model_cls_1,lcm_sampling,lcm_zsnr)
                # pass
            
            # skip layer
            if layer_skip <-1:
                clip_cls = s.af.set_layer_skip(clip_cls,layer_skip,clip_type)

            # input image mask size
            image_resize,mask_output,latent_width,latent_height = s.af.get_resolution(match_size,image_1,image_2,mask,
                                                         resolution,flip,aspectRatio,megapixels,longerEdge,solution_preset,width,height)
            # latent
            latent_output = s.af.latent_set(clip_type,image_resize,mask_output,latent_width,latent_height,batch,vae_cls,use_edit_mode)[0]

            # conditioning
            krea2_per_layer_weights = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"
            pos_output,neg_output = s.af.conditioning_set(clip_cls,pos,neg,clip_type,neg_zero,fluxguidance,krea2_rebalance,krea2_per_layer_weights)

            # # ref image to eidt model or use controlnet model
            if use_edit_mode and not use_controlnet:
                if clip_type == 'flux2':
                    pos_list,neg_list = [],[]
                    input_image = [image_1,image_2,image_3,image_4]
                    run_image_dict,image_num = {},0
                    for i in input_image:
                        if i is not None:
                            run_image_dict[str(image_num)]=i
                            image_num+=1
                    # print(f'========= run_image_num:{len(run_image_dict)} - {run_image_dict.keys()}')
                    if len(run_image_dict) > 0:
                        pos_current = pos_output
                        neg_current = neg_output
                        pos_list = []
                        neg_list = []
                        for i in range(len(run_image_dict)):
                            pos_current = s.af.ref_encode_to_conditioning_for_flux2(vae_cls,run_image_dict[str(i)],pos_current)
                            neg_current = s.af.ref_encode_to_conditioning_for_flux2(vae_cls,run_image_dict[str(i)],neg_current)
                            pos_list.append(pos_current)
                            neg_list.append(neg_current)
                        pos_output = pos_list[-1]
                        neg_output = neg_list[-1]
                elif clip_type == 'qwen_image':
                    pos_output,neg_output,latent_output,main_image_qwen,mask_qwen = s.af.ref_qwen_edit_from_Qweneditutils(clip_cls,pos,vae_cls,
                        image_1,image_2,image_3, image_4,
                        mask,
                        latent_width,latent_height)
                    if mask is not None:
                        mask_output = mask_qwen[0]

            else:
                # controlnet normal
                control_pack = models_dict.get('control_model') if 'control_model' in model_list else None
                ctl_dir,ctl_input = '',''
                if control_pack:
                    ctl_dir = control_pack['dir']
                    ctl_format = control_pack['format']
                    ctl_input = control_pack['input']
                    ctl_path = control_pack['path']
                
                # controlnet model normal
                if str(ctl_dir).lower() == 'controlnet':                
                    control_net_cls = s.af.load_controlnet_normal(ctl_input)
                    if not use_edit_mode and use_controlnet: 
                        if image_2 is None:
                            raise ValueError('controlnet model patch 模式需要 image_2 输入')                      
                        pos_output,neg_output = s.af.set_applycontrolnet_normal(control_net_cls,vae_cls,
                            image_2,control_strength,control_startPercent,control_endPercent,pos_output,neg_output)
                        if use_model_2 and model_cls_2:
                            pos_output,neg_output = s.af.set_applycontrolnet_normal(control_net_cls,vae_cls,
                            image_2,control_strength_model_2,control_startPercent,control_endPercent,pos_output,neg_output)
                # controlnet model patch
                elif str(ctl_dir).lower() == 'model_patches': 
                    control_net_cls = s.af.load_controlnet_model_patch(ctl_input)
                    if not use_edit_mode and use_controlnet:
                        if image_2 is None:
                            raise ValueError('controlnet model patch 模式需要 image_2 输入')
                        model_cls_1 = s.af.set_applycontrolnet_model_patch(control_net_cls,model_cls_1,vae_cls,
                            image_2,control_strength,image_3,mask)
                        if use_model_2 and model_cls_2:
                            model_cls_2 = s.af.set_applycontrolnet_model_patch(control_net_cls,model_cls_2,vae_cls,
                            image_2,control_strength_model_2,image_3,mask)
                                        
                model_cls_1 = s.af.set_DifferentialDiffusion(model_cls_1,1.0)
                if control_strength_model_2>0:
                    model_cls_2 = s.af.set_DifferentialDiffusion(model_cls_2,1.0)

            # auraflow and cfgnorm, ModelSamplingFlux for flux
            if clip_type in ['qwen_image','lumina2','chroma']:
                if model_cls_1:
                    model_cls_1 = s.af.ModelSamplingAuraFlow(model_cls_1,shift)
                    if use_edit_mode:
                        model_cls_1 = s.af.CFGNorm(model_cls_1,cfgNorm) # CFGNorm与2512生图模型冲突,针对编辑模型,比如2511，firered，
                if model_cls_2 and use_model_2:
                    model_cls_2 = s.af.ModelSamplingAuraFlow(model_cls_2,shift)
                    if use_edit_mode:
                        model_cls_2 = s.af.CFGNorm(model_cls_2,cfgNorm)
            elif clip_type in ['flux']:
                if model_cls_1:
                    model_cls_1 = s.af.ModelSamplingFlux(model_cls_1,1.15,0.5,latent_width,latent_height)
                if model_cls_2 and use_model_2:
                    model_cls_2 = s.af.ModelSamplingFlux(model_cls_2,1.15,0.5,latent_width,latent_height)

            # ksampler     
            k_sample = s.af.ksample_set(clip_type,ksample_type,model_cls_1,
                seed, pos_output,neg_output,
                latent_output,latent_width,latent_height,
                steps_1,cfg_1,sampler_name_1,scheduler_1,denoise_1,
                steps_2,cfg_2,sampler_name_2,scheduler_2,denoise_2,
                use_fls,fovea_strength,sharpness,mask_inertia,
                start_step,end_step,
                use_model_2,model_cls_2,
                scale_by_1, scale_by_2,
                upscale_method_1,upscale_method_2,
                # add_noise_1,return_with_leftover_noise_1,
                # add_noise_2,return_with_leftover_noise_2,
                )
            sample_latent = k_sample[0]

            # # vae decode
            image_output = s.af.vaedecoder_set(vae_cls,sample_latent,
                        use_tiled,tile_size,overlap,temporal_size,temporal_overlap)

            return [local_data,image_output,mask_output,
                   refimage1_outpout,refimage2_outpout,model_cls_1,model_cls_2,clip_cls,
                   pos_output,neg_output,latent_output,vae_cls,sample_latent,
                   control_net_cls]
        
        # for pipe unite ksampler
        output_image,ref_mask,ref_image1,ref_image2 = (t:=load_check_list_advance(local_data,afpipe_loras_in))[1],t[2],t[3],t[4]
        return (output_image,ref_mask,ref_image1,ref_image2)


class pipe_unite_loader:
    af = aftools()
    @classmethod
    def INPUT_TYPES(s):
        return{
            'required':{
                'model_1': (s.af.get_model_diffusion_list(),{'default': 'None'}),
            },
            'optional': {
                'lora_1':(s.af.get_model_lora_list(),{'default': 'None'}),
                'lora_1_strength': ('FLOAT', {'default': 1.0, 'min': 0, 'max':10.0, 'step': 0.01}),
                'clip_1': (s.af.get_model_clip_list(),{'default': 'None'}),
                'clip_2': (s.af.get_model_clip_list().copy()+[''],{'default': 'None'}),
                'vae': (s.af.get_model_vae_list(),{'default': 'None'}),             

                'weight_mode': (DTYPE_UNIFIED, {'default': 'default'}),
                'clip_type': (s.af.get_clip_type_list(),{'default': s.af.get_clip_type_list()[0]}),
                
                'use_edit_mode':('BOOLEAN', {'default': False},),
                'use_model_2':('BOOLEAN', {'default': False},),                    
                'model_2': (s.af.get_model_diffusion_list().copy()+[''],{'default': 'None'}), 
                'lora_2': (s.af.get_model_lora_list().copy()+[''],{'default': 'None'},),
                'lora_2_strength': ('FLOAT', {'default': 1.0, 'min': 0, 'max':10.0, 'step': 0.01}),
                
                'use_controlnet':('BOOLEAN', {'default': False},),
                'control_model': (s.af.get_model_controlnet_list(),{'default': 'None'},),                     
                'control_strength': ('FLOAT', {'default': 1.0, 'min': -10.0, 'max': 10.0, 'step': 0.01}), 
                'control_startPercent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}), 
                'control_endPercent': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}), 
                'control_strength_model_2': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'preset': (list(s.af.auto_updata_perset_label().keys()),{'default':list(s.af.auto_updata_perset_label().keys())[0]}), 

                'pos': ('STRING', {'multiline': True, 'dynamicPrompts': True,'default':''},),
                'neg': ('STRING', {'multiline': True,'dynamicPrompts': True,'default':''},),                    
                'neg_zero':('BOOLEAN', {'default': True},),

                'match_size':('BOOLEAN', {'default': False,'tooltip':'auto match resolution from ref image1'},),
                'resolution': (SOLUTJION_TYPE,{'default':SOLUTJION_TYPE[0]}),
                'aspectRatio': (ASPECTRATIO, {'default':ASPECTRATIO[1],}),
                'megapixels': ('FLOAT', {'default': 1.0, 'min': 0.1, 'max': 16.0, 'step':0.1,}),
                'longerEdge': ('INT', {'default': 1536, 'min': 64, 'max': 2560, 'step': 2}),
                'solution_preset':(s.af.get_resolution_preset(), {'default':s.af.get_resolution_preset()[0]}),
                'width': ('INT', {'default': 512, 'min': 64, 'max': 2560, 'step': 2}),
                'height': ('INT', {'default': 1024, 'min': 64, 'max': 2560, 'step': 2}),
                'flip':('BOOLEAN', {'default': False,'tooltip':'flip the width and height in resolution'},),
                
                'batch': ('INT', {'default': 1, 'min': 1, 'max': 10, 'step': 1}),
                'layer_skip': ('INT', {'default': -1, 'min': -24, 'max': -1, 'step': 1, 'advanced': True}),
                # sampling, zsnr
                "lcm_sampling": (["eps", "v_prediction", "lcm", "x0", "img_to_img", "img_to_img_flow"], {'default': "lcm",'tooltip': 'ModelSamplingDiscrete的采样方式,lcm是基于eps的lcm采样,mg_to_img_flow是基于v_prediction的图像到图像的流式采样'},),
                # 'lcm_zsnr': ('BOOLEAN', {'default': False, 'advanced': True}),

                'use_kvcache': ('BOOLEAN', {'default': False,'tooltip':'use kvcache for flux2 kv'},),
                'fluxguidance': ('FLOAT', {'default': 3.5, 'min': -100, 'max': 100, 'step': 0.01, 'tooltip': 'for flux.'}),
                'shift': ('FLOAT', {'default': 3.1, 'min': 0.0, 'max': 100.0, 'step':0.01}),
                'cfgNorm':('FLOAT',{'default':1.0, 'min':0.0, 'max':100.0, 'step':0.01}),

                # krea2
                'krea2_rebalance':('FLOAT',{'default':1.0, 'min':0.0, 'max':10.0, 'step':0.01,'tooltip': 'for nsfw.'}),
                # 'per_layer_weights':('STRING',{'default':"1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0","multiline": False}),

                'image_1':('IMAGE',),
                'image_2':('IMAGE',),
                'image_3':('IMAGE',),
                'image_4':('IMAGE',),
                'mask':('MASK',),
                'lora_stack': (PIPE_TYPE_LORA,),
            },
        }        
    RETURN_TYPES = ('MODEL','MODEL',
                'CONDITIONING','CONDITIONING','LATENT','VAE',
                'IMAGE','IMAGE',
                )
    RETURN_NAMES = ('model_1','model_2',
                'positive','negative','latent','vae',
                'refImage1','refImage2',
                )
    FUNCTION = 'run'
    CATEGORY = 'afar_tools'
    SEARCH_ALIASES = ['模型加载','Model Loader','Universal Loader','通用加载器','多模型加载','Multi Model Loader']

    def run(s,
            model_1,model_2,clip_1,clip_2='None',vae='None',use_model_2=False,
            weight_mode='default',clip_type='None',preset='',
            lora_1='None',lora_1_strength='None',lora_2='None',lora_2_strength='None',
            pos='',neg = '',neg_zero=True,

            fluxguidance=3.5,use_kvcache=False,shift=3.1,cfgNorm=1,

            krea2_rebalance=1.0,

            resolution=SOLUTJION_TYPE[0],width=1024,height=1024,flip=False,batch=1,match_size=False,
            solution_preset = af.get_resolution_preset()[0],
            aspectRatio='1:1',megapixels=1.0,longerEdge=1024,

            use_edit_mode=False,
            use_controlnet=False,
            image_1=None,image_2=None,
            image_3=None,image_4=None,
            mask=None,
            control_model='None',control_strength=1,control_startPercent=0,control_endPercent=1,control_strength_model_2=0,
            layer_skip=0,
            lcm_sampling = 'lcm', lcm_zsnr=False,
            lora_stack=None,
             **kwargs):        
        # run pipe in 
        afpipe_loras_in=None
        if lora_stack !=None:    
            # print(f' -- lora_stack.get(SIGNATURE):{lora_stack.get(SIGNATURE)}')        
            if not isinstance(lora_stack,dict) or not lora_stack.get(SIGNATURE):
                raise ValueError(f"Only supports the '{SIGNATURE}' type output by this plugin.")
        afpipe_loras_in = s.af.unpack_pipe_loras_stack(lora_stack)  
        local_data =s.af.get_cls_slots_vars2(s.INPUT_TYPES(),model_1,model_2,clip_1,clip_2,vae,lora_1,lora_2,control_model)

        def load_check_list_advance(check_dict,lora_extras):
            models_dict = check_dict[0].get('models')

            model_cls_1,model_cls_2,clip_cls_1,clip_cls_2,latent_output,vae_cls,sample_latent = None,None,None,None,None,None,None
            pos_output,neg_output = None,None
            clip_cls = None
            image_output,mask_output = None,None
            control_net_cls = None
            refimage1_outpout = image_1 if image_1 is not None else None
            refimage2_outpout = image_2 if image_2 is not None else None

            if len(models_dict) ==0:
                # raise RuntimeError('没有加载diffuse大模型') 
                out_slot = [local_data,
                   image_output,model_cls_1,model_cls_2,clip_cls,
                   pos_output,neg_output,latent_output,vae_cls,sample_latent,
                   control_net_cls]
                return out_slot                        

            model_list = list(models_dict.keys()) if len(models_dict.keys())>0 else []
            # print(f'-- model_list:{type(model_list)}')

            mod_type_1,mod_1 = s.af.model_load(models_dict.get('model_1'),weight_mode) if 'model_1' in model_list else ['',None]
            # print(f'-- mod_1 load:{type(mod_1)} - combot:{mod_type_1}')

            # load model and check layers
            if mod_1:
                match mod_type_1:
                    case 'model+clip':
                        model_cls_1 = mod_1[0]
                        clip_cls_1 = mod_1[1]
                    case 'model+vae':
                        model_cls_1 = mod_1[0]
                        vae_cls = mod_1[2]
                    case 'model+clip+vae':
                        model_cls_1 = mod_1[0]
                        clip_cls_1 = mod_1[1]
                        vae_cls = mod_1[2]
                    case _:
                        model_cls_1 = mod_1[0]
                if use_model_2 :
                    mod_type_2,mod_2 = s.af.model_load(models_dict.get('model_2'),weight_mode) if 'model_2' in model_list and use_model_2 else ['',None]
                    if mod_2:
                        match mod_type_2:
                            case 'model+clip':
                                model_cls_2 = mod_2[0]
                                clip_cls_2 = mod_2[1]
                            case 'model+vae':
                                model_cls_2 = mod_2[0]
                                vae_cls = mod_2[2]
                            case 'model+clip+vae':
                                model_cls_2 = mod_2[0]
                                clip_cls_2 = mod_2[1]
                                vae_cls = mod_2[2]
                            case _:
                                model_cls_2 = mod_2[0]
                # kv cache for flux2 klein kv
                if clip_type == 'flux2' and use_kvcache:
                    model_cls_1 = s.af.set_flux2_kvcache(model_cls_1)[0]


            # load clip
            if clip_cls_1 == None and clip_cls_2 == None:
                clip_1_pack = models_dict.get('clip_1') if 'clip_1' in model_list else None
                clip_2_pack = models_dict.get('clip_2') if 'clip_2' in model_list else None
                clip_cls = s.af.clip_load(clip_1_pack,clip_2_pack,clip_type) if clip_1_pack or clip_2_pack else None 
                # print(f' --clip_1_pack {clip_1_pack}') 
            else:
                # raise RuntimeWarning('model1已经包含了 clip')
                if clip_cls_2 == None:
                    clip_cls = clip_cls_1
                    print(f' -warning- model 1 已经包含了 clip')
                else:
                    clip_cls = clip_cls_2
                    print(f' -warning- model 2 已经包含了 clip')

            # load vae
            if vae_cls == None:
                vae_pack = models_dict.get('vae') if 'vae' in model_list else None
                vae_cls = s.af.vae_set(vae_pack)
                # print(f' --vae {vae_pack}')
            else:
                # raise RuntimeWarning('model1已经包含了 vae')
                if use_model_2:
                    print(f' -warning- model 2 已经包含了 vae')
                else:
                    print(f' -warning- model 1 已经包含了 vae')

            # load lora
            lora_1_pack = models_dict.get('lora_1') if 'lora_1' in model_list else None
            if lora_1_pack:
                model_cls_1 = s.af.lora_set(model_cls_1,lora_1_pack,lora_1_strength)
            if use_model_2:
                lora_2_pack = models_dict.get('lora_2') if 'lora_2' in model_list else None
                if lora_2_pack:
                    model_cls_2 = s.af.lora_set(model_cls_2,lora_2_pack,lora_2_strength)

            # load lora extras
            if lora_extras and len(lora_extras)>0:
                for lora_pack in lora_extras:
                    lora_extras_pack = lora_pack[0]
                    lora_extras_strength = lora_pack[1] 
                    print(f' -- load loras stack: [{lora_extras_pack.get('input')}]')
                    model_cls_1 = s.af.lora_set(model_cls_1,lora_extras_pack,lora_extras_strength)
                    if use_model_2 and lora_2_pack:
                        model_cls_2 = s.af.lora_set(model_cls_2,lora_extras_pack,lora_extras_strength)

            # chroma
            if clip_type == 'chroma':
                model_cls_1,clip_cls = s.af.set_chroma(model_cls_1,clip_cls,0,3)
                # print(f'-- run chroma:model:{model_cls_1} - clip:{clip_cls}')

            # lcm sampling
            if clip_type in ['sdxl','stable_diffusion','sd3']:
                model_cls_1 = s.af.set_lcm_for_sd_xl(model_cls_1,lcm_sampling,False)
                # pass
            
            # skip layer
            if layer_skip <-1:
                clip_cls = s.af.set_layer_skip(clip_cls,layer_skip,clip_type)

            # input image mask size
            image_resize,mask_output,latent_width,latent_height = s.af.get_resolution(match_size,image_1,image_2,mask,
                                                         resolution,flip,aspectRatio,megapixels,longerEdge,solution_preset,width,height)
            # latent
            latent_output = s.af.latent_set(clip_type,image_resize,mask_output,latent_width,latent_height,batch,vae_cls,use_edit_mode)[0]

            # conditioning
            krea2_per_layer_weights = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"
            pos_output,neg_output = s.af.conditioning_set(clip_cls,pos,neg,clip_type,neg_zero,fluxguidance,krea2_rebalance,krea2_per_layer_weights)

            # # ref image to eidt model or use controlnet model
            if use_edit_mode and not use_controlnet:
                if clip_type == 'flux2':
                    pos_list,neg_list = [],[]
                    input_image = [image_1,image_2,image_3,image_4]
                    run_image_dict,image_num = {},0
                    for i in input_image:
                        if i is not None:
                            run_image_dict[str(image_num)]=i
                            image_num+=1
                    # print(f'========= run_image_num:{len(run_image_dict)} - {run_image_dict.keys()}')
                    if len(run_image_dict) > 0:
                        pos_current = pos_output
                        neg_current = neg_output
                        pos_list = []
                        neg_list = []
                        for i in range(len(run_image_dict)):
                            pos_current = s.af.ref_encode_to_conditioning_for_flux2(vae_cls,run_image_dict[str(i)],pos_current)
                            neg_current = s.af.ref_encode_to_conditioning_for_flux2(vae_cls,run_image_dict[str(i)],neg_current)
                            pos_list.append(pos_current)
                            neg_list.append(neg_current)
                        pos_output = pos_list[-1]
                        neg_output = neg_list[-1]
                elif clip_type == 'qwen_image':
                    pos_output,neg_output,latent_output,main_image_qwen,mask_qwen = s.af.ref_qwen_edit_from_Qweneditutils(clip_cls,pos,vae_cls,
                        image_1,image_2,image_3,image_4,
                        mask,
                        latent_width,latent_height)
                    if mask is not None:
                        mask_output = mask_qwen[0]

            else:
                # controlnet normal
                control_pack = models_dict.get('control_model') if 'control_model' in model_list else None
                ctl_dir,ctl_input = '',''
                if control_pack:
                    ctl_dir = control_pack['dir']
                    ctl_format = control_pack['format']
                    ctl_input = control_pack['input']
                    ctl_path = control_pack['path']
                
                # controlnet model normal
                if str(ctl_dir).lower() == 'controlnet':                
                    control_net_cls = s.af.load_controlnet_normal(ctl_input)
                    if not use_edit_mode and use_controlnet: 
                        if image_2 is None:
                            raise ValueError('controlnet model patch 模式需要 image_2 输入')                      
                        pos_output,neg_output = s.af.set_applycontrolnet_normal(control_net_cls,vae_cls,
                            image_2,control_strength,control_startPercent,control_endPercent,pos_output,neg_output)
                        if use_model_2 and model_cls_2:
                            pos_output,neg_output = s.af.set_applycontrolnet_normal(control_net_cls,vae_cls,
                            image_2,control_strength_model_2,control_startPercent,control_endPercent,pos_output,neg_output)
                # controlnet model patch
                elif str(ctl_dir).lower() == 'model_patches': 
                    control_net_cls = s.af.load_controlnet_model_patch(ctl_input)
                    if not use_edit_mode and use_controlnet:
                        if image_2 is None:
                            raise ValueError('controlnet model patch 模式需要 image_2 输入')
                        model_cls_1 = s.af.set_applycontrolnet_model_patch(control_net_cls,model_cls_1,vae_cls,
                            image_2,control_strength,image_3,mask)
                        if use_model_2 and model_cls_2:
                            model_cls_2 = s.af.set_applycontrolnet_model_patch(control_net_cls,model_cls_2,vae_cls,
                            image_2,control_strength_model_2,image_3,mask)
                                        
                model_cls_1 = s.af.set_DifferentialDiffusion(model_cls_1,1.0)
                if control_strength_model_2>0:
                    model_cls_2 = s.af.set_DifferentialDiffusion(model_cls_2,1.0)

            # auraflow and cfgnorm, ModelSamplingFlux for flux
            if clip_type in ['qwen_image','lumina2','chroma']:
                if model_cls_1:
                    model_cls_1 = s.af.ModelSamplingAuraFlow(model_cls_1,shift)
                    if use_edit_mode:
                        model_cls_1 = s.af.CFGNorm(model_cls_1,cfgNorm) # CFGNorm与2512生图模型冲突,针对编辑模型,比如2511，firered，
                if model_cls_2 and use_model_2:
                    model_cls_2 = s.af.ModelSamplingAuraFlow(model_cls_2,shift)
                    if use_edit_mode:
                        model_cls_2 = s.af.CFGNorm(model_cls_2,cfgNorm)
            elif clip_type in ['flux']:
                if model_cls_1:
                    model_cls_1 = s.af.ModelSamplingFlux(model_cls_1,1.15,0.5,latent_width,latent_height)
                if model_cls_2 and use_model_2:
                    model_cls_2 = s.af.ModelSamplingFlux(model_cls_2,1.15,0.5,latent_width,latent_height)


            return [local_data,image_output,mask_output,
                   refimage1_outpout,refimage2_outpout,model_cls_1,model_cls_2,clip_cls,
                   pos_output,neg_output,latent_output,vae_cls,sample_latent,
                   control_net_cls]
        
        # for pipe unite loader
        t=load_check_list_advance(local_data,afpipe_loras_in)
        return (t[5],t[6],t[8],t[9],t[10],t[11],t[3],t[4])


