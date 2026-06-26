import comfy,torch
import node_helpers
from comfy_api.latest import ComfyExtension, io
import math

class TextEncodeQwenImageEditPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEditPlus",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Image.Input("image4", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None, image4=None) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3, image4]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return io.NodeOutput(conditioning)
    




def get_system_prompt(instruction):
    template_prefix = "<|im_start|>system\n"
    template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    instruction_content = ""
    if instruction == "":
        instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
    else:
        # for handling mis use of instruction
        if template_prefix in instruction:
            # remove prefix from instruction
            instruction = instruction.split(template_prefix)[1]
        if template_suffix in instruction:
            # remove suffix from instruction
            instruction = instruction.split(template_suffix)[0]
        if "{}" in instruction:
            # remove {} from instruction
            instruction = instruction.replace("{}", "")
        instruction_content = instruction
    llama_template = template_prefix + instruction_content + template_suffix
    
    return llama_template






class EditTextEncode_EditUtils:
    # upscale_methods = ["lanczos", "bicubic", "area"]
    # crop_methods = ["pad", "center", "disabled"]
    # example_config = {
    #     "image": None,
    #     # ref part
    #     "to_ref": True,
    #     "ref_main_image": True,
    #     "ref_longest_edge": 1024,
    #     "ref_crop": "center", #"pad" for main image, "center", "disabled"
    #     "ref_upscale": "lanczos",
    #     # vl part
    #     "to_vl": True,
    #     "vl_resize": True,
    #     "vl_target_size": 384,
    #     "vl_crop": "center",
    #     "vl_upscale": "bicubic", #to scale image down, "bicubic", "area" might better than "lanczos"
    # }
    # example_output = {
    #     "pad_info": pad_info,
    #     "noise_mask": noise_mask,
    #     "full_refs_cond": conditioning,
    #     "main_ref_cond": conditioning_only_with_main_ref,
    #     "main_image": main_image,
    #     "vae_images": vae_images,
    #     "ref_latents": ref_latents,
    #     "vl_images": vl_images,
    #     "full_prompt": full_prompt,
    #     "llama_template": llama_template
    # }
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "model_config": ("DICT", {"default": None}),
                "configs": ("LIST", {"default": None})
            },
            # "optional": 
            # {
            #     # "return_full_refs_cond": ("BOOLEAN", {"default": True}),
            #     # "set_noise_mask": ("BOOLEAN", {"default": False, "tooltip": "Only useful when using ref_crop == pad. It would automatically mask out the padding area."}),
            #     "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),   
            # }
        }
    RETURN_TYPES = ("CONDITIONING", "LATENT", "ANY", "IMAGE", "MASK", "ANY")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask", "pad_info")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"
    def encode(self, clip, vae, prompt, 
               model_config=None,
               configs=None,
            #    return_full_refs_cond=True,
            #    set_noise_mask=False,
            #    instruction="",
        ):
        # print("len(configs)")
        # print(len(configs))
        # llama_template = get_system_prompt(instruction)
        model_name = model_config["model_name"] if "model_name" in model_config else None
        is_qwen = model_name == "qwen"
        vae_unit = model_config["vae_unit"] if "vae_unit" in model_config else 8
        llama_template = model_config["llama_template"] if "llama_template" in model_config else ""
        image_prompt = ""
        
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 1.0,
        }
        # print("len(configs)", len(configs))
        # check len(configs)
        assert len(configs) > 0, "No image provided"
        
        main_image_index = -1
        for i, image_obj in enumerate(configs):
            if image_obj["to_ref"]:
                if main_image_index == -1 and image_obj["ref_main_image"]:
                    main_image_index = i
                    continue
                # ensure only one main image
                if main_image_index != -1:
                    image_obj["ref_main_image"] = False
        if main_image_index == -1:
            print("\n Auto fixing main_image_index to the first image index")
            main_image_index = 0
        
        ref_latents = []
        vae_images = []
        vl_images = []
        
        noise_mask = None
        for i, image_obj in enumerate(configs):
            assert "image" in image_obj, "Image is missing"
            image = image_obj["image"]
            to_ref = image_obj["to_ref"]
            ref_main_image = image_obj["ref_main_image"]
            # ref_width = image_obj["ref_width"]
            # ref_height = image_obj["ref_height"]
            ref_longest_edge = image_obj["ref_longest_edge"]
            ref_crop = image_obj["ref_crop"]
            ref_upscale = image_obj["ref_upscale"]
            
            if is_qwen:
                to_vl = image_obj["to_vl"]
                vl_resize = image_obj["vl_resize"]
                vl_target_size = image_obj["vl_target_size"]
                vl_crop = image_obj["vl_crop"]
                vl_upscale = image_obj["vl_upscale"]
            else:
                to_vl = False
            
            mask = None
            if "mask" in image_obj:
                mask = image_obj["mask"]
            
            samples = image.movedim(-1, 1)
            if mask is not None:
                _, c, _, _ = samples.shape
                sample_masks = mask.unsqueeze(1).repeat(1, c, 1, 1)  # same shape
                # sample_masks = mask.movedim(-1, 1)
                # check samples and sample_masks should match
                print('samples.shape',samples.shape)
                print('sample_masks.shape',sample_masks.shape)
                assert samples.shape == sample_masks.shape, "Image and mask shape mismatch"
            
            if not to_ref and not to_vl:
                continue
            if to_ref:
                # print("ori_image.shape",samples.shape)
                # ori_height, ori_width = samples.shape[2:]      
                # 
                # fix_ori          
                ori_longest_edge = max(samples.shape[2], samples.shape[3])
                scale_by = ori_longest_edge / ref_longest_edge
                scaled_height = int(round(samples.shape[2] / scale_by))
                scaled_width = int(round(samples.shape[3] / scale_by))
                
                # scaled_height = int(round(ref_height))
                # scaled_width = int(round(ref_width))

                # pad only apply to main image
                if ref_crop == "pad":
                    # print("In pad mode")
                    # print("scaled_width", scaled_width)
                    # print("scaled_height", scaled_height)
                    crop = "center"
                    
                    width_ceil = math.ceil(scaled_width / vae_unit)
                    # if scaled_width % vae_unit != 0:
                    #     width_ceil +=1
                    canvas_width = width_ceil * vae_unit
                    
                    height_ceil = math.ceil(scaled_height / vae_unit)
                    # if scaled_height % vae_unit != 0:
                    #     height_ceil +=1
                    canvas_height = height_ceil * vae_unit
                    # pad image to canvas size
                    canvas = torch.zeros(
                        (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                        dtype=samples.dtype,
                        device=samples.device
                    )
                    
                    resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, ref_upscale, crop)
                    
                    resized_width = resized_samples.shape[3]
                    resized_height = resized_samples.shape[2]
                    # set resized samples to canvas
                    # canvas[:, :, x_offset:resized_height, y_offset:resized_width] = resized_samples
                    canvas[:, :, :resized_height, :resized_width] = resized_samples
                    
                    
                    # if set_noise_mask:
                        # noise_mask = torch.zeros(canvas.shape, dtype=torch.bool, device=canvas.device)
                        # noise_mask[:, :, x_offset:resized_height, y_offset:resized_width] = 1.0
                        # print("noise_mask.shape", noise_mask.shape)
                        # noise_mask = noise_mask.movedim(1, -1)
                        # print("movedim noise_mask.shape", noise_mask.shape)
                    
                    # only return main image pad info
                    
                    current_total = (samples.shape[3] * samples.shape[2])
                    total = int(resized_width * resized_height)
                    scale_by = math.sqrt(total / current_total)
                    if ref_main_image:
                        pad_info = {
                            "x": 0,
                            "y": 0,
                            "width": canvas_width - resized_width,
                            "height": canvas_height - resized_height,
                            "scale_by": round(1 / scale_by, 3)
                        }
                    
                    # print("pad_info", pad_info)
                    s = canvas
                    
                    if mask is not None and ref_main_image:
                        mask_canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        
                        resized_sample_masks = comfy.utils.common_upscale(sample_masks, scaled_width, scaled_height, ref_upscale, crop)
                        mask_canvas[:, :, :resized_height, :resized_width] = resized_sample_masks
                        m =  mask_canvas
                        
                        # remove noise mask channel
                        noise_mask = m[:, :1, :, :].squeeze(1)
                        print("noise_mask.shape", noise_mask.shape)
                else:
                    crop = ref_crop
                    # handle pad method when not main image
                    if ref_crop == "pad":
                        crop = "center"
                    width = round(scaled_width / vae_unit) * vae_unit
                    height = round(scaled_height / vae_unit) * vae_unit
                    # print("width",width)
                    # print("height",height)
                    s = comfy.utils.common_upscale(samples, width, height, ref_upscale, crop)
                    
                    if mask is not None and ref_main_image:
                        m = comfy.utils.common_upscale(sample_masks, width, height, ref_upscale, crop)
                        # remove noise mask channel
                        noise_mask = m[:, :1, :, :].squeeze(1)
                        print("noise_mask.shape", noise_mask.shape)
                image = s.movedim(1, -1)
                ref_latents.append(vae.encode(image[:, :, :, :3]))
                vae_images.append(image)
            if to_vl:
                if vl_resize:
                    # print("vl_resize")
                    total = int(vl_target_size * vl_target_size)
                else:
                    total = int(samples.shape[3] * samples.shape[2])
                    if total > 2048 * 2048:
                        print("vl_target_size too large, clipping to 2048")
                        total = 2048 * 2048
                current_total = (samples.shape[3] * samples.shape[2])
                scale_by = math.sqrt(total / current_total)
            
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                s = comfy.utils.common_upscale(samples, width, height, vl_upscale, vl_crop)
                
                image = s.movedim(1, -1)
                # handle non resize vl images
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                vl_images.append(image)
                
        full_prompt = image_prompt + prompt
        # print("full_prompt", full_prompt)
        # print("llama_template", llama_template)
        # if is_qwen:
        #     tokens = clip.tokenize(full_prompt, images=vl_images, llama_template=llama_template)
        # else:
        # print("editutils image_prompt", image_prompt)
        # print("editutils prompt", prompt)
        # print("editutils llama_template", llama_template)
        if llama_template == "" or llama_template is None:
            tokens = clip.tokenize(full_prompt, images=vl_images)
        else:
            tokens = clip.tokenize(full_prompt, images=vl_images, llama_template=llama_template)
        # print("editutils tokens", tokens)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        samples = torch.zeros(1, 4, 128, 128)
        # conditioning_only_with_main_ref = None
        if len(ref_latents) > 0:
            # conditioning_only_with_main_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents[main_image_index]]}, append=True)
            conditioning_full_refs = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            # print("editutils conditioning_full_refs before", conditioning_full_refs)
            samples = ref_latents[main_image_index]
            # conditioning_full_refs = node_helpers.conditioning_set_values(conditioning_full_refs, {"concat_latent_image": samples}, append=True)
            # print("editutils conditioning_full_refs after", conditioning_full_refs)
        latent_out = {"samples": samples}
        
        if noise_mask is not None:
            latent_out["noise_mask"] = noise_mask
        
        conditioning_output = conditioning_full_refs
        main_image = None
        if len(vae_images)>0:
            main_image = vae_images[main_image_index]
            
        
        
        custom_output = {
            "pad_info": pad_info,
            "full_refs_cond": conditioning_full_refs,
            # "main_ref_cond": conditioning_only_with_main_ref,
            "main_image": main_image,
            "vae_images": vae_images,
            "ref_latents": ref_latents,
            # "llama_template": llama_template,
            # "no_refs_cond": conditioning,
            "mask": noise_mask,
        }
        if is_qwen:
            custom_output["vl_images"] = vl_images
            custom_output["full_prompt"] = full_prompt
        
        
        # print("editutils conditioning_output", conditioning_output)
        return (conditioning_output, latent_out, custom_output, main_image, noise_mask, pad_info)
    






class QwenEditTextEncode_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                "image4": ("IMAGE", ),
                "ref_longest_edge": ("INT", {"default": 1024, "min": 8, "max": 4096, "step": 1, "tooltip": "Longest edge of the output latent"}),
                # 'ref_width':('INT',{'default':1024,"min": 8, "max": 4096, "step": 1,}),
                # 'ref_height':('INT',{'default':1024,"min": 8, "max": 4096, "step": 1,}),
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "ANY", "IMAGE", "MASK")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"
    
    def encode(self, clip, vae, prompt,
               image1=None, image2=None, image3=None,image4=None,
               mask=None,  # New mask parameter
            #    width=1024,height=1024):
               ref_longest_edge=1024):
        # Prepare model config
        model_config = {
            "model_name": "qwen",
            "instruction": "",
            "vae_unit": 8,
            "llama_template": get_system_prompt("")
        }
        
        # Prepare configs list
        configs = []
        
        # Process each image if provided
        if image1 is not None:
            
            config1 = {
                "image": image1,
                "to_ref": True,  # Default to True
                "ref_main_image": True,  # First image is main by default
                # "ref_width": width,
                # "ref_height": height,
                "ref_longest_edge": ref_longest_edge,
                "ref_crop": "pad",  # Default to pad
                # "ref_crop": "disabled",  # Default to pad
                "ref_upscale": "lanczos",  # Default to lanczos
                "to_vl": True,  # Default to True
                "vl_resize": True,  # Default to True
                "vl_target_size": 384,  # Default to 384
                "vl_crop": "center",  # Default to center
                "vl_upscale": "lanczos"  # Default to lanczos
            }
            # Set mask to image1 if provided
            if mask is not None:
                config1["mask"] = mask
            configs.append(config1)
        
        if image2 is not None:
            config2 = {
                "image": image2,
                "to_ref": True,  # Default to True
                "ref_main_image": False,  # Only first image is main                
                # "ref_width": width,
                # "ref_height": height,
                "ref_longest_edge": ref_longest_edge,
                "ref_crop": "pad",  # Default to pad
                "ref_upscale": "lanczos",  # Default to lanczos
                "to_vl": True,  # Default to True
                "vl_resize": True,  # Default to True
                "vl_target_size": 384,  # Default to 384
                "vl_crop": "center",  # Default to center
                "vl_upscale": "lanczos"  # Default to lanczos
            }
            configs.append(config2)
        
        if image3 is not None:
            config3 = {
                "image": image3,
                "to_ref": True,  # Default to True
                "ref_main_image": False,  # Only first image is main
                # "ref_width": width,
                # "ref_height": height,
                "ref_longest_edge": ref_longest_edge,
                "ref_crop": "pad",  # Default to pad
                "ref_upscale": "lanczos",  # Default to lanczos
                "to_vl": True,  # Default to True
                "vl_resize": True,  # Default to True
                "vl_target_size": 384,  # Default to 384
                "vl_crop": "center",  # Default to center
                "vl_upscale": "lanczos"  # Default to lanczos
            }
            configs.append(config3)

        if image4 is not None:
            config4 = {
                "image": image4,
                "to_ref": True,  # Default to True
                "ref_main_image": False,  # Only first image is main
                # "ref_width": width,
                # "ref_height": height,
                "ref_longest_edge": ref_longest_edge,
                "ref_crop": "pad",  # Default to pad
                "ref_upscale": "lanczos",  # Default to lanczos
                "to_vl": True,  # Default to True
                "vl_resize": True,  # Default to True
                "vl_target_size": 384,  # Default to 384
                "vl_crop": "center",  # Default to center
                "vl_upscale": "lanczos"  # Default to lanczos
            }
            configs.append(config4)
        
        if len(configs) == 0:
            raise ValueError("At least one image must be provided")
        
        # Call the original EditTextEncode function
        node_instance = EditTextEncode_EditUtils()
        return node_instance.encode(
            clip=clip,
            vae=vae,
            prompt=prompt,
            model_config=model_config,
            configs=configs
        )