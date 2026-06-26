import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "aftools.imagetools.js",
        
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // console.log("正在注册节点:", nodeData.name); 


        if (nodeData.name === "ImageMask_Pad_Resize") {  
            const onNodeCreated = nodeType.prototype.onNodeCreated;        
            nodeType.prototype.onNodeCreated = function() { 
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                // UI
                const setWidgetDisabled = (widgets_map,disabled,h_disabled) =>{
                    widgets_map.forEach(w=>{
                        if (w) {
                            w.disabled = disabled
                            w.hidden = h_disabled
                        }
                    });
                    this.setSize([this.size[0], this.computeSize()[1]]);
                    this.setDirtyCanvas(true, true);
                }
                // resolution                
                const scaleMode_Widget = this.widgets.find(w => w.name === "scaleMode");
                const res_flip_Widget = this.widgets.find(w => w.name === "flip");
                const res_megapixels_Widget = this.widgets.find(w => w.name === "megapixels");
                const res_shortEdge_Widget = this.widgets.find(w => w.name === "shortEdge");
                const res_longerEdge_Widget = this.widgets.find(w => w.name === "longerEdge");
                const res_preset_Widget = this.widgets.find(w => w.name === "preset");
                const res_width_Widget = this.widgets.find(w => w.name === "width");
                const res_height_Widget = this.widgets.find(w => w.name === "height");

                const resolutionSwitchsMap_all = [res_megapixels_Widget,res_shortEdge_Widget,res_longerEdge_Widget,res_preset_Widget,res_width_Widget,res_height_Widget]
                const resolutionSwitchsMap_totalPixels = [res_megapixels_Widget]
                const resolutionSwitchsMap_shortEdge = [res_shortEdge_Widget]
                const resolutionSwitchsMap_longerEdge = [res_longerEdge_Widget]
                const resolutionSwitchsMap_preset = [res_preset_Widget]
                const resolutionSwitchsMap_custom = [res_width_Widget, res_height_Widget]
                
                const toggle_scaleMode_Visibility = (v) => {
                    setWidgetDisabled(resolutionSwitchsMap_all,true,true)  
                    if (v === 'totalPixels') {
                        setWidgetDisabled(resolutionSwitchsMap_totalPixels,false,false)  
                    }else if (v === 'shortEdge') {  
                        setWidgetDisabled(resolutionSwitchsMap_shortEdge,false,false)  
                    }else if (v === 'longerEdge') {  
                        setWidgetDisabled(resolutionSwitchsMap_longerEdge,false,false)  
                    }else if (v === 'preset') {  
                        setWidgetDisabled(resolutionSwitchsMap_preset,false,false) 
                    }else if (v === 'custom') {
                        setWidgetDisabled(resolutionSwitchsMap_custom,false,false) 
                    }
                };
                const originalScaleModeCallback = scaleMode_Widget.callback;
                scaleMode_Widget.callback = (value) => {
                    originalScaleModeCallback?.(value);
                    toggle_scaleMode_Visibility(value);
                };                    
                toggle_scaleMode_Visibility(scaleMode_Widget.value);

            }
        }

        if (nodeData.name === "CropByMask_Resize"){
            const Crop_onNodeCreated = nodeType.prototype.onNodeCreated;        
            nodeType.prototype.onNodeCreated = function() { 
                if (Crop_onNodeCreated) Crop_onNodeCreated.apply(this, arguments);
                
                // UI
                const setWidgetDisabled = (widgets_map,disabled,h_disabled) =>{
                    widgets_map.forEach(w=>{
                        if (w) {
                            w.disabled = disabled
                            w.hidden = h_disabled
                        }
                    });
                    this.setSize([this.size[0], this.computeSize()[1]]);
                    this.setDirtyCanvas(true, true);
                }


                // resolution                
                const scaleMode_Widget = this.widgets.find(w => w.name === "scaleMode");
                const res_flip_Widget = this.widgets.find(w => w.name === "flip");
                const res_megapixels_Widget = this.widgets.find(w => w.name === "megapixels");
                const res_shortEdge_Widget = this.widgets.find(w => w.name === "shortEdge");
                const res_longerEdge_Widget = this.widgets.find(w => w.name === "longerEdge");

                const resolutionSwitchsMap_all = [res_megapixels_Widget,res_shortEdge_Widget,res_longerEdge_Widget]
                const resolutionSwitchsMap_totalPixels = [res_megapixels_Widget]
                const resolutionSwitchsMap_shortEdge = [res_shortEdge_Widget]
                const resolutionSwitchsMap_longerEdge = [res_longerEdge_Widget]
                
                const toggle_scaleMode_Visibility = (v) => {
                    setWidgetDisabled(resolutionSwitchsMap_all,true,true)  
                    if (v === 'totalPixels') {
                        setWidgetDisabled(resolutionSwitchsMap_totalPixels,false,false)  
                    }else if (v === 'shortEdge') {  
                        setWidgetDisabled(resolutionSwitchsMap_shortEdge,false,false)  
                    }else if (v === 'longerEdge') {  
                        setWidgetDisabled(resolutionSwitchsMap_longerEdge,false,false)  
                    }
                };
                const originalScaleModeCallback = scaleMode_Widget.callback;
                scaleMode_Widget.callback = (value) => {
                    originalScaleModeCallback?.(value);
                    toggle_scaleMode_Visibility(value);
                };                    
                toggle_scaleMode_Visibility(scaleMode_Widget.value);


                // removeBG                
                const removeBG_Widget = this.widgets.find(w => w.name === "removeBG");
                const bg_type_Widget = this.widgets.find(w => w.name === "bg_type");
                const bg_color_Widget = this.widgets.find(w => w.name === "bg_color");

                const removeBG_map = [bg_type_Widget,bg_color_Widget]
                
                if (removeBG_Widget.value === false){
                     setWidgetDisabled(removeBG_map,true,true)  
                }

                const toggle_removeBG_Visibility = (v) => {
                    if (v === false) setWidgetDisabled(removeBG_map,true,true) 
                    else if (v === true) setWidgetDisabled(removeBG_map,false,false) 
                };
                const originalRemoveBGCallback = removeBG_Widget.callback;
                removeBG_Widget.callback = (value) => {
                    originalRemoveBGCallback?.(value);
                    toggle_removeBG_Visibility(value);
                };                    
                toggle_removeBG_Visibility(removeBG_Widget.value);

            }
        }

        if (nodeData.name === "CropByMask_Resize_sam3"){
            const Crop_onNodeCreated = nodeType.prototype.onNodeCreated;        
            nodeType.prototype.onNodeCreated = function() { 
                if (Crop_onNodeCreated) Crop_onNodeCreated.apply(this, arguments);

                // UI
                const setWidgetDisabled = (widgets_map,disabled,h_disabled) =>{
                    widgets_map.forEach(w=>{
                        if (w) {
                            w.disabled = disabled
                            w.hidden = h_disabled
                        }
                    });
                    this.setSize([this.size[0], this.computeSize()[1]]);
                    this.setDirtyCanvas(true, true);
                }


                // resolution                
                const scaleMode_Widget = this.widgets.find(w => w.name === "scaleMode");
                const res_flip_Widget = this.widgets.find(w => w.name === "flip");
                const res_megapixels_Widget = this.widgets.find(w => w.name === "megapixels");
                const res_shortEdge_Widget = this.widgets.find(w => w.name === "shortEdge");
                const res_longerEdge_Widget = this.widgets.find(w => w.name === "longerEdge");

                const resolutionSwitchsMap_all = [res_megapixels_Widget,res_shortEdge_Widget,res_longerEdge_Widget]
                const resolutionSwitchsMap_totalPixels = [res_megapixels_Widget]
                const resolutionSwitchsMap_shortEdge = [res_shortEdge_Widget]
                const resolutionSwitchsMap_longerEdge = [res_longerEdge_Widget]
                
                const toggle_scaleMode_Visibility = (v) => {
                    setWidgetDisabled(resolutionSwitchsMap_all,true,true)  
                    if (v === 'totalPixels') {
                        setWidgetDisabled(resolutionSwitchsMap_totalPixels,false,false)  
                    }else if (v === 'shortEdge') {  
                        setWidgetDisabled(resolutionSwitchsMap_shortEdge,false,false)  
                    }else if (v === 'longerEdge') {  
                        setWidgetDisabled(resolutionSwitchsMap_longerEdge,false,false)  
                    }
                };
                const originalScaleModeCallback = scaleMode_Widget.callback;
                scaleMode_Widget.callback = (value) => {
                    originalScaleModeCallback?.(value);
                    toggle_scaleMode_Visibility(value);
                };                    
                toggle_scaleMode_Visibility(scaleMode_Widget.value);


                // sam3                
                const prompt_Widget = this.widgets.find(w => w.name === "prompt");
                const threshold_Widget = this.widgets.find(w => w.name === "threshold");
                const refine_iterations_Widget = this.widgets.find(w => w.name === "refine_iterations");
                const individual_masks_Widget = this.widgets.find(w => w.name === "individual_masks");
                const sam_model_name_Widget = this.widgets.find(w => w.name === "sam_model_name");

                const sam3_map = [threshold_Widget,refine_iterations_Widget,individual_masks_Widget,sam_model_name_Widget]
                
                if (prompt_Widget.value === ""){
                     setWidgetDisabled(sam3_map,true,true)  
                }

                const toggle_prompt_Visibility = (v) => {
                    if (v === "" || v.length === 0) setWidgetDisabled(sam3_map,true,true) 
                    else setWidgetDisabled(sam3_map,false,false) 
                };
                const originalPromptCallback = prompt_Widget.callback;
                prompt_Widget.callback = (value) => {
                    originalPromptCallback?.(value);
                    toggle_prompt_Visibility(value);
                };                    
                toggle_prompt_Visibility(prompt_Widget.value);

                // removeBG                
                const removeBG_Widget = this.widgets.find(w => w.name === "removeBG");
                const bg_type_Widget = this.widgets.find(w => w.name === "bg_type");
                const bg_color_Widget = this.widgets.find(w => w.name === "bg_color");

                const removeBG_map = [bg_type_Widget,bg_color_Widget]
                
                if (removeBG_Widget.value === false){
                     setWidgetDisabled(removeBG_map,true,true)  
                }

                const toggle_removeBG_Visibility = (v) => {
                    if (v === false) setWidgetDisabled(removeBG_map,true,true) 
                    else if (v === true) setWidgetDisabled(removeBG_map,false,false) 
                };
                const originalRemoveBGCallback = removeBG_Widget.callback;
                removeBG_Widget.callback = (value) => {
                    originalRemoveBGCallback?.(value);
                    toggle_removeBG_Visibility(value);
                };                    
                toggle_removeBG_Visibility(removeBG_Widget.value);

            }
        }
    }
})