import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "aftools.pipe_unite2.js",
        
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // console.log("正在注册节点:", nodeData.name); 

        if (nodeData.name === "pipe_unite_loader") {   
            // 必须挂载到 nodeType.prototype 上，才能实时响应画布上的操作
            // --- 调试增强版：监听连线变化 ---
            const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, slotIndex, isConnected, link_info, ioSlot) {
                // 1. 无论发生什么，先打印所有连线变动，看看参数到底长什么样
                // console.log(`[连线变动触发] 节点ID: ${this.id}, type: ${type}, slotIndex: ${slotIndex}, 是否连接: ${isConnected}`);
                // console.log("ioSlot 详情:", ioSlot);

                // 2. 执行原有的逻辑
                if (originalOnConnectionsChange) {
                    originalOnConnectionsChange.apply(this, arguments);
                }
                // 3. type === 1 代表输入端口 (Input)
                if (type === 1 && ioSlot) {
                    // 打印出当前节点所有输入端口的名字，
                    // console.log("当前节点所有输入端口:", this.inputs.map(i => i.name));
                    
                    // 暂时去掉名字限制，只要动了输入端口就提示（排查是不是名字写错了）
                    // console.log(`[实时监听] 输入端口 "${ioSlot.name}" 状态变更为: ${isConnected ? '已连接' : '已断开'}`);
                    if (ioSlot.name === "image_1" || ioSlot.name === "image_2" ) {
                        ioSlot.color_on = isConnected ? "rgb(0, 255, 0)" : null;
                        if (ioSlot.name === "image_1") this.image1connect = isConnected
                        if (ioSlot.name === "image_2") this.image2connect = isConnected

                        requestAnimationFrame(() => {
                            this.updateWidgetsState()
                        });
                        this.setDirtyCanvas(true, true);
                    }       
                }                    
                
            };
            

            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure=function(){
                // 先执行原有的配置逻辑
                if (originalOnConfigure) originalOnConfigure.apply(this, arguments);
                
                // 此时节点的 inputs 已经挂载完毕，可以直接读取真实的连线状态！
                const targetInput1 = this.inputs?.find(i => i.name === "image_1");
                const targetInput2 = this.inputs?.find(i => i.name === "image_2");
                // 如果 targetInput1.link 有值，说明初始状态就是连线的
                this.image1connect = !!(targetInput1 && targetInput1.link != null);
                this.image2connect = !!(targetInput2 && targetInput2.link != null);

                // console.log("节点配置完成，当前真实连线状态为:", this.image1connect);
                // 延迟一点点执行 UI 更新，确保 widgets 已经完全渲染出来
                // setTimeout(() => {
                //     this.updateWidgetsState();
                // }, 20);
                requestAnimationFrame(() => {
                    this.updateWidgetsState()
                });
            }

            
            nodeType.prototype.updateWidgetsState = function() {
                if (!this.widgets) return; 
                
                // console.log("所有控件的真实name：", this.widgets.map(w => w.name));
                const matchSizeWidget = this.widgets.find(w => w.name === "match_size");
                const combo_resolution = this.widgets.find(w => w.name === "resolution");
                const res_flip_widget = this.widgets.find(w => w.name === "flip");
                // console.log("找到的 matchSizeWidget 是：", matchSizeWidget);
                const res_aspectRatio_Widget = this.widgets.find(w => w.name === "aspectRatio");
                const res_megapixels_Widget = this.widgets.find(w => w.name === "megapixels");
                const res_longerEdge_Widget = this.widgets.find(w => w.name === "longerEdge");
                const res_solution_preset_Widget = this.widgets.find(w => w.name === "solution_preset");
                const res_width_Widget = this.widgets.find(w => w.name === "width");
                const res_height_Widget = this.widgets.find(w => w.name === "height");

                const resolutionSwitchsMap_all = [res_aspectRatio_Widget,res_megapixels_Widget,res_longerEdge_Widget,res_solution_preset_Widget,res_width_Widget,res_height_Widget]
                const resolutionSwitchsMap_connect_totalPixels = [res_megapixels_Widget]
                const resolutionSwitchsMap_connect_longerEdge = [res_longerEdge_Widget]
                const resolutionSwitchsMap_connect_solution_preset = [res_solution_preset_Widget]
                const resolutionSwitchsMap_connect_custom = [res_width_Widget, res_height_Widget]
                const resolutionSwitchsMap_totalPixels = [res_aspectRatio_Widget,res_megapixels_Widget]
                const resolutionSwitchsMap_longerEdge = [res_aspectRatio_Widget, res_longerEdge_Widget]
                const resolutionSwitchsMap_solution_preset = [res_solution_preset_Widget]
                const resolutionSwitchsMap_custom = [res_width_Widget, res_height_Widget]
                .map(name => this.widgets.find(w => w.name === name))
                .filter(w => w);
                
                const setWidgetDisabled = (widgets_map,disabled,h_disabled=true) =>{
                    widgets_map.forEach(w=>{
                        if (w) {
                            w.disabled = disabled
                            w.hidden = h_disabled
                        }
                    });
                    this.setSize([this.size[0], this.computeSize()[1]]);
                    this.setDirtyCanvas(true, true);
                }
                
                if (matchSizeWidget) {
                    if (!this.image1connect && !this.image2connect) {                   
                        matchSizeWidget.disabled = true;
                        matchSizeWidget.hidden = true;                    
                        matchSizeWidget.value = false;
                    }else{
                        matchSizeWidget.disabled = false;
                        matchSizeWidget.hidden = false; 
                        res_aspectRatio_Widget.disabled = true
                        res_aspectRatio_Widget.hidden = true                   
                    }
                }
                if (this.image1connect || this.image2connect){
                    const toggle_resolution2_Visibility = (v) => {
                        if (v === true) {                       
                            combo_resolution.disabled = true
                            combo_resolution.hidden = true 
                            // res_flip_widget.disabled = false
                            // res_flip_widget.hidden = false
                            // combo_resolution.value = 'totalPixels'
                            setWidgetDisabled(resolutionSwitchsMap_all,true,true)
                            // setWidgetDisabled(resolutionSwitchsMap_connect_totalPixels,false,false)
                        }else{
                            combo_resolution.disabled = false
                            combo_resolution.hidden = false 
                            // res_flip_widget.disabled = false
                            // res_flip_widget.hidden = false
                            if (combo_resolution.value === 'totalPixels'){
                                setWidgetDisabled(resolutionSwitchsMap_connect_totalPixels,false,false)                                
                            }else if (combo_resolution.value === 'longerEdge'){
                                setWidgetDisabled(resolutionSwitchsMap_connect_longerEdge,false,false)                                
                            }else if (combo_resolution.value === 'preset'){
                                setWidgetDisabled(resolutionSwitchsMap_connect_solution_preset,false,false)                                
                            }else if (combo_resolution.value === 'custom'){
                                setWidgetDisabled(resolutionSwitchsMap_connect_custom,false,false)
                            }
                        }
                    }
                    
                    const originalGetImageSizeCallback = matchSizeWidget.callback;
                    matchSizeWidget.callback = (value) => {
                        originalGetImageSizeCallback?.(value);
                        toggle_resolution2_Visibility(value);
                    };
                    toggle_resolution2_Visibility(matchSizeWidget.value);
                }
                else{                   
                    combo_resolution.disabled = false
                    combo_resolution.hidden = false
                    // res_flip_widget.disabled = false
                    // res_flip_widget.hidden = false
                    
                    setWidgetDisabled(resolutionSwitchsMap_all,true,true)  
                    if (combo_resolution.value === 'totalPixels') {
                        setWidgetDisabled(resolutionSwitchsMap_totalPixels,false,false)  
                    }else if (combo_resolution.value === 'longerEdge') { 
                        setWidgetDisabled(resolutionSwitchsMap_longerEdge,false,false)  
                    }else if (combo_resolution.value === 'preset') { 
                        setWidgetDisabled(resolutionSwitchsMap_solution_preset,false,false)  
                    }else if (combo_resolution.value === 'custom') {
                        setWidgetDisabled(resolutionSwitchsMap_custom,false,false)  
                    }                  
                }
                this.setSize([this.size[0], this.computeSize()[1]]);
                this.setDirtyCanvas(true, true);
                // console.log("====================this.image1connect",this.image1connect)
            };



            // 在注册节点时，给它添加一个原型方法
            // const originalOnResize = nodeType.prototype.onResize;
            const onNodeCreated = nodeType.prototype.onNodeCreated;        
            nodeType.prototype.onNodeCreated = function() {
                // 1. 先执行原有的创建逻辑
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

                // layer skip && fluxguidance && shift && cfgnorm
                const comdo_clip_type = this.widgets.find(w => w.name === "clip_type");
                const layer_skip_Widget = this.widgets.find(w => w.name === "layer_skip");
                const lcm_sampling_Widget = this.widgets.find(w => w.name === "lcm_sampling");
                const lcm_zsnr_Widget = this.widgets.find(w => w.name === "lcm_zsnr");
                const fluxguidance_Widget = this.widgets.find(w => w.name === "fluxguidance");
                const use_kvcache_Widget = this.widgets.find(w => w.name === "use_kvcache");
                const shift_Widget = this.widgets.find(w => w.name === "shift");               
                const cfgNorm_Widget = this.widgets.find(w => w.name === "cfgNorm"); 
                const krea2_rebalance_Widget = this.widgets.find(w => w.name === "krea2_rebalance");               
                // const cper_layer_weights_Widget = this.widgets.find(w => w.name === "cper_layer_weights"); 
                
                const CN_startPercent_Widget = this.widgets.find(w => w.name === "control_startPercent");
                const CN_endPercent_Widget = this.widgets.find(w => w.name === "control_endPercent");

                const comdo_clip_Switchs_Map = [layer_skip_Widget, lcm_sampling_Widget,lcm_zsnr_Widget,
                    fluxguidance_Widget,use_kvcache_Widget, shift_Widget,cfgNorm_Widget,
                    krea2_rebalance_Widget]
                const sdxl_sd_map = [layer_skip_Widget, lcm_sampling_Widget,lcm_zsnr_Widget]
                const flux_map = [fluxguidance_Widget]
                const flux2_map = [fluxguidance_Widget,use_kvcache_Widget]
                const krea2_map = [krea2_rebalance_Widget]
                const chroma_map = [shift_Widget]
                const qwen_map = [shift_Widget,cfgNorm_Widget]
                const CN_Percent_map = [CN_startPercent_Widget,CN_endPercent_Widget]

                const toggle_clip_type_Visibility = (v) => {
                    if (v ==='sdxl' || v === 'stable_diffusion' || v === 'sd3') {
                        setWidgetDisabled(comdo_clip_Switchs_Map,true,true)
                        setWidgetDisabled(sdxl_sd_map,false,false)
                    }else if(v ==='flux') {
                        setWidgetDisabled(comdo_clip_Switchs_Map,true,true)
                        setWidgetDisabled(flux_map,false,false)  
                    }else if(v === 'flux2'){
                        setWidgetDisabled(comdo_clip_Switchs_Map,true,true)
                        setWidgetDisabled(flux2_map,false,false) 
                    }else if(v === 'krea2'){
                        setWidgetDisabled(comdo_clip_Switchs_Map,true,true)
                        setWidgetDisabled(krea2_map,false,false)                      
                    }else if(v ==='qwen_image' || v === 'lumina2' || v === 'omnigen2') {
                        setWidgetDisabled(comdo_clip_Switchs_Map,true,true)
                        setWidgetDisabled(qwen_map,false,false)
                    }else if(v ==='chroma'){
                        setWidgetDisabled(comdo_clip_Switchs_Map,true,true)
                        setWidgetDisabled(chroma_map,false,false)
                    }else {
                        setWidgetDisabled(comdo_clip_Switchs_Map,true,true)
                    }


                    // if (v ==='sdxl' || v === 'stable_diffusion' || v === 'sd3' || v ==='flux' || v === 'qwen_image') {
                    //     setWidgetDisabled(CN_Percent_map,false,false)
                    // }else {
                    //     setWidgetDisabled(CN_Percent_map,true,false)
                    // }

                    this.setSize([this.size[0], this.computeSize()[1]]);
                    this.setDirtyCanvas(true, true);
                };                
                const originalClipTypeCallback = comdo_clip_type.callback;
                comdo_clip_type.callback = (value) => {
                    originalClipTypeCallback?.(value);
                    toggle_clip_type_Visibility(value);
                };                    
                toggle_clip_type_Visibility(comdo_clip_type.value);

                // model2
                const use_model_2_widget = this.widgets.find(w => w.name === "use_model_2");
                const model2_Widget = this.widgets.find(w => w.name === "model_2");
                const lora_2_Widget = this.widgets.find(w => w.name === "lora_2");
                const lora_2_strength_Widget = this.widgets.find(w => w.name === "lora_2_strength"); 
                const use_mod2_Switchs_Map = [model2_Widget, lora_2_Widget, lora_2_strength_Widget]
              
                const toggle_Model_2_Visibility = (show) => {
                    if (show) {
                        setWidgetDisabled(use_mod2_Switchs_Map,false,false)
                    }else{
                        setWidgetDisabled(use_mod2_Switchs_Map,true,true)
                    }
                };
                const originalMod2Callback = use_model_2_widget.callback;
                use_model_2_widget.callback = (value) => {
                    originalMod2Callback?.(value);
                    toggle_Model_2_Visibility(value);
                };                    
                toggle_Model_2_Visibility(use_model_2_widget.value);

                // controlnet
                const bool_CN = this.widgets.find(w => w.name === "use_controlnet");
                const CN_model_Widget = this.widgets.find(w => w.name === "control_model");
                const CN_strength_Widget = this.widgets.find(w => w.name === "control_strength");
                const CN_model2_strength_Widget = this.widgets.find(w => w.name === "control_strength_model_2");  
                const CN_Switchs_Map = [CN_model_Widget, CN_strength_Widget,CN_startPercent_Widget,CN_endPercent_Widget,CN_model2_strength_Widget]
                const toggle_CN_2_Visibility = (show) => {
                    if (show) {
                        setWidgetDisabled(CN_Switchs_Map,false,false)
                    }else{
                         setWidgetDisabled(CN_Switchs_Map,true,true)
                    }
                };
                const originalCNCallback = bool_CN.callback;
                bool_CN.callback = (value) => {
                    originalCNCallback?.(value);
                    toggle_CN_2_Visibility(value);
                };                    
                toggle_CN_2_Visibility(bool_CN.value);

                // neg zero 
                const neg_zero_Widget = this.widgets.find(w => w.name === "neg_zero");  
                const pos_Widget = this.widgets.find(w => w.name === "pos");  
                const neg_Widget = this.widgets.find(w => w.name === "neg"); 
                pos_Widget.label = 'pos-t5xxl'
                neg_Widget.label = 'neg-clip_l'
                

                const neg_zero_map = [neg_Widget]
                const toggle_neg_zero_Visibility = (show) => {
                    if (show) {
                        setWidgetDisabled(neg_zero_map,true,false)
                    }else{
                        setWidgetDisabled(neg_zero_map,false,false)
                    }
                };
                const originalNegZeroCallback = neg_zero_Widget.callback;
                neg_zero_Widget.callback = (value) => {
                    originalNegZeroCallback?.(value);
                    toggle_neg_zero_Visibility(value);
                };                    
                toggle_neg_zero_Visibility(neg_zero_Widget.value);


                // resolution                
                const matchSizeWidget = this.widgets.find(w => w.name === "match_size");
                const combo_resolution = this.widgets.find(w => w.name === "resolution");
                const res_flip_widget = this.widgets.find(w => w.name === "flip");
                const res_aspectRatio_Widget = this.widgets.find(w => w.name === "aspectRatio");
                const res_megapixels_Widget = this.widgets.find(w => w.name === "megapixels");
                const res_longerEdge_Widget = this.widgets.find(w => w.name === "longerEdge");
                const res_solution_preset_Widget = this.widgets.find(w => w.name === "solution_preset");
                const res_width_Widget = this.widgets.find(w => w.name === "width");
                const res_height_Widget = this.widgets.find(w => w.name === "height");

                const resolutionSwitchsMap_all = [res_aspectRatio_Widget,res_megapixels_Widget,res_longerEdge_Widget,res_solution_preset_Widget,res_width_Widget,res_height_Widget]
                const resolutionSwitchsMap_connect_totalPixels = [res_megapixels_Widget]
                const resolutionSwitchsMap_connect_longerEdge = [res_longerEdge_Widget]
                const resolutionSwitchsMap_connect_solution_preset = [res_solution_preset_Widget]
                const resolutionSwitchsMap_connect_custom = [res_width_Widget, res_height_Widget]
                const resolutionSwitchsMap_totalPixels = [res_aspectRatio_Widget,res_megapixels_Widget]
                const resolutionSwitchsMap_longerEdge = [res_aspectRatio_Widget, res_longerEdge_Widget]
                const resolutionSwitchsMap_solution_preset = [res_solution_preset_Widget]
                const resolutionSwitchsMap_custom = [res_width_Widget, res_height_Widget]

                setTimeout(() => {
                    this.updateWidgetsState();
                }, 0);

                const toggle_resolution_Visibility = (v) => {
                    setWidgetDisabled(resolutionSwitchsMap_all,true,true)  
                    if (!this.image1connect && !this.image2connect && matchSizeWidget.value === false){
                        if (v === 'totalPixels') {
                            setWidgetDisabled(resolutionSwitchsMap_totalPixels,false,false)  
                        }else if (v === 'longerEdge') {  
                            setWidgetDisabled(resolutionSwitchsMap_longerEdge,false,false)  
                        }else if (v === 'preset') {  
                            setWidgetDisabled(resolutionSwitchsMap_solution_preset,false,false) 
                        }else if (v === 'custom') {
                            setWidgetDisabled(resolutionSwitchsMap_custom,false,false) 
                        }
                    }else{
                        if (v === 'totalPixels') {
                            setWidgetDisabled(resolutionSwitchsMap_connect_totalPixels,false,false)  
                        }else if (v === 'longerEdge') {  
                            setWidgetDisabled(resolutionSwitchsMap_connect_longerEdge,false,false) 
                        }else if (v === 'preset') {  
                            setWidgetDisabled(resolutionSwitchsMap_connect_solution_preset,false,false)  
                        }else if (v === 'custom') {
                            setWidgetDisabled(resolutionSwitchsMap_connect_custom,false,false) 
                        }
                    }
                };
                const originalResolutionCallback = combo_resolution.callback;
                combo_resolution.callback = (value) => {
                    originalResolutionCallback?.(value);
                    toggle_resolution_Visibility(value);
                };                    
                toggle_resolution_Visibility(combo_resolution.value);


                
                // ================== getImageSize切换逻辑 ==================
                // 1. 监听连线的插拔状态 (前端视觉层面)
                            
                const match_map = [matchSizeWidget]
                if (matchSizeWidget){
                    matchSizeWidget.label = '🚩match to image1 size'
                    setWidgetDisabled(match_map,true,true)
                    // console.log("[节点初始化] 已记录基准字典:", matchSizeWidget,'matchSize2Widget[value]',matchSizeWidget.value);
                }


                // ================== 保存按钮逻辑 ==================
                const saveBtn = this.addWidget("button", "💾 save current to preset", "save", () => {
                    const currentVars = {};
                    let clipTypeValue = null;

                    this.widgets.forEach(widget => {
                        // 排除不需要保存的控件
                        // if (widget.type !== "button" && widget.name && widget.name !== "preset" && widget.name !== "clip_type") {
                        if (widget.type !== "button" && widget.name && widget.name !== "preset") {
                            if (widget.value === undefined || widget.value === null) return;

                            const defaultValue = this.defaultValuesMap[widget.name];
                            const currentValue = widget.value;

                            // 统一处理前后端的空值：把 null, undefined, "None", "" 都视为“空”
                            const isCurrentEmpty = (currentValue === null || currentValue === undefined || currentValue === "None" || currentValue === "");
                            const isDefaultEmpty = (defaultValue === null || defaultValue === undefined || defaultValue === "None" || defaultValue === "");

                            // 如果两者都是“空”，视为相等，不保存
                            if (isCurrentEmpty && isDefaultEmpty) return;

                            // 严格对比，只有当前值不等于基准值时才保存
                            if (currentValue !== defaultValue) {
                                currentVars[widget.name] = currentValue;
                            }
                        } 
                        // 单独提取 clip_type
                        // else if (widget.name === "clip_type") {
                        //     clipTypeValue = widget.value;
                        // }
                    });
                    // console.log("[前端] 最终过滤后保存的变量字典:", currentVars);
                    // console.log("[前端] 单独获取的 clip_type:", clipTypeValue);

                    fetch("/aftools/update_params?action=save", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ variables: currentVars })
                        // body: JSON.stringify({ variables: currentVars, clip_type: clipTypeValue })
                    })
                    .then(res => res.json())
                    .then(data => {
                        if (data && data.message) app.ui.dialog.show("✅ " + data.message);
                    })
                    .catch(err => {
                        console.error("[前端错误] 保存请求失败:", err);
                        app.ui.dialog.show("❌ 保存失败，请查看控制台。");
                    });
                });
                saveBtn.serialize = false;

                // ================== 打开预设目录按钮 ==================
                if (!this.widgets.find(w => w.name === "open_folder_btn")) {
                    // 动态获取你的插件根目录下的 preset/param 路径
                    // 假设你的插件文件夹名叫 'my_comfy_plugin'，你可以根据实际情况修改
                    const pluginFolderName = "afar_tools"; 
                    const targetPath = `custom_nodes/${pluginFolderName}/presets`;

                    const openFolderBtn = this.addWidget("button", "📂 open the preset dir", "open_folder", () => {
                        console.log("[前端] 请求打开文件夹:", targetPath);
                        
                        fetch(`/aftools/open_folder?path=${encodeURIComponent(targetPath)}`)
                            .then(res => res.json())
                            .then(data => {
                                if (data && data.status === "success") {
                                    console.log("[前端] 后端已成功触发打开文件夹");
                                } else {
                                    console.error("[前端] 打开文件夹失败:", data.error);
                                    app.ui.dialog.show("❌ 打开文件夹失败: " + data.error);
                                }
                            })
                            .catch(err => {
                                console.error("[前端错误] 请求打开文件夹失败:", err);
                                app.ui.dialog.show("❌ 请求失败，请查看控制台。");
                            });
                    });
                    openFolderBtn.serialize = false; // 防止按钮状态被保存到工作流中
                }
                

                // 同步外部preset
                // ================== 预设切换逻辑 ==================
                //【基准字典】记录节点刚创建时的所有初始值
                this.defaultValuesMap = {};
                this.widgets.forEach(widget => {
                    if (widget.name && widget.type !== "button") {
                        this.defaultValuesMap[widget.name] = JSON.parse(JSON.stringify(widget.value));
                    }
                });
                // console.log("[节点初始化] 已记录基准字典:", this.defaultValuesMap);
                const presetWidget = this.widgets.find(w => w.name === "preset");
                if (presetWidget) {
                    const originalCallback = presetWidget.callback;
                    presetWidget.callback = () => {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        const selectedPreset = presetWidget.value;
                        if (!selectedPreset || selectedPreset === "undefined") return;
                        fetch(`/aftools/update_params?preset=${encodeURIComponent(selectedPreset)}`)
                            .then(res => res.ok ? res.text() : Promise.reject(`状态码: ${res.status}`))
                            .then(text => {
                                if (!text) throw new Error("后端返回了空内容！");
                                const data = JSON.parse(text);
                                // console.log("===原数据",data)
                                for (const [key, value] of Object.entries(data)) {
                                    if (key === "skip_ksample") continue; //跳过skip_ksample
                                    const targetWidget = this.widgets.find(w => w.name === key);
                                    if (targetWidget) {
                                        // 如果控件原本的值是数字或布尔值，就把后端传来的字符串强行转回去
                                        let finalValue = value;
                                        if (typeof targetWidget.value === 'number' && typeof value === 'string') {
                                            finalValue = Number(value); // 转回数字
                                        } else if (typeof targetWidget.value === 'boolean' && typeof value === 'string') {
                                            finalValue = (value === 'true'); // 转回布尔值
                                        }
                                        // 赋值
                                        targetWidget.value = finalValue;        
                                        // console.log("前端读取后端dict更新UI值",key, value," -> ",finalValue) 
                                        if (targetWidget.callback) {
                                            targetWidget.callback(finalValue);
                                        }
                                    }
                                }
                                setTimeout(() => {
                                    this.setDirtyCanvas(true, true); // 这里的 this 指向节点实例，比 this.graph 更直接
                                    app.graph.setDirtyCanvas(true, true); // 双重保险，通知整个画布刷新
                                }, 10); 
                            })
                            .catch(err => console.error("[前端错误] 预设加载失败:", err));
                    }
                }
            }
        }
        // lora stack
        else if (nodeData.name === "pipe_loras_pack"){ 
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(this, arguments);
                }
                // 封装一个函数，专门用来处理某一组 LoRA 的开关逻辑
                const setupLoraGroup = (index) => {
                    // 使用模板字符串动态拼接出对应的 widget 名字
                    const loraOnWidget = this.widgets.find(w => w.name === `lora_${index}`);
                    const loraWidget = this.widgets.find(w => w.name === `lora_${index}_model`);
                    const loraStrengthWidget = this.widgets.find(w => w.name === `lora_${index}_strength`);

                    // 切换禁用/启用状态的逻辑
                    const toggleLoraVisibility = (show) => {
                        if (loraWidget && loraStrengthWidget) {
                            loraWidget.hidden = !show;
                            loraStrengthWidget.hidden = !show;
                            loraWidget.disabled = !show;
                            loraStrengthWidget.disabled = !show;
                            this.setSize([this.size[0], this.computeSize()[1]]);
                        }
                    };
                    // 绑定开关的回调事件
                    if (loraOnWidget) {
                        const originalCallback = loraOnWidget.callback;
                        loraOnWidget.callback = (value) => {
                            originalCallback?.(value);
                            toggleLoraVisibility(value);
                        };  
                        // 初始化时触发一次，保证状态同步
                        toggleLoraVisibility(loraOnWidget.value);
                    }
                };
                // 使用循环批量处理 10 组 LoRA 参数
                for (let i = 1; i <= 10; i++) {
                    setupLoraGroup(i);
                }
                this.setSize([this.size[0], this.computeSize()[1]]);
                setTimeout(() => {
                    this.setDirtyCanvas(true, true); // 这里的 this 指向节点实例，比 this.graph 更直接
                    app.graph.setDirtyCanvas(true, true); // 双重保险，通知整个画布刷新
                }, 10);
            };
        }
    }
})
