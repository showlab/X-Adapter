python inference.py --plugin_type "lora" \
--prompt "masterpiece, best quality, ultra detailed, 1 girl , solo, smile, looking at viewer, holding flowers"  \
--prompt_sd1_5 "masterpiece, best quality, ultra detailed, 1 girl, solo, smile, looking at viewer, holding flowers, shuimobysim, wuchangshuo, bonian, zhenbanqiao, badashanren" \
--adapter_guidance_start_list 0.95 \
--adapter_condition_scale_list 1.50 \
--seed 3943946911 \
--height 1024 \
--width 1024 \
--height_sd1_5 512 \
--width_sd1_5 512 \
