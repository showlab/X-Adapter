python inference.py --plugin_type "controlnet_tile" \
--prompt "best quality, extremely datailed" \
--controlnet_condition_scale_list 1.0  \
--adapter_guidance_start_list 0.7 \
--adapter_condition_scale_list 1.2 \
--input_image_path "./assets/Dog.png" \
--height 1024 \
--width 768 \
--height_sd1_5 512 \
--width_sd1_5 384 \
