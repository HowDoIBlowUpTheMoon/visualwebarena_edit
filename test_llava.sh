python run.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --result_dir $HOME/test \
  --test_config_base_dir=config_files/test_classifieds \
  --model liuhaotian/llava-v1.5-13b \
  --action_set_tag som  --observation_type image_som