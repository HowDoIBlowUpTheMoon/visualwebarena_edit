python run.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --test_start_idx 0 \
  --test_end_idx 3 \
  --result_dir $HOME/test_GPT4_classifieds \
  --test_config_base_dir=config_files/test_classifieds \
  --model gpt-4-vision-preview \
  --action_set_tag som  --observation_type image_som
