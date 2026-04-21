# 此文件用来方便复制粘贴命令到终端执行

# # 3. 训练完成后推理（自动使用 EMA 权重）
# python -m tools.sample_ddpm_cond_hemit --config config/hemit_phikon.yaml --full-image --num-samples 8

# # 4. 评估
# python -m tools.evaluate_hemit --config config/hemit_phikon.yaml

# 第二步：正常启动训练（自动检测到缓存，跳过 phikon 模型加载）
torchrun --nproc_per_node=2 -m tools.train_ddpm_cond_hemit --config config/hemit_phikon_sc.yaml

# 第三步：训练完成后推理（自动使用 EMA 权重）
python -m tools.sample_ddpm_cond_hemit --config config/hemit_phikon_sc.yaml --full-image --num-samples 8

# 第四步：评估
python -m tools.evaluate_hemit --config config/hemit_phikon_sc.yaml

# phikon_img_cond
torchrun --nproc_per_node=2 -m tools.train_ddpm_cond_hemit --config config/hemit_phikon_img.yaml

# 推理
python -m tools.sample_ddpm_cond_hemit --config config/hemit_phikon_img.yaml --full-image --num-samples 8

# 评估
python -m tools.evaluate_hemit --config config/hemit_phikon_img.yaml
