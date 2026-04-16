# 此文件用来方便复制粘贴命令到终端执行

# # 3. 训练完成后推理（自动使用 EMA 权重）
# python -m tools.sample_ddpm_cond_hemit --config config/hemit_phikon.yaml --full-image --num-samples 8

# # 4. 评估
# python -m tools.evaluate_hemit --config config/hemit_phikon.yaml

# 第一步：提取并缓存 phikon 特征（只需执行一次）
python tools/cache_encoder_features.py --config config/hemit_phikon_sc.yaml --splits train --batch_size 64

# 特征保存到：
hemit_shared_artifacts/phikon_features/train_features.pkl

# 第二步：正常启动训练（自动检测到缓存，跳过 phikon 模型加载）
torchrun --nproc_per_node=2 tools/train_ddpm_cond_hemit.py --config config/hemit_phikon_sc.yaml